from collections import deque

import importlib
import io
import numpy as np
import os
from abc import abstractmethod
from contextlib import redirect_stdout
from jax.nn import softmax
from jax.scipy.special import logsumexp
from stlpy.STL import LinearPredicate as baseLinearPredicate, STLTree
from typing import TypeVar, NamedTuple

os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
import ds.utils as ds_utils

importlib.reload(ds_utils)  # Reload the module to change the backend

with redirect_stdout(io.StringIO()):
    pass

import logging

colored, HARDNESS, IMPLIES_TRICK, set_hardness = ds_utils.colored, ds_utils.HARDNESS, ds_utils.IMPLIES_TRICK, ds_utils.set_hardness

# Default soft-min/max approximation for AND/OR/ALWAYS/EVENTUALLY/UNTIL reductions.
# Overridable per-call via STL.eval(..., approx_method=...) or globally by assigning
# ds.stl_jax.APPROX_METHOD before tracing. Options:
#   "softmax"   - legacy softmax-weighted-sum (default; biased, can plateau at the boundary)
#   "logsumexp" - mass-conserving smooth max (no vanishing-gradient plateau)
#   "true"      - exact jnp.max/min (gradient distributed across argmax ties)
APPROX_METHOD = "softmax"
outside_npy = ds_utils.outside_rectangle_formula
inside_npy = ds_utils.inside_rectangle_formula

# Replace with JAX
import jax.numpy as jnp
import jax
import re


class PredicateBase(NamedTuple):
    name: int

    def eval_at_t(self, path: jnp.ndarray, t: int = 0, train_mode: bool = False) -> jnp.ndarray:
        return self.eval_whole_path(path, t, t + 1, train_mode=train_mode)[:, 0]

    @abstractmethod
    def eval_whole_path(
            self, path: jnp.ndarray, start_t: int = 0, end_t: int = None, train_mode: bool = False
    ) -> jnp.ndarray:
        """Stick to JAX when possible."""
        raise NotImplementedError

    @abstractmethod
    def get_stlpy_form(self) -> STLTree:
        """Use Numpy to ensure compatibility with STLpy."""
        raise NotImplementedError

    def __str__(self) -> str:
        # TODO: Get a better mapping to handle obs and goal
        return f"Goal {self.name}"

    def __lt__(self, other: "PredicateBase") -> bool:
        """Sort predicates by name."""
        return self.name < other.name


class RectangularPredicate(NamedTuple):
    """
        Rectangle reachability predicate
        """

    cent: np.ndarray
    size: np.ndarray
    name: int
    shrink_factor: float = 1.0  # shrink (for reach) or expand (for avoid) the rectangle to make it more conservative

    @property
    def size_tensor(self):
        return ds_utils.default_tensor(self.size)

    @property
    def cent_tensor(self):
        return ds_utils.default_tensor(self.cent)

    def eval_at_t(self, path: jnp.ndarray, t: int = 0, train_mode: bool = False) -> jnp.ndarray:
        return self.eval_whole_path(path, t, t + 1, train_mode=train_mode)[:, 0]

    @abstractmethod
    def eval_whole_path(
            self, path: jnp.ndarray, start_t: int = 0, end_t: int = None,
            train_mode: bool = False
    ) -> jnp.ndarray:
        """Stick to JAX when possible."""
        raise NotImplementedError

    @abstractmethod
    def get_stlpy_form(self) -> STLTree:
        """Use Numpy to ensure compatibility with STLpy."""
        raise NotImplementedError

    def __hash__(self):
        return hash(f"{self.cent},{self.size}")

    def __eq__(self, other):
        if not isinstance(other, RectangularPredicate):
            return False
        # return self.cent == other.cent and self.size == other.size
        # Above using float difference
        return np.allclose(self.cent, other.cent) and np.allclose(self.size, other.size)

    def __str__(self) -> str:
        return f"Goal {self.name}"

    def __lt__(self, other: "RectangularPredicate") -> bool:
        """Sort predicates by center (prioritizing y). To get consistent ordering."""
        return (self.cent[1] < other.cent[1]) or (
                np.allclose(self.cent[1], other.cent[1]) and self.cent[0] < other.cent[0])

    def __rich_repr__(self):
        # Assumes that size is common and not important
        yield f"{self.cent}"


# PREDICATE_FORM = TypeVar("PREDICATE_FORM", RectangularPredicate, PredicateBase)
PREDICATE_TYPES = (RectangularPredicate, PredicateBase)
SHRINK_NORM = 2  # Conservative norm used : 1 | 2 | jnp.inf

import jax
@jax.jit
def softnorm(x):
    """Compute the 2-norm, but if x is too small replace it with the squared 2-norm
    to make sure it's differentiable. This function is continuous and has a derivative
    that is defined everywhere, but its derivative is discontinuous.
    """
    eps = 1e-5
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)

# Try importing
try:
    import architect.components.specifications.stl as architect_stl
except ImportError:
    print("architect-rss-22 not found")
    architect_stl = None


class RectReachPredicate(RectangularPredicate):
    """
    Rectangle reachability predicate
    """

    def eval_whole_path(
            self, path: jnp.array, start_t: int = 0, end_t: int = None, train_mode: bool = False
    ) -> jnp.array:
        """Stick to JAX when possible."""
        assert len(path.shape) == 3, "motion must be in batch"
        eval_path = path[:, start_t:end_t]

        def with_shrink(_eval_path):
            """L-SHRINK_NORM Norm version for conservative evaluation"""
            if SHRINK_NORM == 2:
                # Multiply this factor to get an inner circle matching reach
                shrink_multiplier = 1 / jnp.sqrt(2)
            else:
                shrink_multiplier = 1
            return jnp.linalg.norm(self.size_tensor * self.shrink_factor * shrink_multiplier / 2,
                                   ord=SHRINK_NORM) - jnp.linalg.norm(
                _eval_path - self.cent_tensor, axis=-1, ord=SHRINK_NORM)
            # # Adding shrink factor to make it more conservative
            # # self.size_tensor * ( 0.2  / 2) - jnp.abs(_eval_path - self.cent_tensor), axis=-1
            # # jnp.linalg.norm(self.size_tensor * 0.6 / 2) - jnp.linalg.norm(_eval_path - self.cent_tensor, axis=-1)
            # axis=-1
            # # (self.size_tensor * 0.7 / 2) ** 2 - jnp.square(_eval_path - self.cent_tensor), axis=-1

            # jnp.min(
            #     # Adding shrink factor to make it more conservative
            #     self.size_tensor * ( 0.2  / 2) - jnp.abs(_eval_path - self.cent_tensor), axis=-1
            #     # jnp.linalg.norm(self.size_tensor * 0.6 / 2) - jnp.linalg.norm(_eval_path - self.cent_tensor, axis=-1),
            #     # axis=-1
            #     # (self.size_tensor * 0.7 / 2) ** 2 - jnp.square(_eval_path - self.cent_tensor), axis=-1
            # )

        def without_shrink(_eval_path):
            return jnp.min(
                self.size_tensor / 2 - jnp.abs(_eval_path - self.cent_tensor), axis=-1
            )

        res = jax.lax.cond(train_mode, with_shrink, without_shrink, eval_path)

        return res

    def get_stlpy_form(self) -> STLTree:
        """Use Numpy to ensure compatibility with STLpy."""
        bounds = np.stack(
            [self.cent - self.size * self.shrink_factor / 2, self.cent + self.size * self.shrink_factor / 2]
        ).T.flatten()
        return inside_npy(bounds, 0, 1, 2, self.name)


    def get_architect_form(self) -> STLTree:
        """Use Numpy to ensure compatibility with STLpy."""
        min_waiting_radius  = 0.5 * self.shrink_factor
        return architect_stl.STLPredicate(lambda q_t: -softnorm(q_t[:2] - self.cent), min_waiting_radius)
        # return inside_npy(bounds, 0, 1, 2, self.name)


class RectAvoidPredicate(RectangularPredicate):
    """
    Rectangle avoidance predicate
    """

    def eval_whole_path(
            self, path: jnp.array, start_t: int = 0, end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        """Stick to JAX when possible."""
        assert len(path.shape) == 3, "motion must be in batch"
        eval_path = path[:, start_t:end_t]

        def with_shrink(_eval_path):
            return jnp.max(
                # Adding shrink factor to make it more conservative
                jnp.square(_eval_path - self.cent_tensor) - (self.size_tensor * (2 - self.shrink_factor) / 2) ** 2,
                axis=-1
            )

        def without_shrink(_eval_path):
            return jnp.max(
                # Adding shrink factor to make it more conservative
                jnp.abs(eval_path - self.cent_tensor) - self.size_tensor / 2, axis=-1
            )

        res = jax.lax.cond(train_mode, with_shrink, without_shrink, eval_path)

        return res

    def __str__(self) -> str:
        return f"Obs {self.name}"

    def get_stlpy_form(self) -> STLTree:
        """Use Numpy to ensure compatibility with STLpy."""
        bounds = np.stack(
            [self.cent - self.size * (2 - self.shrink_factor) / 2, self.cent + self.size * (2 - self.shrink_factor) / 2]
        ).T.flatten()
        return outside_npy(bounds, 0, 1, 2, self.name)


def inside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_rectangle:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = jnp.zeros((1, d))
    a1.at[:, y1_index].set(1)
    right = LinearPredicate(a1, y1_min)
    left = LinearPredicate(-a1, -y1_max)

    a2 = jnp.zeros((1, d))
    a2.at[:, y2_index].set(1)
    top = LinearPredicate(a2, y2_min)
    bottom = LinearPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        inside_rectangle.__str__ = lambda: name
        inside_rectangle.__repr__ = lambda: name

    return inside_rectangle


def outside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = jnp.zeros((1, d))
    a1.at[:, y1_index].set(1)
    right = LinearPredicate(a1, y1_max)
    left = LinearPredicate(-a1, -y1_min)

    a2 = jnp.zeros((1, d))
    a2.at[:, y2_index].set(1)
    top = LinearPredicate(a2, y2_max)
    bottom = LinearPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        outside_rectangle.__str__ = lambda: name
        outside_rectangle.__repr__ = lambda: name

    return outside_rectangle


class LinearPredicate(baseLinearPredicate):
    """
    A linear STL predicate :math:`\pi` defined by

    .. math::

        a^Ty_t - b \geq 0

    where :math:`y_t \in \mathbb{R}^d` is the value of the signal
    at a given timestep :math:`t`, :math:`a \in \mathbb{R}^d`,
    and :math:`b \in \mathbb{R}`.

    :param a:       a jax numpy array or list representing the vector :math:`a`
    :param b:       a list, jax numpy array, or scalar representing :math:`b`
    :param name:    (optional) a string used to identify this predicate.
    """

    def __init__(self, a, b, name=None):
        # Convert provided constraints to numpy arrays
        self.a = jnp.asarray(a).reshape((-1, 1))
        self.b = jnp.atleast_1d(b)

        # Some dimension-related sanity checks
        assert (self.a.shape[1] == 1), "a must be of shape (d,1)"
        assert (self.b.shape == (1,)), "b must be of shape (1,)"

        # Store the dimensionality of y_t
        self.d = self.a.shape[0]

        # A unique string describing this predicate
        self.name = name


AST = TypeVar("AST", list, PredicateBase)

# ---------------------------------------------------------------------------------
# OPERATOR MAPPINGS
# ---------------------------------------------------------------------------------
OP_SYMBOLS = {
    "~": 0,  # NOT
    "&": 1,  # AND
    "|": 2,  # OR
    "U": 3,  # UNTIL
    "G": 4,  # ALWAYS
    "F": 5,  # EVENTUALLY
    "->": 6,  # IMPLIES
}

# For debugging or printing back the operator from the integer code
OP_SYMBOLS_INV = {v: k for k, v in OP_SYMBOLS.items()}

import functools as ft


def list_to_tuple(x):
    if isinstance(x, list):
        return tuple(list_to_tuple(e) for e in x)
    return x


# Shared by ds.ma_stl_jax (imported via `from .stl_jax import *`); its jitted _eval
# methods have the original 6/7-arg signatures, so these must stay as-is.
STATIC_ARGNUMS_UNARY = (0, 1, 3, 4, 5)
STATIC_ARGNUMS_BINARY = (0, 1, 2, 4, 5, 6)

# stl_jax-only: eval fns additionally take ..., hardness (dynamic/traced), approx_method (static).
# UNARY_AM  sig: (self, ast,  path, start_t, end_t, train_mode, hardness, approx_method)
#                  0    1     2     3        4      5           6         7
# BINARY_AM sig: (self, sf1, sf2,  path, start_t, end_t, train_mode, hardness, approx_method)
#                  0    1    2     3     4        5      6           7         8
# hardness stays dynamic (vary per step, no retrace); approx_method is static.
STATIC_ARGNUMS_UNARY_AM = (0, 1, 3, 4, 5, 7)
STATIC_ARGNUMS_BINARY_AM = (0, 1, 2, 4, 5, 6, 8)


class STL:
    """
    Class for representing STL formulas.

    All methods are functionally pure with no side effects during execution.
    """

    def __init__(self, ast: AST):
        # self.ast = ast
        single_operators = ("~", "G", "F")
        binary_operators = ("&", "|", "->", "U")
        sequence_operators = ("G", "F", "U")
        self.single_operators = tuple(OP_SYMBOLS[op] for op in single_operators)
        self.binary_operators = tuple(OP_SYMBOLS[op] for op in binary_operators)
        self.sequence_operators = tuple(OP_SYMBOLS[op] for op in sequence_operators)
        self.time_bounded_operators = self.sequence_operators
        self.stlpy_form = None
        self.expr_repr = None
        self.end_t = None  # Populated when evaluating
        self.logger = logging.getLogger(__name__)

        # Recursively transform the user-provided AST so that any string operator
        # is replaced by its numeric code.
        self.ast = ast
        self.tuple_ast = list_to_tuple(ast)

    def _preprocess_ast(self, ast):
        """Preprocess the AST like flattening AND/OR chains."""
        raise NotImplementedError("Fill in the preprocessing logic to flatten AND/OR chains.")

    def _transform_ast(self, node):
        """Recursively walk the AST and replace any string operator with its numeric code."""
        if self._is_leaf(node):
            # Leaves (TaskBase, etc.) remain unchanged
            return node

        op = node[0]
        # If operator is a string (like "G", "U", etc.) then map it to integer
        if isinstance(op, str) and op in OP_SYMBOLS:
            op_code = OP_SYMBOLS[op]

            # Single-operator forms, e.g. NOT ("~")
            if op_code == OP_SYMBOLS["~"]:
                return [op_code, self._transform_ast(node[1])]

            # Two-operand forms that also might have time windows:
            # e.g. "G", "F" => shape: [ "G", sub_form, start, end ]
            # e.g. "U" => [ "U", sub_form1, sub_form2, start, end ]
            # e.g. "&", "|", "->" => [ op, sub_form1, sub_form2 ]
            if op_code in (OP_SYMBOLS["G"], OP_SYMBOLS["F"]):
                # time-bounded unary operator
                return [
                    op_code,
                    self._transform_ast(node[1]),
                    node[2],
                    node[3],
                ]
            elif op_code == OP_SYMBOLS["U"]:
                return [
                    op_code,
                    self._transform_ast(node[1]),
                    self._transform_ast(node[2]),
                    node[3],
                    node[4],
                ]
            else:
                # e.g. "&", "|", "->"
                return [
                    op_code,
                    self._transform_ast(node[1]),
                    self._transform_ast(node[2]),
                ]
        else:
            # Possibly already transformed, or an unknown operator
            # If it's a list of length >= 2, we still attempt recursion
            if isinstance(node, list) or isinstance(node, tuple):
                # Recursively transform each child that might be an operator
                transformed_children = []
                # The first element is either an already replaced op or something else
                transformed_children.append(node[0])
                for child in node[1:]:
                    transformed_children.append(self._transform_ast(child))
                return transformed_children

        return node

    """
    Syntax Functions
    """

    def __and__(self, other: "STL") -> "STL":
        ast = [OP_SYMBOLS["&"], self.ast, other.ast]
        return STL(ast)

    def __or__(self, other: "STL") -> "STL":
        ast = [OP_SYMBOLS["|"], self.ast, other.ast]
        return STL(ast)

    def __invert__(self) -> "STL":
        ast = [OP_SYMBOLS["~"], self.ast]
        return STL(ast)

    def implies(self, other: "STL") -> "STL":
        ast = [OP_SYMBOLS["->"], self.ast, other.ast]
        return STL(ast)

    def eventually(self, start: int, end: int):
        ast = [OP_SYMBOLS["F"], self.ast, start, end]
        return STL(ast)

    def always(self, start: int, end: int) -> "STL":
        ast = [OP_SYMBOLS["G"], self.ast, start, end]
        return STL(ast)

    def until(self, other: "STL", start: int, end: int) -> "STL":
        ast = [OP_SYMBOLS["U"], self.ast, other.ast, start, end]
        return STL(ast)

    def eval(self, path: jnp.array, t: int = 0, train_mode: bool = False,
             hardness=None, approx_method: str = None) -> jnp.array:
        """Evaluate the formula at time t.

        :param path:            The motion path to evaluate the formula on.
        :param train_mode:   Whether to evaluate in training mode (conservative with shrink factor).
        :param t:               The time step to evaluate the formula at.
        :param hardness:        Soft-min/max temperature for this call. None -> module HARDNESS.
                                Pass a traced scalar to vary per step (e.g. per denoising step)
                                without retracing.
        :param approx_method:   Soft-min/max approximation ("softmax"/"logsumexp"/"true") for this
                                call. None -> module APPROX_METHOD. Static (a change retraces once).
        """
        # Resolve defaults at the Python level so the (possibly traced) hardness and the
        # static approx_method flow as explicit args into the jitted eval region. None passed
        # down would bake the module globals (the legacy behaviour) instead.
        if hardness is None:
            hardness = HARDNESS
        if approx_method is None:
            approx_method = APPROX_METHOD
        return self._eval(self.tuple_ast, path, t, train_mode=train_mode,
                          hardness=hardness, approx_method=approx_method)

    def eval_train(self, path: jnp.array, t: int = 0,
                   hardness=None, approx_method: str = None) -> jnp.array:
        """To help prevent recompilation in jax.jit, we separate the training mode evaluation."""
        return self.eval(path, t, train_mode=True,
                         hardness=hardness, approx_method=approx_method)

    def end_time(self) -> int:
        """Get the end time of the formula efficiently."""
        if self.end_t is None:
            # Evaluate the formula to get the end time
            # Get max of binary tree at self.ast
            self.end_t = self._get_end_time(self.ast)
        return self.end_t

    def _get_end_time(self, ast: AST) -> int:
        """Get max time of the formula. Runs in O(n) time where n is the number of nodes. Runs once then memoizes."""
        if self._is_leaf(ast):
            return 0
        op = ast[0]
        if op == OP_SYMBOLS["G"]:
            # add end time from inner formula
            return ast[-1] + self._get_end_time(ast[1])
        elif op in self.sequence_operators:
            # The last two elements are the start and end times
            return ast[-1]
        elif op == OP_SYMBOLS["~"]:
            return self._get_end_time(ast[1])
        # Is binary operator
        return max(self._get_end_time(ast[1]), self._get_end_time(ast[2]))

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY_AM)
    def _eval(
            self,
            ast: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        if self._is_leaf(ast):
            return ast.eval_at_t(path, start_t, train_mode=train_mode)

        op_code = ast[0]
        if op_code in self.sequence_operators:
            # NOTE: Overwrite start_t and end_t
            start_t, end_t = start_t + ast[-2], start_t + ast[-1]
            if end_t > path.shape[1]:
                self.logger.warning("end_t is larger than motion length")

        kw = dict(train_mode=train_mode, hardness=hardness, approx_method=approx_method)
        if op_code == OP_SYMBOLS["&"]:
            return self._eval_and(ast[1], ast[2], path, start_t, end_t, **kw)
        elif op_code == OP_SYMBOLS["|"]:
            return self._eval_or(ast[1], ast[2], path, start_t, end_t, **kw)
        elif op_code == OP_SYMBOLS["~"]:
            return self._eval_not(ast[1], path, start_t, end_t, **kw)
        elif op_code == OP_SYMBOLS["->"]:
            return self._eval_implies(ast[1], ast[2], path, start_t, end_t, **kw)
        elif op_code == OP_SYMBOLS["G"]:
            return self._eval_always(ast[1], path, start_t, end_t, **kw)
        elif op_code == OP_SYMBOLS["F"]:
            return self._eval_eventually(ast[1], path, start_t, end_t, **kw)
        elif op_code == OP_SYMBOLS["U"]:
            return self._eval_until(ast[1], ast[2], path, start_t, end_t, **kw)

        raise ValueError(f"Unknown operator {ast[0]}")

    def _flatten_and_or(self, cast: AST, flat_and=True) -> list[AST]:
        """
        Recursively collect every subformula under an & chain into a single list.
        For example, if cast = ['&', A, ['&', B, C]], then
        _flatten_and(cast) = [A, B, C].
        """
        # If it's not an AND node, just return it as a single-element list.
        stack = [cast]
        result = []
        target_code = OP_SYMBOLS["&"] if flat_and else OP_SYMBOLS["|"]
        while stack:
            node = stack.pop()
            # Leaves (RectangularPredicate, ...) are NamedTuples with numpy
            # array fields, so `node[0] == target_code` would broadcast to an
            # array rather than a bool. Guard with _is_leaf before indexing.
            if (not self._is_leaf(node)
                    and isinstance(node, (list, tuple))
                    and node[0] == target_code):
                # node is of the form: ['&', left, right] or ['|', left, right]
                # push its children on the stack
                stack.append(node[2])
                stack.append(node[1])
            else:
                # leaf node (predicate) or non-& \ non-| node
                result.append(node)
        return result

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY_AM)
    def _eval_and(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        def regular_and(_sub_form1, _sub_form2, _path, _start_t, _end_t):
            _train_mode = False
            subforms = self._flatten_and_or([OP_SYMBOLS["&"], _sub_form1, _sub_form2], flat_and=True)

            # 2. Evaluate each subformula
            vals = [
                self._eval(subf, _path, _start_t, _end_t, train_mode=_train_mode,
                           hardness=hardness, approx_method=approx_method)
                for subf in subforms
            ]

            # 3. Stack and do a single min (or your exponential scheme)
            stacked = jnp.stack(vals, axis=-1)
            return self._tensor_min(stacked, axis=-1, hardness=hardness, approx_method=approx_method)

        def pairwise_and(_sub_form1, _sub_form2, _path, _start_t, _end_t):
            """This can cause  brittle or localized gradient."""
            return self._tensor_min(
                jnp.stack(
                    [
                        self._eval(_sub_form1, _path, _start_t, _end_t,
                                   hardness=hardness, approx_method=approx_method),
                        self._eval(_sub_form2, _path, _start_t, _end_t,
                                   hardness=hardness, approx_method=approx_method),
                    ],
                    axis=-1,
                ),
                axis=-1,
                hardness=hardness, approx_method=approx_method,
            )

        return regular_and(sub_form1, sub_form2, path, start_t, end_t)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY_AM)
    def _eval_or(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        def regular_or(_sub_form1, _sub_form2, _path, _start_t, _end_t, _train_mode):
            subforms = self._flatten_and_or([OP_SYMBOLS["|"], sub_form1, sub_form2], flat_and=False)

            # 2. Evaluate each subformula
            vals = [
                self._eval(subf, _path, _start_t, _end_t, train_mode=_train_mode,
                           hardness=hardness, approx_method=approx_method)
                for subf in subforms
            ]

            # 3. Stack and do a single min (or your exponential scheme)
            stacked = jnp.stack(vals, axis=-1)
            return self._tensor_max(stacked, axis=-1, hardness=hardness, approx_method=approx_method)

        def pairwise_or(_sub_form1, _sub_form2, _path, _start_t, _end_t):
            """This can cause  brittle or localized gradient."""
            return self._tensor_max(
                jnp.stack(
                    [
                        self._eval(_sub_form1, _path, _start_t, _end_t,
                                   hardness=hardness, approx_method=approx_method),
                        self._eval(_sub_form2, _path, _start_t, _end_t,
                                   hardness=hardness, approx_method=approx_method),
                    ],
                    axis=-1,
                ),
                axis=-1,
                hardness=hardness, approx_method=approx_method,
            )

        return regular_or(sub_form1, sub_form2, path, start_t, end_t, train_mode)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY_AM)
    def _eval_not(
            self,
            ast: AST,
            path: jnp.array,
            start_t: int,
            end_t: int,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        return -self._eval(ast, path, start_t, end_t, train_mode=train_mode,
                           hardness=hardness, approx_method=approx_method)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY_AM)
    def _eval_implies(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        if IMPLIES_TRICK:
            return (
                    self._eval(sub_form1, path, start_t, end_t, train_mode=train_mode,
                               hardness=hardness, approx_method=approx_method)
                    * self._eval(sub_form2, path, start_t, end_t, train_mode=train_mode,
                                 hardness=hardness, approx_method=approx_method)
            )
        return self._eval_or(
            [OP_SYMBOLS["~"], sub_form1], sub_form2, path, start_t, end_t, train_mode=train_mode,
            hardness=hardness, approx_method=approx_method
        )

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY_AM)
    def _eval_always(
            self,
            sub_form: AST,
            path: jnp.array,
            start_t: int,
            end_t: int,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        if self._is_leaf(sub_form):
            return self._tensor_min(
                sub_form.eval_whole_path(path[:, start_t:end_t], train_mode=train_mode),
                axis=-1, hardness=hardness, approx_method=approx_method
            )

        # unroll always
        val_per_time = jnp.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t, train_mode=train_mode,
                           hardness=hardness, approx_method=approx_method)
                for t in range(end_t - start_t)
            ],
            axis=-1,
        )

        return self._tensor_min(val_per_time, axis=-1, hardness=hardness, approx_method=approx_method)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY_AM)
    def _eval_eventually(
            self,
            sub_form: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        if self._is_leaf(sub_form):
            return self._tensor_max(
                sub_form.eval_whole_path(path[:, start_t:end_t], train_mode=train_mode),
                axis=-1, hardness=hardness, approx_method=approx_method
            )

        # unroll eventually
        val_per_time = jnp.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t, train_mode=train_mode,
                           hardness=hardness, approx_method=approx_method)
                for t in range(end_t - start_t)
            ],
            axis=-1,
        )

        return self._tensor_max(val_per_time, axis=-1, hardness=hardness, approx_method=approx_method)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY_AM)
    def _eval_until(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False,
            hardness=None,
            approx_method: str = None
    ) -> jnp.array:
        # Standard STL until robustness:
        #   ρ(φ₁ U_[start_t, end_t) φ₂)
        #     = max_{t' ∈ [start_t, end_t)} min( min_{t'' ∈ [start_t, t')} ρ(φ₁, t''),
        #                                        ρ(φ₂, t') )
        # The inner min over an empty prefix (at t' = start_t) is vacuously
        # true, encoded by a large sentinel so _tensor_min defers to φ₂.
        def unroll(sub_form):
            if self._is_leaf(sub_form):
                return sub_form.eval_whole_path(path[:, start_t:end_t],
                                                train_mode=train_mode)
            return jnp.stack(
                [
                    self._eval(sub_form, path, start_t=start_t + t, end_t=end_t,
                               train_mode=train_mode, hardness=hardness, approx_method=approx_method)
                    for t in range(end_t - start_t)
                ],
                axis=-1,
            )

        f1 = unroll(sub_form1)  # (..., T)
        f2 = unroll(sub_form2)  # (..., T)

        # Running min of φ₁ over [start_t, t'-1]: shift inclusive cummin right.
        f1_cum = jax.lax.associative_scan(jnp.minimum, f1, axis=-1)
        sentinel = jnp.full(f1_cum.shape[:-1] + (1,), 1e9, dtype=f1_cum.dtype)
        f1_cum_excl = jnp.concatenate([sentinel, f1_cum[..., :-1]], axis=-1)

        per_t = self._tensor_min(jnp.stack([f1_cum_excl, f2], axis=-1), axis=-1,
                                 hardness=hardness, approx_method=approx_method)
        return self._tensor_max(per_t, axis=-1, hardness=hardness, approx_method=approx_method)

    def get_stlpy_form(self):
        # catch already converted form
        if self.stlpy_form is None:
            self.stlpy_form = self._to_stlpy(self.ast)

        return self.stlpy_form

    def _to_stlpy(self, ast) -> STLTree:
        if self._is_leaf(ast):
            ast: AST = ast
            self.stlpy_form = ast.get_stlpy_form()
            return self.stlpy_form

        if ast[0] == OP_SYMBOLS["~"]:
            self.stlpy_form = self._convert_not(ast)
        elif ast[0] == OP_SYMBOLS["G"]:
            self.stlpy_form = self._convert_always(ast)
        elif ast[0] == OP_SYMBOLS["F"]:
            self.stlpy_form = self._convert_eventually(ast)
        elif ast[0] == OP_SYMBOLS["&"]:
            self.stlpy_form = self._convert_and(ast)
        elif ast[0] == OP_SYMBOLS["|"]:
            self.stlpy_form = self._convert_or(ast)
        elif ast[0] == OP_SYMBOLS["->"]:
            self.stlpy_form = self._convert_implies(ast)
        elif ast[0] == OP_SYMBOLS["U"]:
            self.stlpy_form = self._convert_until(ast)
        else:
            raise ValueError(f"Unknown operator {ast[0]}")

        return self.stlpy_form

    def _convert_not(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.negation()

    def _convert_and(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1 & sub_form_2

    def _convert_or(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1 | sub_form_2

    def _convert_implies(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1.negation() | sub_form_2

    def _convert_eventually(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.eventually(ast[2], ast[3] - 1)

    def _convert_always(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.always(ast[2], ast[3] - 1)

    def _convert_until(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1.until(sub_form_2, ast[3], ast[4] - 1)

    @staticmethod
    def _is_leaf(ast: AST):
        # Check is type PREDICATE_FORM
        for pred_type in PREDICATE_TYPES:
            if isinstance(ast, pred_type):
                return True
        return issubclass(type(ast), PredicateBase)

    def _tensor_max(self, tensor: jnp.array, axis=-1, hardness=None,
                    approx_method: str = None) -> jnp.array:
        """Soft max over `axis`. hardness=None -> module HARDNESS; approx_method=None -> APPROX_METHOD."""
        if hardness is None:
            hardness = HARDNESS
        if approx_method is None:
            approx_method = APPROX_METHOD
        if approx_method == "softmax":  # legacy weighted-sum (biased <= true max)
            ratio = softmax(tensor * hardness, axis=axis)
            return jnp.sum(tensor * ratio, axis=axis)
        elif approx_method == "logsumexp":  # mass-conserving smooth max, -> true max as h -> inf
            return logsumexp(hardness * tensor, axis=axis) / hardness
        elif approx_method == "true":  # exact; jax spreads grad across argmax ties
            return jnp.max(tensor, axis=axis)
        raise ValueError(f"Unknown approx_method {approx_method!r}")

    def _tensor_min(self, tensor: jnp.array, axis=-1, hardness=None,
                    approx_method: str = None) -> jnp.array:
        """Soft min via min(x) = -max(-x); identity holds for all three approx methods."""
        return -self._tensor_max(-tensor, axis=axis, hardness=hardness, approx_method=approx_method)

    def simplify(self):
        if self.stlpy_form is None:
            self.get_stlpy_form()
        self.stlpy_form.simplify()

    def __repr__(self):
        if self.expr_repr is not None:
            return self.expr_repr

        expr = self._extract_repr()

        self.expr_repr = expr
        return expr

    def _extract_repr(self, print_rich=False):
        # traverse ast
        operator_stack = [self.ast]
        expr = ""
        cur = self.ast

        def push_stack(ast):
            if isinstance(ast, int) and ast in self.time_bounded_operators:
                time_window = f"[{cur[-2]}, {cur[-1]}]"
                operator_stack.append(time_window)
            operator_stack.append(ast)

        while operator_stack:
            cur = operator_stack.pop()
            if self._is_leaf(cur):
                if print_rich:
                    expr += f"({str(next(cur.__rich_repr__()))})"
                else:
                    expr += cur.__str__()
            elif isinstance(cur, str):
                if cur == "(" or cur == ")":
                    expr += cur
                elif cur.startswith("["):
                    expr += colored(cur, "yellow") + " "
            elif isinstance(cur, int):
                if cur in (OP_SYMBOLS["G"], OP_SYMBOLS["F"], OP_SYMBOLS["~"]):
                    expr += colored(OP_SYMBOLS_INV[cur], "magenta")
                elif cur in (OP_SYMBOLS["&"], OP_SYMBOLS["|"], OP_SYMBOLS["->"], OP_SYMBOLS["U"]):
                    expr += " " + colored(OP_SYMBOLS_INV[cur], "magenta")
                    if cur != OP_SYMBOLS["U"]:
                        expr += " "
            elif cur[0] in self.single_operators:
                # single operator
                if not self._is_leaf(cur[1]):
                    push_stack(")")
                push_stack(cur[1])
                if not self._is_leaf(cur[1]):
                    push_stack("(")
                push_stack(cur[0])
            elif cur[0] in self.binary_operators:
                # binary operator
                if not self._is_leaf(cur[2]) and cur[2][0] in self.binary_operators:
                    push_stack(")")
                    push_stack(cur[2])
                    push_stack("(")
                else:
                    push_stack(cur[2])
                push_stack(cur[0])
                if not self._is_leaf(cur[1]) and cur[1][0] in self.binary_operators:
                    push_stack(")")
                    push_stack(cur[1])
                    push_stack("(")
                else:
                    push_stack(cur[1])
        return expr

    def latex_repr(self):
        repr = self.__repr__()

        def replace_special_chars(match):
            return {
                "~": r"\neg",
                "&": r"\land",
                "|": r"\lor",
                "->": r"\rightarrow",
                "G": r"\Box",
                "F": r"\Diamond",
                "U": r"U",
            }[match.group(0)]

        replaced_symb = re.sub(r"~|&|\||->|G|F|U", replace_special_chars, repr)
        # replace any [a, b] with _{[a,b]}
        replaced_symb = re.sub(r"\[(\d+), (\d+)\]", r"_{[\1,\2]}", replaced_symb)
        # replace goal_0, goal_1, ... with A, B, C, ...
        replaced_symb = re.sub(r"goal_(\d+)", lambda x: chr(ord('A') + int(x.group(1))), replaced_symb)
        return replaced_symb

    def get_all_predicates(self):
        all_preds = []
        queue = deque([self.ast])

        while queue:
            cur = queue.popleft()

            if self._is_leaf(cur):
                all_preds.append(cur)
            elif cur[0] in self.single_operators:
                queue.append(cur[1])
            elif cur[0] in self.binary_operators:
                queue.append(cur[1])
                queue.append(cur[2])
            else:
                raise RuntimeError("Should never visit here")

        return all_preds

# TODO: Properly register if needed for use with JAX

# # Register PredicateBase as a PyTree
# jtu.register_pytree_node(
#     PredicateBase,
#     lambda pred: ((), (pred.name,)),  # Flatten: no JAX-tracked fields, only auxiliary data
#     lambda aux, _: PredicateBase(aux[0])  # Unflatten
# )
#
# # Register RectangularPredicate as a PyTree
# jtu.register_pytree_node(
#     RectangularPredicate,
#     lambda pred: ((pred.cent, pred.size), (pred.name, pred.shrink_factor)),  # Flatten
#     lambda aux, children: RectangularPredicate(children[0], children[1], aux[0], aux[1])  # Unflatten
# )
#
# # Register STL as a PyTree
# jtu.register_pytree_node(
#     STL,
#     lambda stl: ((stl.ast,), ()),  # Flatten: AST (JAX-tracked), no auxiliary data
#     lambda aux, children: STL(children[0])  # Unflatten
# )
