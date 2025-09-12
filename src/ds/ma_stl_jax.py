from typing import Optional

import jax.lax
from jax import Array

from .stl_jax import *

"""Multi agent STL specifications"""

EXP_ROBUSTNESS_ALPHA = 1.0  # For task exponential robustness
EXP_ROBUSTNESS_BETA = .5  # For AND exponential robustness


class TaskBase(NamedTuple):
    """Base class for tasks in CaTL+ <https://ieeexplore.ieee.org/document/10156237>."""
    name: int
    spec: STL
    num_satisfied_agents: int
    capability: Optional[list[int]] = None  # Capability set for the task # TODO: Implement this

    def eval_at_t(self, path: jnp.ndarray, t: int = 0, train_mode: bool = False) -> jnp.ndarray:
        return self.spec.eval(path, t, train_mode=train_mode)  # [:, 0]

    @abstractmethod
    def eval_whole_path(
            self, path: jnp.ndarray, start_t: int = 0, end_t: int = None, train_mode: bool = False
    ) -> jnp.ndarray:
        """Stick to JAX when possible."""
        raise NotImplementedError

    # @abstractmethod
    # def get_stlpy_form(self) -> STLTree:
    #     """Use Numpy to ensure compatibility with STLpy."""
    #     raise NotImplementedError

    def __str__(self) -> str:
        # TODO: Implement better string representation
        return f"<Task {self.name} with spec \'{self.spec}\' , m={self.num_satisfied_agents}, capability={self.capability}>"

    def __lt__(self, other: "TaskBase") -> bool:
        """Sort predicates by name."""
        return self.name < other.name


class Task(TaskBase):
    """Task for CaTL+."""

    def eval_whole_path(
            self, path: jnp.ndarray, start_t: int = 0, end_t: int = None, train_mode: bool = False
    ) -> jnp.ndarray:
        # NOTE: Does not support end_t
        # TODO: If train_mode, return the exponential robustness
        # return self.spec.eval(path, start_t, train_mode) > 0
        if train_mode:
            return self.exponential_val(path, start_t, train_mode)
        else:
            return self.regular_val(path, start_t, train_mode)

    def exponential_val(self, path, start_t, train_mode):
        topk_vals, topk_ind = jax.lax.top_k(self.spec.eval(path, start_t, train_mode=train_mode),
                                            self.num_satisfied_agents)
        vals = self.spec.eval(path, start_t, train_mode=train_mode)
        topk_val = topk_vals[-1]
        # Compute an index based on the sign of min_val:
        #   0 -> negative, 1 -> positive, 2 -> zero.

        branch_index = jnp.where(topk_val < 0,
                                 0,
                                 jnp.where(topk_val > 0, 1, 2))

        def negative(_stacked_vals, _topk_val):
            # When min is negative
            return -2 * EXP_ROBUSTNESS_ALPHA * (jnp.exp(-_stacked_vals) - 1) / (
                    1 + jnp.exp(EXP_ROBUSTNESS_ALPHA * (_stacked_vals - _topk_val)))

        def positive(_stacked_vals, _topk_val):
            # When min is positive; note the rearrangement for the given formula.
            return 2 * EXP_ROBUSTNESS_ALPHA * (jnp.exp(_stacked_vals) - 1) / (
                    1 + jnp.exp(-EXP_ROBUSTNESS_ALPHA * (_stacked_vals - _topk_val)))

        # List of branch functions. Each branch must have the same output type.
        branches = [negative, positive, negative]

        # Use jax.lax.switch to select and run the correct branch.
        res = jax.lax.switch(branch_index, branches, vals, topk_val)

        return res.mean()

    def regular_val(self, path, start_t, train_mode):
        topk_vals, topk_ind = jax.lax.top_k(self.spec.eval(path, start_t, train_mode=train_mode),
                                            self.num_satisfied_agents)
        # Only evaluate satisfaction (not best for gradient updates since only m-th best is considered)
        return topk_vals[-1]

    def eval_at_t(self, path: jnp.ndarray, t: int = 0, train_mode: bool = False) -> jnp.ndarray:
        return self.eval_whole_path(path, t, t + 1, train_mode)

    def count_satisfied_agents(self, path: jnp.ndarray, start_t: int = 0, end_t: int = None,
                               train_mode: bool = False) -> Array:
        return jnp.sum(self.eval_whole_path(path, start_t, end_t, train_mode) > 0)

    def __new__(cls, name, spec, num_satisfied_agents, capability=None):
        return super(Task, cls).__new__(cls, name, spec, num_satisfied_agents, capability)


TASK_TYPES = (Task, TaskBase)

cAST = TypeVar("AST", list, TaskBase)


class CaTLPlus:
    """Outerlogic for CaTL+. Takes the whole system of agents NxMxT and returns the satisfaction of the formula.
    Allows exponential robustness evaluation as in CaTL+ using the train_mode flag."""

    def __init__(self, cast: cAST):
        # self.cast = cast  # Original AST

        single_operators = ("~",)
        binary_operators = ("&", "|", "->", "U")
        sequence_operators = ("G", "F", "U")
        not_implemented = ("G", "F", "U", "->")
        self.single_operators = tuple(OP_SYMBOLS[op] for op in single_operators)
        self.binary_operators = tuple(OP_SYMBOLS[op] for op in binary_operators)
        self.sequence_operators = tuple(OP_SYMBOLS[op] for op in sequence_operators)
        self.time_bounded_operators = tuple(OP_SYMBOLS[op] for op in sequence_operators)
        self.not_implemented = tuple(OP_SYMBOLS[op] for op in not_implemented)
        self.expr_repr = None
        self.end_t = None  # Populated when evaluating
        self.logger = logging.getLogger(__name__)

        # Recursively transform the user-provided AST so that any string operator
        # is replaced by its numeric code.
        self.cast = cast
        self.tuple_cast = list_to_tuple(cast)

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
    (They produce integer-coded cASTs rather than string-coded.)
    """

    def __and__(self, other: "CaTLPlus") -> "CaTLPlus":
        cast = [OP_SYMBOLS["&"], self.cast, other.cast]
        return CaTLPlus(cast)

    def __or__(self, other: "CaTLPlus") -> "CaTLPlus":
        cast = [OP_SYMBOLS["|"], self.cast, other.cast]
        return CaTLPlus(cast)

    def __invert__(self) -> "CaTLPlus":
        cast = [OP_SYMBOLS["~"], self.cast]
        return CaTLPlus(cast)

    def implies(self, other: "CaTLPlus") -> "CaTLPlus":
        cast = [OP_SYMBOLS["->"], self.cast, other.cast]
        return CaTLPlus(cast)

    def eventually(self, start: int, end: int):
        cast = [OP_SYMBOLS["F"], self.cast, start, end]
        return CaTLPlus(cast)

    def always(self, start: int, end: int) -> "CaTLPlus":
        cast = [OP_SYMBOLS["G"], self.cast, start, end]
        return CaTLPlus(cast)

    def until(self, other: "CaTLPlus", start: int, end: int) -> "CaTLPlus":
        cast = [OP_SYMBOLS["U"], self.cast, other.cast, start, end]
        return CaTLPlus(cast)

    def eval_on_batch(self, path: jnp.array, t: int = 0, train_mode: bool = False) -> jnp.array:
        """Evaluate the formula at time t.

        :param path:            The motion path to evaluate the formula on (Batch x Num_agents x Traj_len x Dim)
        :param train_mode:   Whether to evaluate in training mode (conservative with shrink factor).
        :param t:               The time step to evaluate the formula at.
        """
        eval_fn = ft.partial(self.eval, train_mode=train_mode)
        return jax.vmap(eval_fn, in_axes=0)(path, t)

    def eval(self, path: jnp.array, t: int = 0, train_mode: bool = False) -> jnp.array:
        """Evaluate the formula at time t. Training mode is for exponential robustness as in CaTL+.

        :param path:            The motion path to evaluate the formula on (Num_agents x Traj_len x Dim)
        :param train_mode:      Whether to evaluate in training mode (exponential robustness).
        :param t:               The time step to evaluate the formula at.
        """
        return self._eval(self.tuple_cast, path, t, train_mode=train_mode)

    def eval_train(self, path: jnp.array, t: int = 0) -> jnp.array:
        """To help prevent recompilation in jax.jit, we separate the training mode evaluation."""
        return self.eval(path, t, train_mode=True)

    def end_time(self) -> int:
        """Get the end time of the formula efficiently."""
        if self.end_t is None:
            # Evaluate the formula to get the end time
            # Get max of binary tree at self.cast
            self.end_t = self._get_end_time(self.cast)
        return self.end_t

    def _get_end_time(self, cast: cAST) -> int:
        """Get max time of the formula. Runs in O(n) time where n is the number of nodes. Runs once then memoizes."""
        if self._is_leaf(cast):
            return 1
        op = cast[0]
        if op == OP_SYMBOLS["G"]:
            # add end time from inner formula
            return cast[-1] + self._get_end_time(cast[1])
        elif op in self.sequence_operators:
            # The last two elements are the start and end times
            return cast[-1]
        elif op == OP_SYMBOLS["~"]:
            return self._get_end_time(cast[1])
        elif op in self.binary_operators:
            return max(self._get_end_time(cast[1]), self._get_end_time(cast[2]))
        return 1

    @staticmethod
    def _is_leaf(cast: cAST):
        # Check is type PREDICATE_FORM
        for pred_type in TASK_TYPES:
            if isinstance(cast, pred_type):
                return True
        return issubclass(type(cast), PredicateBase)

    def _tensor_min(self, tensor: jnp.array, axis=-1) -> jnp.array:
        ratio = softmax(tensor * -HARDNESS, axis=axis)
        return jnp.sum(tensor * ratio, axis=axis)

    def _tensor_max(self, tensor: jnp.array, axis=-1) -> jnp.array:
        ratio = softmax(tensor * HARDNESS, axis=axis)
        return jnp.sum(tensor * ratio, axis=axis)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY)
    def _eval(
            self,
            cast: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        if self._is_leaf(cast):
            return cast.eval_at_t(path, start_t, train_mode=train_mode)

        op_code = cast[0]
        # If it's one of the time-bounded operators (U, G, F), update start/end
        if op_code in self.sequence_operators:  # "U" => 3
            start_t, end_t = start_t + cast[-2], start_t + cast[-1]
            if end_t > path.shape[1]:
                self.logger.warning("end_t is larger than motion length")

        if op_code in self.not_implemented:
            raise NotImplementedError(f"Operator {OP_SYMBOLS_INV[op_code]} is not implemented")

        if op_code == OP_SYMBOLS["&"]:
            return self._eval_and(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        elif op_code == OP_SYMBOLS["|"]:
            return self._eval_or(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        elif op_code == OP_SYMBOLS["~"]:
            return self._eval_not(cast[1], path, start_t, end_t, train_mode=train_mode)
        elif op_code == OP_SYMBOLS["->"]:
            return self._eval_implies(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        elif op_code == OP_SYMBOLS["G"]:
            return self._eval_always(cast[1], path, start_t, end_t, train_mode=train_mode)
        elif op_code == OP_SYMBOLS["F"]:
            return self._eval_eventually(cast[1], path, start_t, end_t, train_mode=train_mode)
        elif op_code == OP_SYMBOLS["U"]:
            return self._eval_until(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)

        raise ValueError(f"Unknown operator code {op_code}")

    def _flatten_and_or(self, cast: cAST, flat_and=True) -> list[cAST]:
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
            if (isinstance(node, list) or isinstance(node, tuple)) and node[0] == target_code:
                # node is of the form: ['&', left, right]
                # push its children on the stack
                stack.append(node[2])
                stack.append(node[1])
            else:
                # leaf node (predicate) or non-& operator
                result.append(node)
        return result

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY)
    def _eval_and(
            self,
            sub_form1: cAST,
            sub_form2: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        def regular_and(_sub_form1, _sub_form2, _path, _start_t, _end_t):
            _train_mode = False
            subforms = self._flatten_and_or([OP_SYMBOLS["&"], _sub_form1, _sub_form2], flat_and=True)

            # 2. Evaluate each subformula
            vals = [
                self._eval(subf, _path, _start_t, _end_t, train_mode=_train_mode)
                for subf in subforms
            ]

            # 3. Stack and do a single min (or your exponential scheme)
            stacked = jnp.stack(vals, axis=-1)
            return self._tensor_min(stacked, axis=-1)

        def exponential_and(_sub_form1, _sub_form2, _path, _start_t, _end_t):
            _train_mode = True
            subforms = self._flatten_and_or([OP_SYMBOLS["&"], _sub_form1, _sub_form2], flat_and=True)

            # 2. Evaluate each subformula
            vals = [
                self._eval(subf, _path, _start_t, _end_t, train_mode=_train_mode)
                for subf in subforms
            ]
            stacked = jnp.stack(vals, axis=-1)
            min_val = jnp.min(stacked, axis=-1)

            # If min is negative, return min * exp(val-min / min)
            # If min is positive, return min * (2 - exp (min-val / min))
            # If min is zero, return min
            # Compute an index based on the sign of min_val:
            #   0 -> negative, 1 -> positive, 2 -> zero.

            branch_index = jnp.where(min_val < 0,
                                     0,
                                     jnp.where(min_val > 0, 1, 2))

            def negative(_stacked_vals, _min_val):
                # When min is negative
                return _min_val * jnp.exp((_stacked_vals - _min_val) / _min_val)

            def positive(_stacked_vals, _min_val):
                # When min is positive; note the rearrangement for the given formula.
                return _min_val * (2 - jnp.exp((_min_val - _stacked_vals) / _min_val))

            def zero(_stacked_vals, _min_val):
                # When min is zero, simply return _min_val (which is zero)
                return jnp.ones_like(_stacked_vals) * _min_val

            # List of branch functions. Each branch must have the same output type.
            branches = [negative, positive, zero]

            # Use jax.lax.switch to select and run the correct branch.
            res = jax.lax.switch(branch_index, branches, stacked, min_val)

            return res.mean() * (1 - EXP_ROBUSTNESS_BETA) + min_val * EXP_ROBUSTNESS_BETA

        if train_mode:
            return exponential_and(sub_form1, sub_form2, path, start_t, end_t)
        else:
            return regular_and(sub_form1, sub_form2, path, start_t, end_t)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY)
    def _eval_or(
            self,
            sub_form1: cAST,
            sub_form2: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        # TODO: Implement exponential robustness for OR as inverse of AND
        def regular_or(_sub_form1, _sub_form2, _path, _start_t, _end_t, _train_mode):
            subforms = self._flatten_and_or([OP_SYMBOLS["|"], sub_form1, sub_form2], flat_and=False)

            # 2. Evaluate each subformula
            vals = [
                self._eval(subf, _path, _start_t, _end_t, train_mode=_train_mode)
                for subf in subforms
            ]

            # 3. Stack and do a single min (or your exponential scheme)
            stacked = jnp.stack(vals, axis=-1)
            return self._tensor_max(stacked, axis=-1)

        return regular_or(sub_form1, sub_form2, path, start_t, end_t, train_mode)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY)
    def _eval_not(
            self,
            cast: cAST,
            path: jnp.array,
            start_t: int,
            end_t: int,
            train_mode: bool = False
    ) -> jnp.array:
        return -self._eval(cast, path, start_t, end_t, train_mode=train_mode)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY)
    def _eval_implies(
            self,
            sub_form1: cAST,
            sub_form2: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        if IMPLIES_TRICK:
            return (
                    self._eval(sub_form1, path, start_t, end_t, train_mode=train_mode)
                    * self._eval(sub_form2, path, start_t, end_t, train_mode=train_mode)
            )
        return self._eval_or(
            [OP_SYMBOLS["~"], sub_form1], sub_form2, path, start_t, end_t, train_mode=train_mode
        )

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY)
    def _eval_always(
            self,
            sub_form: cAST,
            path: jnp.array,
            start_t: int,
            end_t: int,
            train_mode: bool = False
    ) -> jnp.array:
        if self._is_leaf(sub_form):
            return self._tensor_min(
                sub_form.eval_whole_path(path[:, start_t:end_t], train_mode=train_mode),
                axis=-1
            )

        # unroll always
        val_per_time = jnp.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t, train_mode=train_mode)
                for t in range(end_t - start_t)
            ],
            axis=-1,
        )

        return self._tensor_min(val_per_time, axis=-1)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_UNARY)
    def _eval_eventually(
            self,
            sub_form: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        if self._is_leaf(sub_form):
            return self._tensor_max(
                sub_form.eval_whole_path(path[:, start_t:end_t], train_mode=train_mode),
                axis=-1
            )

        # unroll eventually
        val_per_time = jnp.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t, train_mode=train_mode)
                for t in range(end_t - start_t)
            ],
            axis=-1,
        )

        return self._tensor_max(val_per_time, axis=-1)

    @ft.partial(jax.jit, static_argnums=STATIC_ARGNUMS_BINARY)
    def _eval_until(
            self,
            sub_form1: cAST,
            sub_form2: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        if self._is_leaf(sub_form2):
            till_pred = sub_form2.eval_whole_path(path[:, start_t:end_t], train_mode=train_mode)
        else:
            till_pred = jnp.stack(
                [
                    self._eval(sub_form2, path, start_t=t, end_t=end_t, train_mode=train_mode)
                    for t in range(end_t - start_t)
                ],
                axis=-1,
            )

        # mask condition...
        cond = (till_pred > 0).castype(int)
        index = jnp.argmax(cond, axis=-1)
        batch_size, seq_len = cond.shape
        row_indices = jnp.arange(batch_size)[:, None]
        col_indices = jnp.arange(seq_len)
        mask = col_indices >= index[:, None]
        cond = ~mask.castype(bool)

        # Set true values after 'till' is satisfied
        till_pred = jnp.where(cond, till_pred, ds_utils.default_tensor(1))

        if self._is_leaf(sub_form1):
            res = sub_form1.eval_whole_path(path[:, start_t:end_t], train_mode=train_mode)
        else:
            res = jnp.stack(
                [
                    self._eval(sub_form1, path, start_t=t, end_t=end_t, train_mode=train_mode)
                    for t in range(end_t - start_t)
                ],
                axis=-1,
            )

        res = jnp.where(cond, res, ds_utils.default_tensor(-1))
        # when cond < 0, res should always > 0 to be hold
        return self._tensor_min(-res * till_pred, axis=-1)

    def __repr__(self):
        if self.expr_repr is not None:
            return self.expr_repr

        self.expr_repr = self._extract_repr()
        return self.expr_repr

    def _extract_repr(self, print_rich=False):
        # traverse cast
        operator_stack = [self.cast]
        expr = ""
        cur = self.cast

        def push_stack(cast):
            if isinstance(cast, int) and cast in self.time_bounded_operators:
                time_window = f"[{cur[-2]}, {cur[-1]}]"
                operator_stack.append(time_window)
            operator_stack.append(cast)

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
        queue = deque([self.cast])

        while queue:
            cur = queue.popleft()
            op_code = cur[0]

            if self._is_leaf(cur):
                all_preds.append(cur)
            elif op_code in self.single_operators:
                queue.append(cur[1])
            elif op_code in self.binary_operators:
                queue.append(cur[1])
                queue.append(cur[2])
            else:
                raise RuntimeError("Should never visit here")

        return all_preds

# TODO: Properly register if needed for use with JAX

# jtu.register_pytree_node(
#     Task,
#     lambda task: ((task.spec,), (task.name, task.num_satisfied_agents, task.capability)),  # Flatten
#     lambda aux, children: Task(aux[0], children[0], aux[1], aux[2])  # Unflatten
# )
