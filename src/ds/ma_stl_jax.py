from typing import Optional

import jax.lax

from .stl_jax import *

"""Multi agent STL specifications"""


class TaskBase(NamedTuple):
    """Base class for tasks in CaTL+ <https://ieeexplore.ieee.org/document/10156237>."""
    name: str
    spec: STL
    num_satisfied_agents: int
    capability: Optional[list[str]] = None  # Capability set for the task # TODO: Implement this

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
        return self.name

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
        topk_val, topk_ind = jax.lax.top_k(self.spec.eval(path, start_t, train_mode=train_mode),
                                           self.num_satisfied_agents)

        # Only evaluate satisfaction (not best for gradient updates since only m-th best is considered)
        return topk_val[-1]

    def eval_at_t(self, path: jnp.ndarray, t: int = 0, train_mode: bool = False) -> jnp.ndarray:
        return self.eval_whole_path(path, t, t + 1, train_mode)

    def count_satisfied_agents(self, path: jnp.ndarray, start_t: int = 0, end_t: int = None,
                               train_mode: bool = False) -> int:
        return jnp.sum(self.eval_whole_path(path, start_t, end_t, train_mode) > 0)

    def __new__(cls, name, spec, num_satisfied_agents, capability=None):
        return super(Task, cls).__new__(cls, name, spec, num_satisfied_agents, capability)


jtu.register_pytree_node(
    Task,
    lambda task: ((task.spec,), (task.name, task.num_satisfied_agents, task.capability)),  # Flatten
    lambda aux, children: Task(aux[0], children[0], aux[1], aux[2])  # Unflatten
)

TASK_TYPES = (Task, TaskBase)

cAST = TypeVar("AST", list, TaskBase)

EXP_ROBUSTNESS_BETA = .5


class CaTLPlus:
    """Outerlogic for CaTL+. Takes the whole system of agents NxMxT and returns the satisfaction of the formula.
    Allows exponential robustness evaluation as in CaTL+ using the train_mode flag."""

    def __init__(self, cast: cAST):
        self.cast = cast
        self.single_operators = ("~")
        self.binary_operators = ("&", "|", "U")
        self.sequence_operators = ("U")  # ("G", "F", "U") ?
        self.not_implemented = ("G", "F", "U", "->")
        self.expr_repr = None
        self.end_t = None  # Populated when evaluating
        self.logger = logging.getLogger(__name__)

    """
        Syntax Functions
        """

    def __and__(self, other: "CaTLPlus") -> "CaTLPlus":
        cast = ["&", self.cast, other.cast]
        return CaTLPlus(cast)

    def __or__(self, other: "CaTLPlus") -> "CaTLPlus":
        cast = ["|", self.cast, other.cast]
        return CaTLPlus(cast)

    def __invert__(self) -> "CaTLPlus":
        cast = ["~", self.cast]
        return CaTLPlus(cast)

    def implies(self, other: "CaTLPlus") -> "CaTLPlus":
        cast = ["->", self.cast, other.cast]
        return CaTLPlus(cast)

    def eventually(self, start: int, end: int):
        cast = ["F", self.cast, start, end]
        return CaTLPlus(cast)

    def always(self, start: int, end: int) -> "CaTLPlus":
        cast = ["G", self.cast, start, end]
        return CaTLPlus(cast)

    def until(self, other: "CaTLPlus", start: int, end: int) -> "CaTLPlus":
        cast = ["U", self.cast, other.cast, start, end]
        return CaTLPlus(cast)

    def eval_on_batch(self, path: jnp.array, t: int = 0, train_mode: bool = False) -> jnp.array:
        """Evaluate the formula at time t.

        :param path:            The motion path to evaluate the formula on (Batch x Num_agents x Traj_len x Dim)
        :param train_mode:   Whether to evaluate in training mode (conservative with shrink factor).
        :param t:               The time step to evaluate the formula at.
        """
        return jax.vmap(self.eval, in_axes=0)(path, t, train_mode)

    def eval(self, path: jnp.array, t: int = 0, train_mode: bool = False) -> jnp.array:
        """Evaluate the formula at time t. Training mode is for exponential robustness as in CaTL+.

        :param path:            The motion path to evaluate the formula on (Num_agents x Traj_len x Dim)
        :param train_mode:      Whether to evaluate in training mode (exponential robustness).
        :param t:               The time step to evaluate the formula at.
        """
        return self._eval(self.cast, path, t, train_mode=train_mode)

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
        if cast[0] == "G":
            # Add end time from inner formula since always is unrolled
            return cast[-1] + self._get_end_time(cast[1])
        if cast[0] in self.sequence_operators:
            # The lcast two elements are the start and end times
            return cast[-1]
        if cast[0] == "~":
            return self._get_end_time(cast[1])
        # Is binary operator
        return max(self._get_end_time(cast[1]), self._get_end_time(cast[2]))

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

        if cast[0] in self.sequence_operators:
            # NOTE: Overwrite start_t and end_t
            start_t, end_t = start_t + cast[-2], start_t + cast[-1]
            if end_t > path.shape[1]:
                self.logger.warning("end_t is larger than motion length")

        if cast[0] in self.not_implemented:
            raise NotImplementedError(f"Operator {cast[0]} is not implemented")

        if cast[0] == "&":
            res = self._eval_and(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        elif cast[0] == "|":
            res = self._eval_or(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        elif cast[0] == "~":
            res = self._eval_not(cast[1], path, start_t, end_t, train_mode=train_mode)
        elif cast[0] == "->":
            res = self._eval_implies(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        elif cast[0] == "G":
            res = self._eval_always(cast[1], path, start_t, end_t, train_mode=train_mode)
        elif cast[0] == "F":
            res = self._eval_eventually(cast[1], path, start_t, end_t, train_mode=train_mode)
        elif cast[0] == "U":
            res = self._eval_until(cast[1], cast[2], path, start_t, end_t, train_mode=train_mode)
        else:
            raise ValueError(f"Unknown operator {cast[0]}")

        return res

    def _flatten_and_or(self, cast: cAST, flat_and=True) -> list[cAST]:
        """
        Recursively collect every subformula under an & chain into a single list.
        For example, if cast = ['&', A, ['&', B, C]], then
        _flatten_and(cast) = [A, B, C].
        """
        # If it's not an AND node, just return it as a single-element list.
        stack = [cast]
        result = []
        flat_char = "&" if flat_and else "|"
        while stack:
            node = stack.pop()
            if isinstance(node, list) and node[0] == flat_char:
                # node is of the form: ['&', left, right]
                # push its children on the stack
                stack.append(node[2])
                stack.append(node[1])
            else:
                # leaf node (predicate) or non-& operator
                result.append(node)
        return result

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
            subforms = self._flatten_and_or(["&", _sub_form1, _sub_form2], flat_and=True)

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
            subforms = self._flatten_and_or(["&", _sub_form1, _sub_form2], flat_and=True)

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
                return jnp.ones_like(stacked) * _min_val

            # List of branch functions. Each branch must have the same output type.
            branches = [negative, positive, zero]

            # Use jax.lax.switch to select and run the correct branch.
            res = jax.lax.switch(branch_index, branches, stacked, min_val)

            return res.mean() * (1 - EXP_ROBUSTNESS_BETA) + min_val * EXP_ROBUSTNESS_BETA

        if train_mode:
            return exponential_and(sub_form1, sub_form2, path, start_t, end_t)
        else:
            return regular_and(sub_form1, sub_form2, path, start_t, end_t)

    def _eval_or(
            self,
            sub_form1: cAST,
            sub_form2: cAST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
            train_mode: bool = False
    ) -> jnp.array:
        def regular_or(_sub_form1, _sub_form2, _path, _start_t, _end_t, _train_mode):
            subforms = self._flatten_and_or(["|", _sub_form1, _sub_form2], flat_and=False)

            # 2. Evaluate each subformula
            vals = [
                self._eval(subf, _path, _start_t, _end_t, train_mode=_train_mode)
                for subf in subforms
            ]

            # 3. Stack and do a single min (or your exponential scheme)
            stacked = jnp.stack(vals, axis=-1)
            return self._tensor_max(stacked, axis=-1)

        return regular_or(sub_form1, sub_form2, path, start_t, end_t, train_mode)

    def _eval_not(
            self,
            cast: cAST,
            path: jnp.array,
            start_t: int,
            end_t: int,
            train_mode: bool = False
    ) -> jnp.array:
        return -self._eval(cast, path, start_t, end_t, train_mode=train_mode)

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
            ["~", sub_form1], sub_form2, path, start_t, end_t, train_mode=train_mode
        )

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

        expr = self._extract_repr()

        self.expr_repr = expr
        return expr

    def _extract_repr(self, print_rich=False):
        single_operators = ("~", "G", "F")
        binary_operators = ("&", "|", "->", "U")
        time_bounded_operators = ("G", "F", "U")
        # traverse cast
        operator_stack = [self.cast]
        expr = ""
        cur = self.cast

        def push_stack(cast):
            if isinstance(cast, str) and cast in time_bounded_operators:
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
                else:
                    if cur in ("G", "F"):
                        if cur == "F":
                            expr += colored("F", "magenta")
                        else:
                            expr += colored(cur, "magenta")
                    elif cur in ("&", "|", "->", "U"):
                        expr += " " + colored(cur, "magenta")
                        if cur != "U":
                            expr += " "
                    elif cur in ("~",):
                        expr += colored(cur, "magenta")
            elif cur[0] in single_operators:
                # single operator
                if not self._is_leaf(cur[1]):
                    push_stack(")")
                push_stack(cur[1])
                if not self._is_leaf(cur[1]):
                    push_stack("(")
                push_stack(cur[0])
            elif cur[0] in binary_operators:
                # binary operator
                if not self._is_leaf(cur[2]) and cur[2][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[2])
                    push_stack("(")
                else:
                    push_stack(cur[2])
                push_stack(cur[0])
                if not self._is_leaf(cur[1]) and cur[1][0] in binary_operators:
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
