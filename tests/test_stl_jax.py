import importlib
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

os.environ["DIFF_STL_BACKEND"] = "jax"  # So ds_utils does not require torch

import ds.utils as ds_utils
import examples.stl.differentiability as stl_diff_examples
from ds.stl_jax import STL, RectReachPredicate

from ds.stl import StlpySolver

# LARGE_TEST_TOLERANCE shouldn't be too small since avoid_backward sometimes fails
LARGE_TEST_TOLERANCE = 2e-2  # Smallish number close to 0
TEST_TOLERANCE = 1e-3  # Small number close to 0


class TestJAXExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
        importlib.reload(stl_diff_examples)  # Reload the module to reset the backend
        importlib.reload(ds_utils)  # Reload the module to reset the backend

        self.key = jax.random.PRNGKey(0)

        self.goal_1 = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), 1))
        # goal_2 is a rectangle area centered in [2, 2] with width and height 1
        self.goal_2 = STL(RectReachPredicate(np.array([2, 2]), np.array([1, 1]), 2))

        # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
        # and that holds always in 0 to 8
        # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
        self.form = (self.goal_1.eventually(0, 5) & self.goal_2.eventually(0, 5)).always(0, 8)
        self.loop_form = (self.goal_1.eventually(0, 4) & self.goal_2.eventually(0, 4)).always(0, 8)
        self.cover_form = self.goal_1.eventually(0, 12) & self.goal_2.eventually(0, 12)
        self.seq_form = self.goal_1.eventually(0, 6) & self.goal_2.eventually(6, 12)

        self.all_forms = [self.form, self.loop_form, self.cover_form, self.seq_form]

    def test_repr(self):
        print(self.form)
        for form in self.all_forms:
            print(form)

    def test_run(self):
        # TODO: Study jit decorator and see optimizations
        # jit(eval_reach_avoid)()

        final_result = []
        for _ in range(1000):
            # Magic of jax
            res = jit(stl_diff_examples.eval_reach_avoid)()
            final_result.append(res)
            # Match expected output
            assert jnp.all((res[0] > 0) == jnp.array([True, False, False]))
            assert jnp.all((res[1] > 0) == jnp.array([True, False, True]))

        print(final_result)

        # Test differentiability
        path, loss = stl_diff_examples.backward(self.key)
        print('Path', path)
        assert loss < TEST_TOLERANCE  # Loss should be less than 0 to satisfy the formula
        # (jax.lax.fori_loop(0, 1000, lambda i, _: jit(eval_reach_avoid)(), None)).block_until_ready()
        # for _ in range(1000):
        #     eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here

    def test_avoid_backward(self):

        def avoid_test(_key):
            path, loss = stl_diff_examples.backward(_key, avoid_spec=True)
            return path, loss

        num_tests = 50
        keys = jax.random.split(self.key, num_tests)
        paths, losses = jax.vmap(avoid_test)(keys)

        print('AvoidPath', losses[0], paths[0])
        print(f"Unsatisfied losses {sum(losses > TEST_TOLERANCE)} out of {num_tests}")
        print(f"Max loss {jnp.max(losses)}")
        assert (losses < LARGE_TEST_TOLERANCE).all()  # Loss should be less than 0 to satisfy the formula

    def test_evaluations(self, num_tiles=3):
        """Run simple evaluations to test shapes and types"""
        path = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0, 0],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [1, 0],
                        [1, 0],
                    ],
                ]
            )
        )

        loss = self.form.eval(jax.numpy.tile(path, (num_tiles, 1, 1)))  # Make a batch of size num_tiles
        self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertEqual(loss.shape[0], num_tiles, f"Not returning {num_tiles} values")

    def test_loop(self):
        # Test loop spec
        num_tiles = 4
        path = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [0, 0],
                        [0, 2],
                        [2, 0],
                        [2, 2],
                    ] * 3
                ]))
        loss = self.loop_form.eval(jax.numpy.tile(path, (num_tiles, 1, 1)))  # Make a batch of size num_tiles
        self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertEqual(loss.shape[0], num_tiles, f"Not returning {num_tiles} values")
        self.assertGreater(loss[0], 0, f"Loss is not greater than 0")

        unsat_path = path.at[0, -4].set([0, 2])  # Make the last point unsatisfiable
        loss = self.loop_form.eval(jax.numpy.tile(unsat_path, (num_tiles, 1, 1)))
        self.assertLess(loss[0], 0, f"Loss is not less than 0 for unsat path")

    def test_stlpy_solver(self):
        """Test the stlpy solver with different forms of STL formulas"""
        x_0 = np.array([0, 0])
        solver = StlpySolver(space_dim=2)
        total_time = 12  # Common total time for all formulas

        for form in [self.loop_form, self.cover_form, self.seq_form]:
            stlpy_form = form.get_stlpy_form()
            path, info = solver.solve_stlpy_formula(stlpy_form, x0=x_0, total_time=total_time)

            num_tiles = 4
            loss = form.eval(jax.numpy.tile(path, (num_tiles, 1, 1)))  # Make a batch of size num_tiles
            self.assertGreater(loss[0], 0, f"STLPY solved path loss is not greater than 0 for {form}")


class TestUntilSemantics(unittest.TestCase):
    """Targeted tests for the `until` operator semantics.

    Ground truth for sign/ordering comes from stlpy's analytic robustness.
    Soft-min/soft-max approximations used by the JAX backend can shift the
    magnitude, so we assert on *sign* and on leaf-vs-nonleaf consistency
    rather than exact numeric equality.
    """

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"
        importlib.reload(ds_utils)

        # Two disjoint reach regions
        self.phi1 = STL(RectReachPredicate(np.array([0, 0]), np.array([2, 2]), "phi1"))
        self.phi2 = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "phi2"))

        # phi1 U_[0, 10] phi2, evaluated on paths of length 11
        self.T = 11
        self.interval = (0, self.T - 1)
        self.until_form = self.phi1.until(self.phi2, *self.interval)

    @staticmethod
    def _stlpy_robustness(stl_form, np_path):
        """Ground-truth robustness at t=0. np_path has shape (T, d); stlpy wants (d, T)."""
        return stl_form.get_stlpy_form().robustness(np_path.T, 0)

    def _jax_eval(self, form, np_path):
        """Evaluate on a single-batch path."""
        return float(form.eval(ds_utils.default_tensor(np_path[None])).squeeze())

    # --- paths ---

    @property
    def _path_sat(self):
        """Stay in phi1 for the first half, then move to phi2 and stay."""
        return np.array([[0, 0]] * 5 + [[4, 4]] * 6, dtype=np.float32)

    @property
    def _path_never_phi2(self):
        """Stay in phi1 the whole time; phi2 is never reached."""
        return np.array([[0, 0]] * self.T, dtype=np.float32)

    @property
    def _path_phi1_broken(self):
        """Leave phi1 into neutral space, then arrive at phi2."""
        return np.array(
            [[0, 0], [0, 0]]
            + [[10, 10]] * 4          # far from both regions
            + [[4, 4]] * 5,
            dtype=np.float32,
        )

    # --- tests ---

    def test_until_satisfying(self):
        """phi1 holds, then phi2 is reached → robustness > 0."""
        gt = self._stlpy_robustness(self.until_form, self._path_sat)
        rho = self._jax_eval(self.until_form, self._path_sat)
        self.assertGreater(gt, 0, f"stlpy GT should be positive, got {gt}")
        self.assertGreater(rho, 0, f"JAX until() should be positive on sat path, got {rho}")

    def test_until_unsat_phi2_never_reached(self):
        """phi2 is never reached → robustness < 0."""
        gt = self._stlpy_robustness(self.until_form, self._path_never_phi2)
        rho = self._jax_eval(self.until_form, self._path_never_phi2)
        self.assertLess(gt, 0, f"stlpy GT should be negative, got {gt}")
        self.assertLess(rho, 0, f"JAX until() should be negative on never-phi2 path, got {rho}")

    def test_until_unsat_phi1_broken_before_phi2(self):
        """phi1 is violated before phi2 is reached → robustness < 0."""
        gt = self._stlpy_robustness(self.until_form, self._path_phi1_broken)
        rho = self._jax_eval(self.until_form, self._path_phi1_broken)
        self.assertLess(gt, 0, f"stlpy GT should be negative, got {gt}")
        self.assertLess(rho, 0, f"JAX until() should be negative on phi1-broken path, got {rho}")

    def test_until_leaf_vs_nonleaf_phi2(self):
        """Wrapping phi2 in a trivial conjunction must not change the semantics."""
        leaf = self.phi1.until(self.phi2, *self.interval)
        compound = self.phi1.until(self.phi2 & self.phi2, *self.interval)
        for name, p in [("sat", self._path_sat), ("never_phi2", self._path_never_phi2)]:
            leaf_rho = self._jax_eval(leaf, p)
            comp_rho = self._jax_eval(compound, p)
            self.assertTrue(
                np.sign(leaf_rho) == np.sign(comp_rho) or abs(leaf_rho - comp_rho) < 1e-2,
                f"leaf vs non-leaf phi2 disagree on {name}: leaf={leaf_rho}, compound={comp_rho}",
            )

    def test_until_leaf_vs_nonleaf_phi1(self):
        """Same, wrapping phi1."""
        leaf = self.phi1.until(self.phi2, *self.interval)
        compound = (self.phi1 & self.phi1).until(self.phi2, *self.interval)
        for name, p in [("sat", self._path_sat), ("phi1_broken", self._path_phi1_broken)]:
            leaf_rho = self._jax_eval(leaf, p)
            comp_rho = self._jax_eval(compound, p)
            self.assertTrue(
                np.sign(leaf_rho) == np.sign(comp_rho) or abs(leaf_rho - comp_rho) < 1e-2,
                f"leaf vs non-leaf phi1 disagree on {name}: leaf={leaf_rho}, compound={comp_rho}",
            )

    def test_until_nested_in_always(self):
        """G_[0,1] (phi1 U_[0,9] phi2) — exercises non-zero start_t in until."""
        path = np.concatenate([
            np.array([[0, 0]] * 4, dtype=np.float32),
            np.array([[4, 4]] * 7, dtype=np.float32),
        ])
        form = self.phi1.until(self.phi2, 0, 9).always(0, 1)
        gt = self._stlpy_robustness(form, path)
        rho = self._jax_eval(form, path)
        self.assertGreater(gt, 0, f"stlpy GT should be positive for nested until, got {gt}")
        self.assertGreater(rho, 0, f"JAX nested until should be positive, got {rho}")


if __name__ == '__main__':
    unittest.main()
