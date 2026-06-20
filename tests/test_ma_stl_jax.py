import importlib
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

os.environ["DIFF_STL_BACKEND"] = "jax"  # So ds_utils does not require torch

import ds.utils as ds_utils
from ds.stl_jax import STL, RectReachPredicate
from ds.ma_stl_jax import CaTLPlus, Task
import examples.stl.differentiability as stl_diff_examples

TEST_TOLERANCE = 1e-3  # Small number close to 0


class TestMASTLJAXExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
        importlib.reload(ds_utils)  # Reload the module to reset the backend

        self.key = jax.random.PRNGKey(0)

        self.goal_1 = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "goal_1"))
        # goal_2 is a rectangle area centered in [2, 2] with width and height 1
        self.goal_2 = STL(RectReachPredicate(np.array([2, 2]), np.array([1, 1]), "goal_2"))

        # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
        # and that holds always in 0 to 8
        # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
        self.form = (self.goal_1.eventually(0, 5) & self.goal_2.eventually(0, 5)).always(0, 8)
        self.loop_form = (self.goal_1.eventually(0, 4) & self.goal_2.eventually(0, 4)).always(0, 8)
        self.cover_form = self.goal_1.eventually(0, 12) & self.goal_2.eventually(0, 12)
        self.seq_form = self.goal_1.eventually(0, 6) & self.goal_2.eventually(6, 12)
        self.seq_inv_form = self.goal_2.eventually(0, 6) & self.goal_1.eventually(6, 12)

        self.num_satisfied_agents = 2
        task = Task(1, self.form, self.num_satisfied_agents, None)
        task_seq = Task(2, self.seq_form, self.num_satisfied_agents, None)
        task_seq_inv = Task(3, self.seq_inv_form, self.num_satisfied_agents, None)
        task_cover = Task(4, self.cover_form, self.num_satisfied_agents, None)
        task_loop = Task(5, self.loop_form, self.num_satisfied_agents, None)
        TASK_NAMES = ["task", "task_seq", "task_seq_inv", "task_cover", "task_loop"]
        self.catl_form = CaTLPlus(task)
        self.catl_form_or = CaTLPlus(task) | CaTLPlus(task_seq)
        self.catl_form_and = CaTLPlus(task_cover) & CaTLPlus(task_seq)
        self.catl_form_3and = CaTLPlus(task_cover) & CaTLPlus(task_seq) & CaTLPlus(task_seq_inv)

        self.all_catl_forms = [self.catl_form, self.catl_form_or, self.catl_form_and, self.catl_form_3and]

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

        path_sat = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0, 0],
                        [2, 2],
                        [0, 1],
                        [0, 1],
                        [0, 0],
                        [2, 2],
                        [0, 0],
                        [2, 2],
                        [0, 1],
                        [0, 0],
                        [2, 2],
                    ],
                ]
            )
        )

        path_sat_seq = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [2, 2],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ],
                ]
            )
        )

        path_sat_seq_inv = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [2, 2],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ],
                ]
            )
        )

        path_sat_cover = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [0, 0],
                        [0, 0],
                        [2, 2],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ],
                ]
            )
        )

        input_path = jax.numpy.tile(path, (num_tiles, 1, 1))

        loss = self.form.eval(input_path)  # Make a batch of size num_tiles

        paths = {
            "not_satisfy": (jnp.concatenate([input_path, path_sat], axis=0), -1),
            "satisfy": (jnp.concatenate([input_path, path_sat, path_sat], axis=0), +1),
            "satisfy_seq": (jnp.concatenate([input_path, path_sat_seq, path_sat_seq], axis=0), +1),
            "satisfy_seq_and": (jnp.concatenate([input_path, path_sat_cover, path_sat_cover,
                                                 path_sat_seq, path_sat_seq], axis=0), +1),
            "satisfy_seq_3and": (jnp.concatenate([input_path, path_sat_cover, path_sat_seq, path_sat_seq,
                                                  path_sat_seq_inv, path_sat_seq_inv], axis=0), +1),
            "notsatisfy_seq_and": (jnp.concatenate([input_path, path_sat_cover, path_sat_cover, path_sat_cover,
                                                    path_sat_seq], axis=0), -1),
            "notsatisfy_seq_3and": (jnp.concatenate([input_path, path_sat_seq, path_sat_seq,
                                                     path_sat_seq_inv], axis=0), -1),
            "not_satisfy_seq": (jnp.concatenate([input_path, path_sat_seq, input_path], axis=0), -1),
        }

        # Define the test cases: each is a tuple with the form to use, the key from paths, and a custom message.
        test_cases = [
            (self.catl_form, "not_satisfy", "Not returning correct value"),
            (self.catl_form, "satisfy", "Not returning correct value"),
            (self.catl_form_or, "satisfy", "Not returning correct value for or sat path"),
            (self.catl_form_or, "satisfy_seq", "Not returning correct value for or sat path"),
            (self.catl_form_or, "not_satisfy", "Not returning correct value for or unsat path"),
            (self.catl_form_or, "not_satisfy_seq", "Not returning correct value for or unsat path"),
            (self.catl_form_and, "notsatisfy_seq_and", "Not returning correct value for and unsat path"),
            (self.catl_form_and, "satisfy_seq_and", "Not returning correct value for and sat path"),
            (self.catl_form_3and, "notsatisfy_seq_3and", "Not returning correct value for 3and unsat path"),
            (self.catl_form_3and, "satisfy_seq_3and", "Not returning correct value for 3and sat path"),
        ]

        # Helper function to evaluate and assert based on expected outcome.

        form_evals = list(map(lambda x: x[0].eval, test_cases))
        paths_to_test_list = list(map(lambda x: paths[x[1]][0], test_cases))

        def check_eval(i_eval, path_data, train_mode=False):
            return form_evals[i_eval](path_data, train_mode=train_mode)

        def eval_result(_loss, path_key, train_mode):
            path_data, expected_sign = paths[path_key]
            if expected_sign > 0:
                self.assertGreater(_loss, 0, f"{path_key} failed in train_mode={train_mode}")
            else:
                self.assertLess(_loss, 0, f"{path_key} failed in train_mode={train_mode}")

        # Iterate over test cases for both training modes
        for train_mode in [False, True]:
            # check_eval_fn = jit(ft.partial(check_eval, train_mode=train_mode))
            # TODO: jit over all the test cases
            # Now check the results
            for i, path in enumerate(paths_to_test_list):
                loss = check_eval(i, path, train_mode)
                eval_result(loss, test_cases[i][1], train_mode)

    def test_repr(self):
        print(self.form)
        for form in self.all_catl_forms:
            print(form)

    def test_run(self):

        # Test differentiability
        for catl_form in self.all_catl_forms:
            path, loss = stl_diff_examples.mabackward(jax_key=self.key, ma_stl_spec=catl_form)
            print('loss', loss)
            assert loss < TEST_TOLERANCE  # Loss should be less than 0 to satisfy the formula

    def _test_avoid_backward(self):
        path, loss = stl_diff_examples.backward(self.key, avoid_spec=True)
        print('AvoidPath', loss, path)
        assert loss < TEST_TOLERANCE  # Loss should be less than 0 to satisfy the formula

    def _test_loop(self):
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


class TestCaTLPlusUntilSemantics(unittest.TestCase):
    """Targeted semantics tests for CaTLPlus's `until` operator.

    Should mirror `STL.until` for tasks with a single agent and
    `num_satisfied_agents=1`: standard-STL robustness, differentiable,
    sign-consistent with stlpy.
    """

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"
        importlib.reload(ds_utils)

        # Task names chosen outside the 0..6 op-code range to avoid any
        # residual collisions in flatten-style traversals.
        self.phi1_stl = STL(RectReachPredicate(np.array([0, 0]), np.array([2, 2]), "phi1"))
        self.phi2_stl = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "phi2"))
        self.task_A = Task(10, self.phi1_stl, 1, None)
        self.task_B = Task(11, self.phi2_stl, 1, None)

        self.A = CaTLPlus(self.task_A)
        self.B = CaTLPlus(self.task_B)
        self.T = 11

    def _batch(self, np_path_per_agent):
        """np_path_per_agent has shape (num_agents, T, d); return a jax tensor."""
        return ds_utils.default_tensor(np_path_per_agent)

    def test_until_satisfying(self):
        """Agent stays in phi1 then reaches phi2 → robustness > 0."""
        path = self._batch(np.array([[[0, 0]] * 5 + [[4, 4]] * 6], dtype=np.float32))
        form = self.A.until(self.B, 0, self.T - 1)
        rho = float(form.eval(path))
        self.assertGreater(rho, 0, f"sat path should have positive robustness, got {rho}")

    def test_until_unsat_phi2_never_reached(self):
        """Agent stays in phi1, never reaches phi2 → robustness < 0.

        The pre-fix code returned a spurious +1.0 here.
        """
        path = self._batch(np.array([[[0, 0]] * self.T], dtype=np.float32))
        form = self.A.until(self.B, 0, self.T - 1)
        rho = float(form.eval(path))
        self.assertLess(rho, 0, f"never-phi2 path should be negative, got {rho}")

    def test_until_unsat_phi1_broken_before_phi2(self):
        """Agent leaves phi1 before phi2 arrives → robustness < 0."""
        path = self._batch(
            np.array([[
                [0, 0], [0, 0],
                [10, 10], [10, 10], [10, 10], [10, 10],
                [4, 4], [4, 4], [4, 4], [4, 4], [4, 4],
            ]], dtype=np.float32)
        )
        form = self.A.until(self.B, 0, self.T - 1)
        rho = float(form.eval(path))
        self.assertLess(rho, 0, f"phi1-broken path should be negative, got {rho}")

    def test_until_multi_agent_m_of_n(self):
        """With num_satisfied_agents=2, both agents must contribute to the witness."""
        task_A2 = Task(12, self.phi1_stl, 2, None)
        task_B2 = Task(13, self.phi2_stl, 2, None)
        A2 = CaTLPlus(task_A2)
        B2 = CaTLPlus(task_B2)

        # Two agents both follow the satisfying pattern.
        ok_path = self._batch(np.stack([
            np.array([[0, 0]] * 5 + [[4, 4]] * 6, dtype=np.float32),
            np.array([[0, 0]] * 5 + [[4, 4]] * 6, dtype=np.float32),
        ], axis=0))
        # Only one agent ever reaches phi2, so the 2-of-2 witness must fail.
        bad_path = self._batch(np.stack([
            np.array([[0, 0]] * 5 + [[4, 4]] * 6, dtype=np.float32),
            np.array([[0, 0]] * self.T, dtype=np.float32),
        ], axis=0))

        form = A2.until(B2, 0, self.T - 1)
        self.assertGreater(float(form.eval(ok_path)), 0)
        self.assertLess(float(form.eval(bad_path)), 0)


import ds.ma_stl_jax as ma_stl_jax


class TestCaTLPlusApproxMethods(unittest.TestCase):
    """Per-call hardness/approx_method threading on CaTLPlus.eval.

    CaTLPlus has two default hardness sources: ma_stl_jax.HARDNESS for the
    CaTLPlus-level soft reductions, and stl_jax.HARDNESS for the STL inside
    each Task. Threading is lazy (None -> each level's own global) so the
    no-arg path stays byte-identical; explicit values override everything
    (CaTLPlus ops AND the inner STL).
    """

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"
        importlib.reload(ds_utils)
        self.T = 11
        phi1 = STL(RectReachPredicate(np.array([0, 0]), np.array([2, 2]), "phi1"))
        phi2 = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "phi2"))
        self.A = CaTLPlus(Task(10, phi1, 1, None))
        self.B = CaTLPlus(Task(11, phi2, 1, None))
        self.until_form = self.A.until(self.B, 0, self.T - 1)
        # Task wrapping an inner-STL eventually — exercises threading through Task -> STL.
        goal = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "g"))
        self.inner_temporal = CaTLPlus(Task(12, goal.eventually(0, self.T - 1), 1, None))
        self._orig_method = ma_stl_jax.APPROX_METHOD

    def tearDown(self):
        ma_stl_jax.APPROX_METHOD = self._orig_method

    def _path(self, pts):
        # CaTLPlus path is (num_agents, T, d); single agent here.
        return ds_utils.default_tensor(np.array(pts, dtype=np.float32)[None])

    @property
    def _sat(self):
        return self._path([[0, 0]] * 5 + [[4, 4]] * 6)

    @property
    def _unsat(self):
        return self._path([[0, 0]] * self.T)

    def test_default_matches_explicit_softmax(self):
        """no-arg eval == explicit approx='softmax' (hardness left lazy) -> approx default unchanged."""
        d = float(self.until_form.eval(self._sat))
        e = float(self.until_form.eval(self._sat, approx_method="softmax"))
        self.assertAlmostEqual(d, e, places=5)

    def test_train_mode_default_unchanged(self):
        """Exponential-robustness (scoring) path unchanged by default approx."""
        d = float(self.until_form.eval_train(self._sat))
        e = float(self.until_form.eval_train(self._sat, approx_method="softmax"))
        self.assertAlmostEqual(d, e, places=5)
        self.assertTrue(np.isfinite(d))

    def test_sign_all_methods(self):
        for m in ("softmax", "logsumexp", "true"):
            sat = float(self.until_form.eval(self._sat, approx_method=m))
            uns = float(self.until_form.eval(self._unsat, approx_method=m))
            self.assertGreater(sat, 0, f"{m}: sat should be > 0, got {sat}")
            self.assertLess(uns, 0, f"{m}: unsat should be < 0, got {uns}")

    def test_approx_propagates_into_inner_stl(self):
        """softmax vs true differ on a Task's inner-STL eventually -> threading reaches STL."""
        ramp = self._path([[i * 0.4, i * 0.4] for i in range(self.T)])
        soft = float(self.inner_temporal.eval(ramp, hardness=2.0, approx_method="softmax"))
        true = float(self.inner_temporal.eval(ramp, hardness=2.0, approx_method="true"))
        self.assertTrue(np.isfinite(soft) and np.isfinite(true))
        self.assertNotAlmostEqual(soft, true, places=3)

    def test_hardness_override_is_live(self):
        """Varying hardness changes until robustness -> hardness reaches CaTLPlus _tensor_*."""
        lo = float(self.until_form.eval(self._sat, hardness=1.0))
        hi = float(self.until_form.eval(self._sat, hardness=100.0))
        self.assertNotAlmostEqual(lo, hi, places=3)

    def test_module_global_opt_in(self):
        """Set ds.ma_stl_jax.APPROX_METHOD before first trace of a fresh form -> honored."""
        phi1 = STL(RectReachPredicate(np.array([0, 0]), np.array([2, 2]), "phi1"))
        phi2 = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "phi2"))
        ma_stl_jax.APPROX_METHOD = "true"
        try:
            fresh = CaTLPlus(Task(10, phi1, 1, None)).until(CaTLPlus(Task(11, phi2, 1, None)),
                                                            0, self.T - 1)
            v_global = float(fresh.eval(self._sat))
        finally:
            ma_stl_jax.APPROX_METHOD = self._orig_method
        v_explicit = float(self.until_form.eval(self._sat, approx_method="true"))
        self.assertAlmostEqual(v_global, v_explicit, places=5)

    def test_gradient_flows_each_method(self):
        for m in ("softmax", "logsumexp", "true"):
            g = jax.grad(lambda p: self.until_form.eval(p, hardness=20.0, approx_method=m))(self._sat)
            self.assertTrue(bool(jnp.isfinite(g).all()), f"{m}: grad non-finite")


if __name__ == '__main__':
    unittest.main()
