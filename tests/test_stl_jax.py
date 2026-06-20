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


import ds.stl_jax as stl_jax


class TestApproxMethodsUnit(unittest.TestCase):
    """Unit tests for the approx_method switch on _tensor_min / _tensor_max.

    Three methods: "softmax" (legacy default, weighted-sum), "logsumexp"
    (mass-conserving smooth max), "true" (exact jnp.max/min, gradient
    distributed across ties). hardness/approx_method are per-call overrides.
    """

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"
        importlib.reload(ds_utils)
        self.form = STL(RectReachPredicate(np.array([0, 0]), np.array([2, 2]), "p"))
        self.arr = jnp.array([0.1, -0.3, 0.5, -0.2, 0.05])
        self._orig_hardness = stl_jax.HARDNESS
        self._orig_method = stl_jax.APPROX_METHOD

    def tearDown(self):
        stl_jax.HARDNESS = self._orig_hardness
        stl_jax.APPROX_METHOD = self._orig_method

    def test_true_equals_exact(self):
        """approx_method='true' is bit-for-bit jnp.max / jnp.min."""
        mx = self.form._tensor_max(self.arr, axis=-1, approx_method="true")
        mn = self.form._tensor_min(self.arr, axis=-1, approx_method="true")
        self.assertAlmostEqual(float(mx), float(jnp.max(self.arr)), places=6)
        self.assertAlmostEqual(float(mn), float(jnp.min(self.arr)), places=6)

    def test_ordering_inequalities(self):
        """softmax_max <= true_max <= logsumexp_max ; mins reversed."""
        h = 10.0
        smax = float(self.form._tensor_max(self.arr, hardness=h, approx_method="softmax"))
        tmax = float(self.form._tensor_max(self.arr, hardness=h, approx_method="true"))
        lmax = float(self.form._tensor_max(self.arr, hardness=h, approx_method="logsumexp"))
        self.assertLessEqual(smax, tmax + 1e-6)
        self.assertLessEqual(tmax, lmax + 1e-6)
        smin = float(self.form._tensor_min(self.arr, hardness=h, approx_method="softmax"))
        tmin = float(self.form._tensor_min(self.arr, hardness=h, approx_method="true"))
        lmin = float(self.form._tensor_min(self.arr, hardness=h, approx_method="logsumexp"))
        self.assertGreaterEqual(smin, tmin - 1e-6)
        self.assertGreaterEqual(tmin, lmin - 1e-6)

    def test_default_is_legacy_softmax(self):
        """Default (no override) must equal the legacy softmax-weighted-sum formula."""
        from jax.nn import softmax
        h = stl_jax.HARDNESS
        legacy_max = float(jnp.sum(self.arr * softmax(self.arr * h, axis=-1), axis=-1))
        legacy_min = float(jnp.sum(self.arr * softmax(self.arr * -h, axis=-1), axis=-1))
        self.assertAlmostEqual(float(self.form._tensor_max(self.arr)), legacy_max, places=5)
        self.assertAlmostEqual(float(self.form._tensor_min(self.arr)), legacy_min, places=5)

    def test_hardness_override_is_live(self):
        """Higher hardness pulls softmax-max toward the true max (proves arg is used)."""
        lo = float(self.form._tensor_max(self.arr, hardness=1.0, approx_method="softmax"))
        hi = float(self.form._tensor_max(self.arr, hardness=100.0, approx_method="softmax"))
        true = float(jnp.max(self.arr))
        self.assertLess(abs(hi - true), abs(lo - true))

    def test_gradient_finite_all_methods(self):
        for m in ("softmax", "logsumexp", "true"):
            g = jax.grad(lambda a: self.form._tensor_max(a, hardness=10.0, approx_method=m))(self.arr)
            self.assertTrue(bool(jnp.isfinite(g).all()), f"{m} max grad non-finite")
            g2 = jax.grad(lambda a: self.form._tensor_min(a, hardness=10.0, approx_method=m))(self.arr)
            self.assertTrue(bool(jnp.isfinite(g2).all()), f"{m} min grad non-finite")

    def test_invalid_method_raises(self):
        with self.assertRaises((ValueError, KeyError)):
            float(self.form._tensor_max(self.arr, approx_method="bogus"))


class TestApproxMethodsEval(unittest.TestCase):
    """End-to-end: per-call hardness/approx_method flow through eval into nested temporal ops."""

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"
        importlib.reload(ds_utils)
        self.phi1 = STL(RectReachPredicate(np.array([0, 0]), np.array([2, 2]), "phi1"))
        self.phi2 = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "phi2"))
        self.T = 11
        self.until_form = self.phi1.until(self.phi2, 0, self.T - 1)
        # stl_mixin-style compound nested temporal (eventually & eventually).always
        g0 = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "g0"))
        g1 = STL(RectReachPredicate(np.array([2, 2]), np.array([1, 1]), "g1"))
        self.compound = (g0.eventually(0, 5) & g1.eventually(0, 5)).always(0, 8)
        self._orig_method = stl_jax.APPROX_METHOD

    def tearDown(self):
        stl_jax.APPROX_METHOD = self._orig_method

    def _path(self, pts):
        return ds_utils.default_tensor(np.array(pts, dtype=np.float32)[None])

    @property
    def _sat(self):
        return self._path([[0, 0]] * 5 + [[4, 4]] * 6)

    @property
    def _unsat(self):
        return self._path([[0, 0]] * self.T)  # never reaches phi2

    def test_sign_all_methods(self):
        """until sign correctness preserved under every approx_method."""
        for m in ("softmax", "logsumexp", "true"):
            sat = float(self.until_form.eval(self._sat, approx_method=m).squeeze())
            uns = float(self.until_form.eval(self._unsat, approx_method=m).squeeze())
            self.assertGreater(sat, 0, f"{m}: sat should be > 0, got {sat}")
            self.assertLess(uns, 0, f"{m}: unsat should be < 0, got {uns}")

    def test_default_matches_explicit_softmax(self):
        """Backward-compat: no-arg eval == explicit softmax at module HARDNESS."""
        d = float(self.until_form.eval(self._sat).squeeze())
        e = float(self.until_form.eval(
            self._sat, hardness=stl_jax.HARDNESS, approx_method="softmax").squeeze())
        self.assertAlmostEqual(d, e, places=5)

    def test_approx_override_propagates(self):
        """softmax vs true differ on a varied robustness trace -> approx_method override
        reaches the temporal reduction (eventually). true (exact max) >= softmax-weighted."""
        goal = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "gr"))
        form = goal.eventually(0, 10)
        ramp = self._path([[i * 0.4, i * 0.4] for i in range(11)])  # [0,0] -> [4,4]
        soft = float(form.eval(ramp, hardness=2.0, approx_method="softmax").squeeze())
        true = float(form.eval(ramp, hardness=2.0, approx_method="true").squeeze())
        self.assertTrue(np.isfinite(soft) and np.isfinite(true))
        self.assertNotAlmostEqual(soft, true, places=3)
        self.assertGreaterEqual(true, soft - 1e-6)

    def test_hardness_override_reaches_nested(self):
        """Varying hardness changes the compound value -> hardness propagated into nested ops."""
        path = self._path([[0.5, 0.5], [1.5, 1.5]] * 4 + [[1.0, 1.0]])
        lo = float(self.compound.eval(path, hardness=1.0, approx_method="softmax").squeeze())
        hi = float(self.compound.eval(path, hardness=100.0, approx_method="softmax").squeeze())
        self.assertNotAlmostEqual(lo, hi, places=3)

    def test_hardness_is_traced_arg(self):
        """A traced hardness scalar varies output without error (single-trace dynamic arg)."""
        vals = [float(self.until_form.eval(self._sat, hardness=jnp.asarray(h),
                                           approx_method="logsumexp").squeeze())
                for h in (1.0, 5.0, 50.0)]
        self.assertGreaterEqual(len(set(np.round(vals, 4))), 2, f"hardness not live: {vals}")

    def test_module_global_opt_in(self):
        """Setting ds.stl_jax.APPROX_METHOD before .eval is honored (stl_mixin opt-in pattern)."""
        stl_jax.APPROX_METHOD = "true"
        try:
            v_global = float(self.until_form.eval(self._sat).squeeze())
        finally:
            stl_jax.APPROX_METHOD = self._orig_method
        v_explicit = float(self.until_form.eval(self._sat, approx_method="true").squeeze())
        self.assertAlmostEqual(v_global, v_explicit, places=5)

    def test_gradient_flows_each_method(self):
        for m in ("softmax", "logsumexp", "true"):
            g = jax.grad(lambda p: self.until_form.eval(
                p, hardness=20.0, approx_method=m).mean())(self._sat)
            self.assertTrue(bool(jnp.isfinite(g).all()), f"{m}: grad non-finite")


class TestPerStepSchedule(unittest.TestCase):
    """Soft/dense early -> sharp logsumexp late: logsumexp crosses the boundary
    where softmax-weighted-sum plateaus, at the same hardness."""

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"
        importlib.reload(ds_utils)
        self.goal = STL(RectReachPredicate(np.array([4, 4]), np.array([2, 2]), "g"))
        self.form = self.goal.eventually(0, 10)

    def _optimize(self, method, hardness, steps=150, lr=0.2):
        import optax
        T = 11
        path0 = ds_utils.default_tensor(np.full((1, T, 2), 10.0, dtype=np.float32))
        loss = lambda p: -self.form.eval(p, hardness=hardness, approx_method=method).mean()
        opt = optax.adam(lr)
        st = opt.init(path0)
        p = path0
        best = float(-loss(p))
        for _ in range(steps):
            g = jax.grad(loss)(p)
            u, st = opt.update(g, st)
            p = optax.apply_updates(p, u)
            best = max(best, float(-loss(p)))
        return best

    def test_logsumexp_crosses_boundary(self):
        """logsumexp reaches satisfaction and is no worse than softmax at the same hardness."""
        soft = self._optimize("softmax", hardness=10.0)
        lse = self._optimize("logsumexp", hardness=10.0)
        self.assertGreater(lse, 0.0, f"logsumexp should reach sat robustness, got {lse}")
        self.assertGreaterEqual(lse, soft - 1e-3, f"logsumexp={lse} softmax={soft}")


if __name__ == '__main__':
    unittest.main()
