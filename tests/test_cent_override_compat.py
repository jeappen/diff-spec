"""Merge-safety tests for the traced ``cent_override`` feature.

Written while evaluating the merge of ``feature/traced-cent-override`` into ``feature/jax``.
Three groups:

1. **Golden values** captured on ``feature/jax@902e50d`` (pre-merge) for the default,
   no-override path. They pin ``eval`` / ``eval_train`` / per-call hardness+approx /
   gradient numerics of the formula shapes gcbfplus builds (sequential F, G(F) "signal",
   F(G), avoid+boundary, negation-until, or). Runs on both branches.
2. **end_time() horizon** semantics: ``end_time()`` must be a sufficient path length
   (no "end_t is larger than motion length" warning). Pins the nested-F fix.
3. **cent_override** behaviour. Skipped automatically when the installed ``ds.stl_jax``
   lacks the kwarg, so the file is also runnable on pre-merge ``feature/jax``.

Run: ``python -m unittest tests.test_cent_override_compat``
"""
import inspect
import os
import unittest

os.environ["DIFF_STL_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np

import ds.stl_jax as sj
from ds.stl_jax import STL, RectAvoidPredicate, RectReachPredicate

HAS_OVERRIDE = "cent_override" in inspect.signature(STL.eval).parameters

T, B = 40, 4
GS = np.array([0.5, 0.5])
CENTS = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
GOLDEN_HARDNESS = 100.0

# Captured on feature/jax@902e50d, GPU float32, HARDNESS=100, APPROX_METHOD="softmax",
# path = default_rng(0).uniform(-2, 2, (4, 40, 2)).astype(float32).
GOLDEN = {
    'seq': dict(eval=[-0.3234887719154358, -0.5211422443389893, -0.2703758180141449, -0.31669875979423523],
                eval_train=[-0.3234887719154358, -0.5211422443389893, -0.2703758180141449, -0.31669875979423523],
                eval_lse_h5=[-0.35530510544776917, -0.41858378052711487, -0.30117636919021606, -0.35528403520584106],
                grad_train_abs_sum=4.031424045562744),
    'signal_GF': dict(eval=[-1.0592446327209473, -1.02889084815979, -0.7796295881271362, -1.0618036985397339],
                      eval_train=[-1.0592446327209473, -1.02889084815979, -0.7796295881271362, -1.0618036985397339],
                      eval_lse_h5=[-1.4448292255401611, -1.3594852685928345, -1.1486936807632446, -1.1595008373260498],
                      grad_train_abs_sum=4.256439208984375),
    'fg': dict(eval=[-1.2129963636398315, -1.5963151454925537, -1.6706528663635254, -1.0535740852355957],
               eval_train=[-1.311086893081665, -2.0364673137664795, -1.7217929363250732, -1.106410026550293],
               eval_lse_h5=[-1.2615270614624023, -1.4225958585739136, -1.480137586593628, -1.012618899345398],
               grad_train_abs_sum=5.3112897872924805),
    'avoid_bnd': dict(eval=[-0.15094654262065887, -0.24406559765338898, -0.22591249644756317, -0.009385832585394382],
                      eval_train=[-0.15094654262065887, -0.24406559765338898, -0.22591249644756317, -0.009385832585394382],
                      eval_lse_h5=[-0.4876467287540436, -0.5006635785102844, -0.4806508719921112, -0.4822521209716797],
                      grad_train_abs_sum=4.493091583251953),
    'auntil': dict(eval=[-0.1509343981742859, -0.30974242091178894, -0.2703925371170044, 0.0769408792257309],
                   eval_train=[-0.1509343981742859, -0.30974242091178894, -0.2703925371170044, 0.0769408792257309],
                   eval_lse_h5=[-0.053738441318273544, -0.16334830224514008, -0.1767355054616928, 0.015782607719302177],
                   grad_train_abs_sum=4.014796733856201),
    'or': dict(eval=[-0.15111230313777924, -0.2927015721797943, -0.18762800097465515, 0.08367502689361572],
               eval_train=[-0.22415615618228912, -0.296772301197052, -0.2094118744134903, 0.021541355177760124],
               eval_lse_h5=[0.11988847702741623, 0.040848709642887115, 0.1012415811419487, 0.21584074199199677],
               grad_train_abs_sum=5.055286407470703),
}


def reach(cent, name, shrink=1.0):
    return STL(RectReachPredicate(np.array(cent, float), GS, name, shrink))


def build_forms(cents=CENTS):
    """Formula shapes mirroring gcbfplus stl_mixin builders."""
    g0, g1, g2 = reach(cents[0], 0), reach(cents[1], 1), reach(cents[2], 2)
    obs = STL(RectAvoidPredicate(np.array([0., 0.]), np.array([0.6, 0.6]), 7))
    bnd = STL(RectReachPredicate(np.array([0., 0.]), np.array([4., 4.]), -1))  # boundary: name < 0
    return {
        'seq': g0.eventually(0, 10) & g1.eventually(10, 20) & g2.eventually(20, T),
        'signal_GF': (g0.eventually(0, 10) & g1.eventually(0, 10)).always(0, 30),
        'fg': g0.always(0, 5).eventually(0, 35),
        'avoid_bnd': reach(cents[0], 0, 0.8).eventually(0, T) & obs.always(0, T) & bnd.always(0, T),
        'auntil': (~g1).until(g0, 0, 20) & g1.eventually(20, T),
        'or': g0.eventually(0, 20) | g1.eventually(0, 20),
    }


def golden_path():
    return jnp.asarray(np.random.default_rng(0).uniform(-2, 2, size=(B, T, 2)).astype(np.float32))


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # HARDNESS is resolved per eval() call (not baked at trace time) so this is safe here.
        cls._saved = (sj.HARDNESS, sj.APPROX_METHOD)
        sj.HARDNESS, sj.APPROX_METHOD = GOLDEN_HARDNESS, "softmax"
        cls.path = golden_path()
        cls.forms = build_forms()

    @classmethod
    def tearDownClass(cls):
        sj.HARDNESS, sj.APPROX_METHOD = cls._saved


class TestGoldenDefaultPath(_Base):
    """Default (no override) numerics must match pre-merge feature/jax."""

    def _check(self, name):
        f, g = self.forms[name], GOLDEN[name]
        np.testing.assert_allclose(np.asarray(f.eval(self.path)), g['eval'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(f.eval_train(self.path)), g['eval_train'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(f.eval(self.path, hardness=5.0, approx_method='logsumexp')),
                                   g['eval_lse_h5'], rtol=1e-5, atol=1e-6)
        grad = jax.grad(lambda p: f.eval_train(p).sum())(self.path)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        np.testing.assert_allclose(float(jnp.abs(grad).sum()), g['grad_train_abs_sum'], rtol=1e-4)

    def test_seq(self): self._check('seq')
    def test_signal_GF(self): self._check('signal_GF')
    def test_fg(self): self._check('fg')
    def test_avoid_bnd(self): self._check('avoid_bnd')
    def test_auntil(self): self._check('auntil')
    def test_or(self): self._check('or')


class TestEndTimeHorizon(_Base):
    """end_time() is what gcbfplus divides max_step by; it must cover every index _eval reads."""

    def test_leaf_and_flat_shapes_unchanged(self):
        self.assertEqual(self.forms['seq'].end_time(), T)
        self.assertEqual(self.forms['signal_GF'].end_time(), T)      # G[0,30](F[0,10]) = 40, as before
        self.assertEqual(self.forms['avoid_bnd'].end_time(), T)
        self.assertEqual(self.forms['or'].end_time(), 20)
        self.assertEqual(reach(CENTS[0], 0).eventually(T - 1, T).end_time(), T)

    def test_nested_F_over_G(self):
        # F[0,35](G[0,5] g) reads up to t=39 -> horizon 40. Pre-merge feature/jax reported 35.
        self.assertEqual(self.forms['fg'].end_time(), 40)

    def test_end_time_is_sufficient_horizon(self):
        for name, f in self.forms.items():
            with self.subTest(name=name):
                p = self.path[:, :f.end_time()]
                with self.assertNoLogs('ds.stl_jax', level='WARNING'):
                    jax.block_until_ready(f.eval(p))


@unittest.skipUnless(HAS_OVERRIDE, "ds.stl_jax has no cent_override (pre-merge feature/jax)")
class TestCentOverride(_Base):
    OV = jnp.asarray(CENTS, dtype=jnp.float32)
    NEW = jnp.asarray(CENTS + np.array([0.3, -0.2]), dtype=jnp.float32)

    def test_override_with_baked_values_matches_baked(self):
        for name, f in self.forms.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(np.asarray(f.eval(self.path, cent_override=self.OV)),
                                           np.asarray(f.eval(self.path)), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(np.asarray(f.eval_train(self.path, cent_override=self.OV)),
                                           np.asarray(f.eval_train(self.path)), rtol=1e-6, atol=1e-6)
                gb = jax.grad(lambda p: f.eval_train(p).sum())(self.path)
                go = jax.grad(lambda p: f.eval_train(p, cent_override=self.OV).sum())(self.path)
                np.testing.assert_allclose(np.asarray(go), np.asarray(gb), rtol=1e-6, atol=1e-6)

    def test_override_matches_rebuilt_formula(self):
        rebuilt = build_forms(np.asarray(self.NEW))
        for name, f in self.forms.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(np.asarray(f.eval(self.path, cent_override=self.NEW)),
                                           np.asarray(rebuilt[name].eval(self.path)), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(np.asarray(f.eval_train(self.path, cent_override=self.NEW)),
                                           np.asarray(rebuilt[name].eval_train(self.path)), rtol=1e-6, atol=1e-6)

    def test_avoid_and_boundary_ignore_override(self):
        # rows for obstacle (name 7) and boundary (name -1) are garbage; only goal 0 moves.
        ov = jnp.full((8, 2), 99.0, dtype=jnp.float32).at[0].set(self.NEW[0])
        rebuilt = build_forms(np.asarray(self.NEW))['avoid_bnd']
        np.testing.assert_allclose(np.asarray(self.forms['avoid_bnd'].eval(self.path, cent_override=ov)),
                                   np.asarray(rebuilt.eval(self.path)), rtol=1e-6, atol=1e-6)

    def test_tuple_override_sizes(self):
        sizes = jnp.full((4, 2), 0.9, dtype=jnp.float32)
        big = STL(RectReachPredicate(np.asarray(self.NEW[0]), np.array([0.9, 0.9]), 0)).eventually(0, T)
        f = reach(CENTS[0], 0).eventually(0, T)
        np.testing.assert_allclose(np.asarray(f.eval(self.path, cent_override=(self.NEW, sizes))),
                                   np.asarray(big.eval(self.path)), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(f.eval_train(self.path, cent_override=(self.NEW, sizes))),
                                   np.asarray(big.eval_train(self.path)), rtol=1e-6, atol=1e-6)

    def test_no_retrace_across_goal_sets(self):
        f = self.forms['seq']
        fn = jax.jit(lambda p, c: f.eval(p, cent_override=c))
        a = jax.block_until_ready(fn(self.path, self.OV))
        b = jax.block_until_ready(fn(self.path, self.NEW))
        self.assertEqual(fn._cache_size(), 1)
        self.assertFalse(np.allclose(np.asarray(a), np.asarray(b)))

    def test_stlpy_override_matches_rebuilt_and_keeps_cache(self):
        f = self.forms['seq']
        baked = str(f.get_stlpy_form())
        over = str(f.get_stlpy_form(cent_override=np.asarray(self.NEW)))
        self.assertEqual(over, str(build_forms(np.asarray(self.NEW))['seq'].get_stlpy_form()))
        self.assertNotEqual(over, baked)
        self.assertEqual(str(f.get_stlpy_form()), baked)  # cache not polluted by the override call

    def test_stlpy_negated_reach_is_positive_normal_form(self):
        # stlpy STLTree.negation() raises NotImplementedError; ~reach must be pushed into the leaf.
        g0 = reach(CENTS[0], 0)
        y_in = np.array([[1.0], [1.0]])
        y_out = np.array([[0.0], [0.0]])
        pos, neg = g0.get_stlpy_form(), (~g0).get_stlpy_form()
        self.assertGreater(pos.robustness(y_in, 0), 0)
        self.assertLess(neg.robustness(y_in, 0), 0)
        self.assertGreater(neg.robustness(y_out, 0), 0)


if __name__ == "__main__":
    unittest.main()
