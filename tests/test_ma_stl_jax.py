import importlib
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

os.environ["DIFF_STL_BACKEND"] = "jax"  # So ds_utils does not require torch

import ds.utils as ds_utils
from ds.stl_jax import STL, RectReachPredicate
from ds.ma_stl_jax import CaTLPlus, Task

TEST_TOLERANCE = 1e-3  # Small number close to 0


class TestMASTLJAXExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
        importlib.reload(ds_utils)  # Reload the module to reset the backend

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

        self.num_satisfied_agents = 2
        task = Task("task", self.form, self.num_satisfied_agents, None)
        task_seq = Task("task_seq", self.seq_form, self.num_satisfied_agents, None)
        task_cover = Task("task_cover", self.cover_form, self.num_satisfied_agents, None)
        task_loop = Task("task_loop", self.loop_form, self.num_satisfied_agents, None)
        self.catl_form = CaTLPlus(task)
        self.catl_form_or = CaTLPlus(task) | CaTLPlus(task_seq)
        self.catl_form_and = CaTLPlus(task_cover) & CaTLPlus(task_seq)

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

        not_satisfy_catl_path = jnp.concatenate([input_path, path_sat], axis=0)
        satisfy_catl_path = jnp.concatenate([input_path, path_sat, path_sat], axis=0)
        satisfy_catl_path_seq = jnp.concatenate([input_path, path_sat_seq, path_sat_seq], axis=0)
        satisfy_catl_path_seq_and = jnp.concatenate(
            [input_path, path_sat_cover, path_sat_cover, path_sat_seq, path_sat_seq],
            axis=0)
        notsatisfy_catl_path_seq_and = jnp.concatenate(
            [input_path, path_sat_cover, path_sat_cover, path_sat_cover, path_sat_seq],
            axis=0)
        not_satisfy_catl_path_seq = jnp.concatenate([input_path, path_sat_seq, input_path], axis=0)

        loss = self.catl_form.eval(not_satisfy_catl_path)

        # self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertLess(loss, 0, f"Not returning correct value")

        loss = self.catl_form.eval(satisfy_catl_path)
        # self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertGreater(loss, 0, f"Not returning correct value")
        # self.assertEqual(loss.shape[0], num_tiles, f"Not returning {num_tiles} values")

        # Now test the or and and operators
        loss = self.catl_form_or.eval(satisfy_catl_path)
        self.assertGreater(loss, 0, f"Not returning correct value for or sat path")
        loss = self.catl_form_or.eval(satisfy_catl_path_seq)
        self.assertGreater(loss, 0, f"Not returning correct value for or sat path")

        loss = self.catl_form_or.eval(not_satisfy_catl_path)
        self.assertLess(loss, 0, f"Not returning correct value for or unsat path")
        loss = self.catl_form_or.eval(not_satisfy_catl_path_seq)
        self.assertLess(loss, 0, f"Not returning correct value for or unsat path")

        loss = self.catl_form_and.eval(notsatisfy_catl_path_seq_and)
        self.assertLess(loss, 0, f"Not returning correct value for and unsat path")
        loss = self.catl_form_and.eval(satisfy_catl_path_seq_and)
        self.assertGreater(loss, 0, f"Not returning correct value for and sat path")

    def test_repr(self):
        print(self.form)
        print(self.catl_form)

    def _test_run(self):
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
        path, loss = stl_diff_examples.backward()
        print('Path', path)
        assert loss < TEST_TOLERANCE  # Loss should be less than 0 to satisfy the formula
        # (jax.lax.fori_loop(0, 1000, lambda i, _: jit(eval_reach_avoid)(), None)).block_until_ready()
        # for _ in range(1000):
        #     eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here

    def _test_avoid_backward(self):
        path, loss = stl_diff_examples.backward(avoid_spec=True)
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


if __name__ == '__main__':
    unittest.main()
