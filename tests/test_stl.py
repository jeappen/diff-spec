import importlib
import numpy as np
import os
import torch
import unittest

import ds.utils as ds_utils
import examples.stl.differentiability as stl_diff_examples
from ds.stl import STL, RectReachPredicate

TEST_TOLERANCE = 1e-3  # Small number close to 0

class TestExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = ""
        importlib.reload(stl_diff_examples)  # Reload the module to reset the backend
        importlib.reload(ds_utils)  # Reload the module to reset the backend

    def test_run(self):
        # Test Eval
        final_result = []
        for _ in range(1000):
            # Fair test with jax
            res = stl_diff_examples.eval_reach_avoid(mute=True)
            final_result.append(res)
            # Match expected output
            assert ((res[0] > 0) == torch.tensor([True, False, False])).all()
            assert ((res[1] > 0) == torch.tensor([True, False, True])).all()

        # print(final_result)

        # Test differentiability
        path, loss = stl_diff_examples.backward()
        print('Path', path)
        assert loss < TEST_TOLERANCE  # Loss should be less than 0 to satisfy the formula

    def test_avoid_backward(self):
        path, loss = stl_diff_examples.backward(avoid_spec=True)
        print('AvoidPath', loss, path)
        assert loss < TEST_TOLERANCE  # Loss should be less than 0 to satisfy the formula

    def test_evaluations(self, num_tiles=3):
        """Run simple evaluations to test shapes and types"""
        goal_1 = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "goal_1"))
        # goal_2 is a rectangle area centered in [2, 2] with width and height 1
        goal_2 = STL(RectReachPredicate(np.array([2, 2]), np.array([1, 1]), "goal_2"))

        # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
        # and that holds always in 0 to 8
        # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
        form = (goal_1.eventually(0, 5) & goal_2.eventually(0, 5)).always(0, 8)
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

        loss = form.eval(path.tile(num_tiles, 1, 1))  # Make a batch of size num_tiles
        self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertEqual(loss.shape[0], num_tiles, f"Not returning {num_tiles} values")


if __name__ == '__main__':
    unittest.main()
