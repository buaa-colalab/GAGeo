import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path

import torch


class RoPE2DTest(unittest.TestCase):
    def test_curope_extension_is_not_part_of_repository(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "curope").exists())

    def test_rope2d_imports_without_cuda_extension_warning_and_backpropagates(self):
        sys.modules.pop("models.layers.pos_embed", None)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            pos_embed = importlib.import_module("models.layers.pos_embed")

        self.assertNotIn("cuda-compiled", stdout.getvalue())

        rope = pos_embed.RoPE2D(freq=100.0)
        tokens = torch.randn(2, 3, 4, 8, requires_grad=True)
        positions = torch.tensor(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],
                [[1, 1], [1, 2], [2, 1], [2, 2]],
            ],
            dtype=torch.long,
        )

        output = rope(tokens, positions)
        self.assertEqual(output.shape, tokens.shape)

        output.sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad.shape, tokens.shape)


if __name__ == "__main__":
    unittest.main()
