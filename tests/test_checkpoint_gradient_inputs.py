"""Checkpointing preserves ordinary gradients with optional or frozen inputs."""

import copy
import unittest

import torch

from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.openaimodel import ResBlock, UNetModel
from ldm.modules.diffusionmodules.util import checkpoint


class CheckpointGradientInputsTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    def run_module(self, module, inputs):
        module.zero_grad()
        copied = [
            (
                x.detach().clone().requires_grad_(x.requires_grad)
                if isinstance(x, torch.Tensor)
                else x
            )
            for x in inputs
        ]
        value = module(*copied)
        value.pow(2).mean().backward()
        grads = {
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in module.named_parameters()
        }
        input_grads = [
            (
                None
                if not isinstance(x, torch.Tensor) or x.grad is None
                else x.grad.detach().clone()
            )
            for x in copied
        ]
        return value.detach(), grads, input_grads

    def assert_module_parity(self, reference, actual, inputs, frozen_reference=False):
        expected, expected_grads, expected_inputs = self.run_module(reference, inputs)
        value, grads, input_grads = self.run_module(actual, inputs)
        self.assertTrue(torch.equal(value, expected))
        for name, param in actual.named_parameters():
            if (frozen_reference and not param.requires_grad) or expected_grads[
                name
            ] is None:
                self.assertIsNone(grads[name])
            else:
                self.assert_tensor_matches(grads[name], expected_grads[name])
        for got, want in zip(input_grads, expected_inputs):
            if want is None:
                self.assertIsNone(got)
            else:
                self.assert_tensor_matches(got, want)

    def freeze_parameters(self, module, mode):
        params = list(module.parameters())
        if mode == "one":
            params[0].requires_grad_(False)
        elif mode == "mixed":
            for param in params[::2]:
                param.requires_grad_(False)
        elif mode == "all":
            module.requires_grad_(False)

    def test_transformer_optional_context(self):
        for dtype in (torch.float32, torch.float64):
            for has_context in (False, True):
                for mode in ("none", "one", "all"):
                    for input_grad in (False, True):
                        if not has_context and mode == "all" and not input_grad:
                            continue
                        with self.subTest(
                            dtype=dtype,
                            context=has_context,
                            frozen=mode,
                            input_grad=input_grad,
                        ):
                            torch.manual_seed(79)
                            reference = BasicTransformerBlock(
                                4, 1, 4, checkpoint=False
                            ).to(dtype)
                            self.freeze_parameters(reference, mode)
                            actual = copy.deepcopy(reference)
                            actual.checkpoint = True
                            x = torch.randn(2, 3, 4, dtype=dtype).requires_grad_(
                                input_grad
                            )
                            context = (
                                torch.randn(2, 2, 4, dtype=dtype).requires_grad_()
                                if has_context
                                else None
                            )
                            self.assert_module_parity(reference, actual, (x, context))

    def test_transformer_frozen_conditioning(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("none", "one", "all"):
                for input_grad in (False, True):
                    if mode == "all" and not input_grad:
                        continue
                    with self.subTest(dtype=dtype, frozen=mode, input_grad=input_grad):
                        torch.manual_seed(80)
                        reference = BasicTransformerBlock(4, 1, 4, checkpoint=False).to(
                            dtype
                        )
                        self.freeze_parameters(reference, mode)
                        actual = copy.deepcopy(reference)
                        actual.checkpoint = True
                        x = torch.randn(1, 3, 4, dtype=dtype).requires_grad_(input_grad)
                        context = torch.randn(1, 2, 4, dtype=dtype)
                        self.assert_module_parity(reference, actual, (x, context))

    def test_residual_block_frozen_parameters_and_embeddings(self):
        for mode in ("none", "one", "all"):
            for input_grad in (False, True):
                for embedding_grad in (False, True):
                    if mode == "all" and not input_grad and not embedding_grad:
                        continue
                    with self.subTest(
                        frozen=mode,
                        input_grad=input_grad,
                        embedding_grad=embedding_grad,
                    ):
                        torch.manual_seed(81)
                        reference = ResBlock(32, 8, 0, use_checkpoint=False)
                        with torch.no_grad():
                            reference.out_layers[-1].weight.normal_(std=0.02)
                        self.freeze_parameters(reference, mode)
                        actual = copy.deepcopy(reference)
                        actual.use_checkpoint = True
                        x = torch.randn(1, 32, 3, 3).requires_grad_(input_grad)
                        embedding = torch.randn(1, 8).requires_grad_(embedding_grad)
                        self.assert_module_parity(reference, actual, (x, embedding))

    def test_unet_with_frozen_torso_parameters(self):
        for mode in ("none", "mixed", "all"):
            with self.subTest(frozen=mode):
                torch.manual_seed(82)
                reference = UNetModel(
                    image_size=4,
                    in_channels=2,
                    model_channels=32,
                    out_channels=2,
                    num_res_blocks=1,
                    attention_resolutions=(1,),
                    channel_mult=(1,),
                    num_heads=1,
                    use_checkpoint=True,
                )
                with torch.no_grad():
                    reference.out[-1].weight.normal_(std=0.02)
                actual = copy.deepcopy(reference)
                self.freeze_parameters(actual, mode)
                x = torch.randn(1, 2, 4, 4).requires_grad_()
                timesteps = torch.tensor([3])
                # Freezing parameters leaves forward values and all remaining
                # partial derivatives unchanged. The reference remains fully
                # trainable, including its checkpointed attention blocks.
                self.assert_module_parity(
                    reference, actual, (x, timesteps), frozen_reference=True
                )

    def test_constants_and_unused_parameters_against_analytic_derivatives(self):
        for dtype in (torch.float32, torch.float64):
            for input_grad in (False, True):
                with self.subTest(dtype=dtype, input_grad=input_grad):
                    x = torch.tensor(
                        [0.5, -0.75, 1.25, 0.25], dtype=dtype
                    ).requires_grad_(input_grad)
                    weight = torch.nn.Parameter(torch.tensor(0.75, dtype=dtype))
                    frozen = torch.nn.Parameter(
                        torch.tensor(-0.5, dtype=dtype), requires_grad=False
                    )
                    unused = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))
                    index = torch.tensor([2, 0, 2])

                    def function(
                        values, indices, optional, scale, weight=weight, frozen=frozen
                    ):
                        self.assertIsNone(optional)
                        return weight * values[indices].pow(3) + frozen + scale

                    value = checkpoint(
                        function,
                        (x, index, None, 0.125),
                        (weight, frozen, unused),
                        True,
                    )
                    expected = 0.75 * x.detach()[index].pow(3) - 0.375
                    self.assertTrue(torch.equal(value.detach(), expected))
                    value.sum().backward()
                    self.assert_tensor_matches(
                        weight.grad, x.detach()[index].pow(3).sum()
                    )
                    self.assertIsNone(frozen.grad)
                    self.assertIsNone(unused.grad)
                    if input_grad:
                        derivative = torch.zeros_like(x)
                        derivative.scatter_add_(
                            0, index, 2.25 * x.detach()[index].pow(2)
                        )
                        self.assert_tensor_matches(x.grad, derivative)
                    else:
                        self.assertIsNone(x.grad)

    def test_forward_only_and_no_grad_controls(self):
        for dtype in (torch.float32, torch.float64):
            for has_context in (False, True):
                with self.subTest(dtype=dtype, context=has_context):
                    torch.manual_seed(83)
                    reference = BasicTransformerBlock(4, 1, 4, checkpoint=False).to(
                        dtype
                    )
                    actual = copy.deepcopy(reference)
                    actual.checkpoint = True
                    x = torch.randn(1, 2, 4, dtype=dtype)
                    context = torch.randn(1, 3, 4, dtype=dtype) if has_context else None
                    with torch.no_grad():
                        expected = reference(x, context)
                        value = actual(x, context)
                    self.assertFalse(value.requires_grad)
                    self.assertTrue(torch.equal(value, expected))
                    reference.requires_grad_(False)
                    actual.requires_grad_(False)
                    value = actual(x, context)
                    self.assertFalse(value.requires_grad)
                    self.assertTrue(torch.equal(value, reference(x, context)))

    def test_constant_output_with_unused_trainable_parameter(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([0.5, -0.75, 1.25], dtype=dtype)
                unused = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))
                value = checkpoint(lambda values: values.pow(2), (x,), (unused,), True)
                self.assertTrue(torch.equal(value.detach(), x.pow(2)))
                value.sum().backward()
                self.assertIsNone(x.grad)
                self.assertIsNone(unused.grad)

    def test_mixed_constant_and_differentiable_outputs(self):
        for dtype in (torch.float32, torch.float64):
            for input_grad in (False, True):
                for selected in ("constant", "parameter", "both"):
                    with self.subTest(
                        dtype=dtype, input_grad=input_grad, selected=selected
                    ):
                        x = torch.tensor(
                            [0.5, -0.75, 1.25], dtype=dtype
                        ).requires_grad_(input_grad)
                        constant_input = torch.tensor([-0.5, 0.25, 1.5], dtype=dtype)
                        weight = torch.nn.Parameter(torch.tensor(0.75, dtype=dtype))
                        unused = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))

                        def function(values, fixed, weight=weight):
                            return fixed.pow(2), values * weight

                        constant, differentiable = checkpoint(
                            function, (x, constant_input), (weight, unused), True
                        )
                        self.assertTrue(
                            torch.equal(constant.detach(), constant_input.pow(2))
                        )
                        self.assertTrue(
                            torch.equal(differentiable.detach(), x * weight.detach())
                        )
                        if selected == "constant":
                            loss = constant.sum()
                        elif selected == "parameter":
                            loss = differentiable.sum()
                        else:
                            loss = constant.sum() + differentiable.sum()
                        loss.backward()
                        expected = (
                            x.new_zeros(())
                            if selected == "constant"
                            else x.detach().sum()
                        )
                        self.assert_tensor_matches(weight.grad, expected)
                        self.assertIsNone(constant_input.grad)
                        if input_grad:
                            expected_x = (
                                torch.zeros_like(x)
                                if selected == "constant"
                                else torch.full_like(x, 0.75)
                            )
                            self.assert_tensor_matches(x.grad, expected_x)
                        else:
                            self.assertIsNone(x.grad)
                        self.assertIsNone(unused.grad)

    def test_repeated_optimizer_steps(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(84)
                reference = BasicTransformerBlock(4, 1, 4, checkpoint=False).to(dtype)
                reference.attn1.to_q.weight.requires_grad_(False)
                actual = copy.deepcopy(reference)
                actual.checkpoint = True
                expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
                x = torch.randn(1, 3, 4, dtype=dtype)
                for _ in range(2):
                    self.assert_module_parity(reference, actual, (x, None))
                    expected_optimizer.step()
                    optimizer.step()
                    for got, want in zip(actual.parameters(), reference.parameters()):
                        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
