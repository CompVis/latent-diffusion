import contextlib
import unittest

import torch

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


@contextlib.contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class DiagonalGaussianSamplingTests(unittest.TestCase):
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    @staticmethod
    def parameters(dtype, device="cpu"):
        return torch.linspace(-0.25, 0.25, 72, dtype=dtype, device=device).reshape(
            2, 6, 2, 3
        )

    def require_cpu_kernels(self, dtype, convolution=False, parameters=None):
        names = {
            torch.float16: "Half",
            torch.bfloat16: "BFloat16",
            torch.bool: "Bool",
            torch.uint8: "Byte",
            torch.int8: "Char",
            torch.int16: "Short",
            torch.int32: "Int",
            torch.int64: "Long",
        }
        if dtype not in names:
            return
        try:
            values = self.parameters(dtype) if parameters is None else parameters
            _, logvar = values.chunk(2, dim=1)
            bounded = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * bounded)
            torch.exp(bounded)
            _ = std + std * std
            if convolution:
                projection = torch.nn.Conv2d(3, 4, 1).to(dtype=dtype)
                projection(values[:, :3])
        except RuntimeError as error:
            unsupported_kernel = "not implemented for '{}'".format(names[dtype]) in str(
                error
            )
            unsupported_uint8_bound = dtype == torch.uint8 and (
                str(error)
                == "value cannot be converted to type uint8_t without overflow: -30"
            )
            if not (unsupported_kernel or unsupported_uint8_bound):
                raise
            self.skipTest("{} CPU primitive is unavailable: {}".format(dtype, error))

    def test_sample_preserves_parameter_dtype_and_device(self):
        for dtype in self.dtypes:
            for factory_dtype in (torch.float32, torch.float64):
                for deterministic in (False, True):
                    with self.subTest(
                        dtype=dtype,
                        factory_dtype=factory_dtype,
                        deterministic=deterministic,
                    ):
                        with default_dtype(factory_dtype):
                            self.require_cpu_kernels(dtype)
                            posterior = DiagonalGaussianDistribution(
                                self.parameters(dtype), deterministic=deterministic
                            )
                            sample = posterior.sample()
                        self.assertEqual(sample.dtype, posterior.mean.dtype)
                        self.assertEqual(sample.device, posterior.mean.device)
                        self.assertEqual(sample.shape, posterior.mean.shape)
                        if deterministic:
                            self.assertTrue(torch.equal(sample, posterior.mode()))

    def test_sample_is_accepted_by_autoencoder_decoder_projection(self):
        # AutoencoderKL.decode begins with this 1x1 post_quant_conv.
        for dtype in self.dtypes:
            for factory_dtype in (torch.float32, torch.float64):
                with self.subTest(dtype=dtype, factory_dtype=factory_dtype):
                    with default_dtype(factory_dtype):
                        self.require_cpu_kernels(dtype, convolution=True)
                        posterior = DiagonalGaussianDistribution(self.parameters(dtype))
                        projection = torch.nn.Conv2d(3, 4, 1).to(dtype=dtype)
                        mode_result = projection(posterior.mode())
                        sample_result = projection(posterior.sample())
                    self.assertEqual(sample_result.dtype, mode_result.dtype)
                    self.assertEqual(sample_result.shape, mode_result.shape)
                    self.assertTrue(torch.isfinite(sample_result.double()).all())

    def test_standard_precision_values_and_rng_are_unchanged(self):
        for dtype, factory_dtype in (
            (torch.float32, torch.float32),
            (torch.float64, torch.float32),
            (torch.float64, torch.float64),
        ):
            for seed in (0, 17, 209):
                with self.subTest(dtype=dtype, factory_dtype=factory_dtype, seed=seed):
                    with default_dtype(factory_dtype):
                        posterior = DiagonalGaussianDistribution(self.parameters(dtype))
                        torch.manual_seed(seed)
                        noise = torch.randn(posterior.mean.shape)
                        expected = posterior.mean + posterior.std * noise
                        expected_rng = torch.get_rng_state().clone()
                        torch.manual_seed(seed)
                        actual = posterior.sample()
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_nonfloating_parameters_keep_gaussian_values_and_rng(self):
        for dtype in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            for construction_dtype in (torch.float32, torch.float64):
                for factory_dtype in (torch.float32, torch.float64):
                    for deterministic in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            construction_dtype=construction_dtype,
                            factory_dtype=factory_dtype,
                            deterministic=deterministic,
                        ):
                            with default_dtype(construction_dtype):
                                parameters = torch.zeros(2048, 2, 1, 1, dtype=dtype)
                                parameters[:, 0] = 1
                                self.require_cpu_kernels(dtype, parameters=parameters)
                                reference_mean, reference_logvar = parameters.chunk(
                                    2, dim=1
                                )
                                reference_logvar = torch.clamp(
                                    reference_logvar, -30.0, 20.0
                                )
                                reference_std = (
                                    torch.zeros_like(reference_mean)
                                    if deterministic
                                    else torch.exp(0.5 * reference_logvar)
                                )
                                posterior = DiagonalGaussianDistribution(
                                    parameters, deterministic=deterministic
                                )
                            with default_dtype(factory_dtype):
                                torch.manual_seed(13)
                                noise = torch.randn(posterior.mean.shape)
                                expected_rng = torch.get_rng_state().clone()
                                expected = reference_mean + reference_std * noise
                                torch.manual_seed(13)
                                actual = posterior.sample()
                            self.assertEqual(actual.dtype, expected.dtype)
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertTrue(
                                torch.equal(torch.get_rng_state(), expected_rng)
                            )

    def test_sampling_keeps_reparameterization_gradients(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                parameters = self.parameters(dtype).requires_grad_()
                posterior = DiagonalGaussianDistribution(parameters)
                torch.manual_seed(17)
                sample = posterior.sample()
                sample.sum().backward()
                mean_gradient, logvar_gradient = parameters.grad.chunk(2, dim=1)
                self.assertTrue(
                    torch.equal(mean_gradient, torch.ones_like(mean_gradient))
                )
                expected = 0.5 * (sample.detach() - posterior.mean.detach())
                self.assertTrue(torch.allclose(logvar_gradient, expected))

    def test_strided_parameters_remain_supported(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self.require_cpu_kernels(dtype)
                parameters = self.parameters(dtype).transpose(2, 3)
                self.assertFalse(parameters.is_contiguous())
                posterior = DiagonalGaussianDistribution(parameters)
                torch.manual_seed(71)
                first = posterior.sample()
                torch.manual_seed(71)
                second = posterior.sample()
                self.assertEqual(first.shape, posterior.mean.shape)
                self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.equal(posterior.mode(), parameters[:, :3]))

    def test_samples_retain_gaussian_moments(self):
        with default_dtype(torch.float32):
            parameters = torch.empty(16384, 2, 1, 1)
            parameters[:, 0] = 0.25
            parameters[:, 1] = -1.3862943611198906
            posterior = DiagonalGaussianDistribution(parameters)
            torch.manual_seed(43)
            samples = posterior.sample()
        self.assertAlmostEqual(samples.mean().item(), 0.25, delta=0.02)
        self.assertAlmostEqual(samples.std().item(), 0.5, delta=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_sampling_uses_posterior_device_and_dtype(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                posterior = DiagonalGaussianDistribution(
                    self.parameters(dtype, device="cuda")
                )
                sample = posterior.sample()
                self.assertEqual(sample.dtype, dtype)
                self.assertEqual(sample.device, posterior.mean.device)


if __name__ == "__main__":
    unittest.main()
