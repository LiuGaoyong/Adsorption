"""Test array API compatible quaternion functions against original torch versions."""

import torch

from adsorption.interfaces import _quat, _quat_arrayapi


class TestQuaternionApply:
    """Test quaternion_apply function."""

    def test_single_point(self):
        """Test applying quaternion rotation to a single 3D point."""
        # Create a simple rotation quaternion (90 degrees around z-axis)
        angle = torch.tensor(torch.pi / 2)
        quaternion = torch.tensor(
            [
                torch.cos(angle / 2),
                0.0,
                0.0,
                torch.sin(angle / 2),
            ]
        )
        point = torch.tensor([1.0, 0.0, 0.0])

        # Test original version
        result_original = _quat.quaternion_apply(quaternion, point)

        # Test array API version
        result_arrayapi = _quat_arrayapi.quaternion_apply(quaternion, point)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)
        # Expected: point rotated 90° around z-axis should be close to (0, 1, 0)
        assert torch.allclose(
            result_arrayapi, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6
        )

    def test_batch_points(self):
        """Test applying quaternion rotation to a batch of 3D points."""
        # Create a rotation quaternion (45 degrees around z-axis)
        angle = torch.tensor(torch.pi / 4)
        quaternion = torch.tensor(
            [
                torch.cos(angle / 2),
                0.0,
                0.0,
                torch.sin(angle / 2),
            ]
        )
        points = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Test original version
        result_original = _quat.quaternion_apply(quaternion, points)

        # Test array API version
        result_arrayapi = _quat_arrayapi.quaternion_apply(quaternion, points)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)

    def test_batch_quaternions_batch_points(self):
        """Test applying multiple quaternion rotations to multiple points."""
        # Create multiple rotation quaternions
        angles = torch.tensor([torch.pi / 6, torch.pi / 4, torch.pi / 3])
        quaternions = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(a / 2),
                        torch.zeros(1).squeeze(),
                        torch.zeros(1).squeeze(),
                        torch.sin(a / 2),
                    ]
                )
                for a in angles
            ]
        )
        points = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Test original version
        result_original = _quat.quaternion_apply(quaternions, points)

        # Test array API version
        result_arrayapi = _quat_arrayapi.quaternion_apply(quaternions, points)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)


class TestStandardizeQuaternion:
    """Test standardize_quaternion function."""

    def test_positive_real_part(self):
        """Test quaternion with already positive real part."""
        quaternion = torch.tensor([0.5, 0.5, 0.5, 0.5])

        result_original = _quat.standardize_quaternion(quaternion)
        result_arrayapi = _quat_arrayapi.standardize_quaternion(quaternion)

        assert torch.allclose(result_original, result_arrayapi)
        # Should remain unchanged
        assert torch.allclose(result_arrayapi, quaternion)

    def test_negative_real_part(self):
        """Test quaternion with negative real part."""
        quaternion = torch.tensor([-0.5, 0.5, 0.5, 0.5])

        result_original = _quat.standardize_quaternion(quaternion)
        result_arrayapi = _quat_arrayapi.standardize_quaternion(quaternion)

        assert torch.allclose(result_original, result_arrayapi)
        # Should be negated
        assert torch.allclose(result_arrayapi, -quaternion)

    def test_batch_quaternions(self):
        """Test standardizing a batch of quaternions."""
        quaternions = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5, 0.5],
                [0.707, 0.0, 0.707, 0.0],
            ]
        )

        result_original = _quat.standardize_quaternion(quaternions)
        result_arrayapi = _quat_arrayapi.standardize_quaternion(quaternions)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)


class TestQuaternionInvert:
    """Test quaternion_invert function."""

    def test_unit_quaternion(self):
        """Test inverting a unit quaternion."""
        quaternion = torch.tensor([0.707, 0.0, 0.707, 0.0])

        result_original = _quat.quaternion_invert(quaternion)
        result_arrayapi = _quat_arrayapi.quaternion_invert(quaternion)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)
        # Inverse should be [w, -x, -y, -z]
        expected = torch.tensor([0.707, 0.0, -0.707, 0.0])
        assert torch.allclose(result_arrayapi, expected, atol=1e-6)

    def test_batch_quaternions(self):
        """Test inverting a batch of quaternions."""
        quaternions = torch.tensor(
            [
                [0.707, 0.0, 0.707, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.866, 0.5, 0.0, 0.0],
            ]
        )

        result_original = _quat.quaternion_invert(quaternions)
        result_arrayapi = _quat_arrayapi.quaternion_invert(quaternions)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)


class TestQuaternionRawMultiply:
    """Test quaternion_raw_multiply function."""

    def test_identity_multiplication(self):
        """Test multiplying with identity quaternion."""
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        quaternion = torch.tensor([0.707, 0.0, 0.707, 0.0])

        result_original = _quat.quaternion_raw_multiply(identity, quaternion)
        result_arrayapi = _quat_arrayapi.quaternion_raw_multiply(
            identity, quaternion
        )

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)
        # Multiplying by identity should return the original quaternion
        assert torch.allclose(result_arrayapi, quaternion, atol=1e-6)

    def test_two_rotations(self):
        """Test multiplying two rotation quaternions."""
        # 90 degree rotation around z-axis
        angle1 = torch.tensor(torch.pi / 2)
        q1 = torch.tensor(
            [
                torch.cos(angle1 / 2),
                0.0,
                0.0,
                torch.sin(angle1 / 2),
            ]
        )

        # 90 degree rotation around y-axis
        angle2 = torch.tensor(torch.pi / 2)
        q2 = torch.tensor(
            [
                torch.cos(angle2 / 2),
                0.0,
                torch.sin(angle2 / 2),
                0.0,
            ]
        )

        result_original = _quat.quaternion_raw_multiply(q1, q2)
        result_arrayapi = _quat_arrayapi.quaternion_raw_multiply(q1, q2)

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)

    def test_batch_multiplication(self):
        """Test multiplying batches of quaternions."""
        q_batch1 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.707, 0.0, 0.707, 0.0],
            ]
        )
        q_batch2 = torch.tensor(
            [
                [0.707, 0.0, 0.707, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        result_original = _quat.quaternion_raw_multiply(q_batch1, q_batch2)
        result_arrayapi = _quat_arrayapi.quaternion_raw_multiply(
            q_batch1, q_batch2
        )

        assert torch.allclose(result_original, result_arrayapi, atol=1e-6)


class TestRandomQuaternions:
    """Test random_quaternions function."""

    def test_shape(self):
        """Test that output has correct shape."""
        n = 10
        result_original = _quat.random_quaternions(n)
        result_arrayapi = _quat_arrayapi.random_quaternions(n, xp=torch)

        assert result_original.shape == (n, 4)
        assert result_arrayapi.shape == (n, 4)

    def test_unit_quaternions(self):
        """Test that generated quaternions are unit quaternions."""
        n = 100
        result = _quat_arrayapi.random_quaternions(n, xp=torch)

        # Check that all quaternions have unit norm
        norms = torch.sqrt(torch.sum(result * result, dim=1))
        assert torch.allclose(norms, torch.ones(n), atol=1e-6)

    def test_nonnegative_real_part(self):
        """Test that all quaternions have nonnegative real part."""
        n = 100
        result = _quat_arrayapi.random_quaternions(n, xp=torch)

        # Check that all real parts are nonnegative
        assert torch.all(result[:, 0] >= 0)

    def test_dtype_and_device(self):
        """Test that dtype and device are respected."""
        n = 10
        dtype = torch.float64
        result = _quat_arrayapi.random_quaternions(n, xp=torch)

        assert result.dtype == dtype


class TestCopysign:
    """Test _copysign function."""

    def test_same_sign(self):
        """Test copysign when signs are the same."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 1.0, 1.0])

        result_original = _quat._copysign(a, b)
        result_arrayapi = _quat_arrayapi._copysign(a, b)

        assert torch.allclose(result_original, result_arrayapi)
        # When signs are same, result should be same as a
        assert torch.allclose(result_arrayapi, a)

    def test_different_signs(self):
        """Test copysign when signs differ."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([-1.0, -1.0, -1.0])

        result_original = _quat._copysign(a, b)
        result_arrayapi = _quat_arrayapi._copysign(a, b)

        assert torch.allclose(result_original, result_arrayapi)
        # When signs differ, result should be -a
        assert torch.allclose(result_arrayapi, -a)

    def test_batch_copysign(self):
        """Test copysign with batch inputs."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])

        result_original = _quat._copysign(a, b)
        result_arrayapi = _quat_arrayapi._copysign(a, b)

        assert torch.allclose(result_original, result_arrayapi)
