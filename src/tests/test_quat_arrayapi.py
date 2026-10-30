"""Test the Array-API compatible quaternion functions.

The functions in `adsorption.common._quat` are namespace agnostic: they
dispatch on the input arrays via the Python Array API standard, so they are
tested here with plain NumPy arrays.
"""

import numpy as np
import pytest

from adsorption.common import _quat


class TestQuaternionApply:
    """Test quaternion_apply function."""

    def test_single_point(self):
        """Test applying quaternion rotation to a single 3D point."""
        # A simple rotation quaternion (90 degrees around z-axis)
        angle = np.pi / 2
        quaternion = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        point = np.array([1.0, 0.0, 0.0])

        result = _quat.quaternion_apply(quaternion, point)

        # Expected: point rotated 90 deg around z-axis is (0, 1, 0)
        assert np.allclose(result, np.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_batch_points(self):
        """Test applying quaternion rotation to a batch of 3D points."""
        # A rotation quaternion (45 degrees around z-axis)
        angle = np.pi / 4
        quaternion = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = _quat.quaternion_apply(quaternion, points)

        assert result.shape == points.shape
        # z component is invariant under a rotation about the z-axis
        assert np.allclose(result[:, 2], points[:, 2], atol=1e-6)
        # rotating (1, 0, 0) by 45 deg gives (sqrt(2)/2, sqrt(2)/2, 0)
        assert np.allclose(
            result[0], np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0]), atol=1e-6
        )

    def test_batch_quaternions_batch_points(self):
        """Test applying multiple quaternion rotations to multiple points."""
        angles = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
        quaternions = np.stack(
            [np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]) for a in angles]
        )
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = _quat.quaternion_apply(quaternions, points)

        assert result.shape == points.shape
        assert np.allclose(result[:, 2], points[:, 2], atol=1e-6)
        # each rotated point keeps its norm
        assert np.allclose(
            np.linalg.norm(result, axis=-1),
            np.linalg.norm(points, axis=-1),
            atol=1e-6,
        )

    def test_invalid_point_shape(self):
        """Test that points which are not 3D raise a ValueError."""
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            _quat.quaternion_apply(quaternion, np.zeros((2, 2)))


class TestStandardizeQuaternion:
    """Test standardize_quaternion function."""

    def test_positive_real_part(self):
        """Test quaternion with already positive real part."""
        quaternion = np.array([0.5, 0.5, 0.5, 0.5])

        result = _quat.standardize_quaternion(quaternion)

        # Should remain unchanged
        assert np.allclose(result, quaternion)

    def test_negative_real_part(self):
        """Test quaternion with negative real part."""
        quaternion = np.array([-0.5, 0.5, 0.5, 0.5])

        result = _quat.standardize_quaternion(quaternion)

        # Should be negated
        assert np.allclose(result, -quaternion)

    def test_batch_quaternions(self):
        """Test standardizing a batch of quaternions."""
        quaternions = np.array(
            [
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5, 0.5],
                [0.707, 0.0, 0.707, 0.0],
            ]
        )

        result = _quat.standardize_quaternion(quaternions)

        assert result.shape == quaternions.shape
        assert np.all(result[:, 0] >= 0)
        assert np.allclose(np.abs(result), np.abs(quaternions))


class TestQuaternionInvert:
    """Test quaternion_invert function."""

    def test_unit_quaternion(self):
        """Test inverting a unit quaternion."""
        quaternion = np.array([0.707, 0.0, 0.707, 0.0])

        result = _quat.quaternion_invert(quaternion)

        # Inverse should be [w, -x, -y, -z]
        expected = np.array([0.707, 0.0, -0.707, 0.0])
        assert np.allclose(result, expected, atol=1e-6)

    def test_batch_quaternions(self):
        """Test inverting a batch of quaternions."""
        quaternions = np.array(
            [
                [0.707, 0.0, 0.707, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.866, 0.5, 0.0, 0.0],
            ]
        )

        result = _quat.quaternion_invert(quaternions)

        assert np.allclose(result[:, 0], quaternions[:, 0], atol=1e-6)
        assert np.allclose(result[:, 1:], -quaternions[:, 1:], atol=1e-6)
        # inverting twice restores the original quaternions
        assert np.allclose(
            _quat.quaternion_invert(result), quaternions, atol=1e-6
        )


class TestQuaternionRawMultiply:
    """Test quaternion_raw_multiply function."""

    def test_identity_multiplication(self):
        """Test multiplying with identity quaternion."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        quaternion = np.array([0.707, 0.0, 0.707, 0.0])

        result = _quat.quaternion_raw_multiply(identity, quaternion)

        # Multiplying by identity should return the original quaternion
        assert np.allclose(result, quaternion, atol=1e-6)

    def test_two_rotations(self):
        """Test multiplying two rotation quaternions."""
        # 90 degree rotation around z-axis
        angle1 = np.pi / 2
        q1 = np.array([np.cos(angle1 / 2), 0.0, 0.0, np.sin(angle1 / 2)])

        # 90 degree rotation around y-axis
        angle2 = np.pi / 2
        q2 = np.array([np.cos(angle2 / 2), 0.0, np.sin(angle2 / 2), 0.0])

        result = _quat.quaternion_raw_multiply(q1, q2)

        # (w, x, y, z) components of the composition: z first, then y
        expected = np.array([0.5, -0.5, 0.5, 0.5])
        assert np.allclose(result, expected, atol=1e-6)

    def test_batch_multiplication(self):
        """Test multiplying batches of quaternions."""
        q_batch1 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.707, 0.0, 0.707, 0.0],
            ]
        )
        q_batch2 = np.array(
            [
                [0.707, 0.0, 0.707, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        result = _quat.quaternion_raw_multiply(q_batch1, q_batch2)

        assert result.shape == q_batch1.shape
        assert np.allclose(result[0], q_batch2[0], atol=1e-6)
        assert np.allclose(result[1], q_batch1[1], atol=1e-6)


class TestRandomQuaternions:
    """Test random_quaternions function."""

    def test_shape(self):
        """Test that output has correct shape."""
        n = 10
        result = _quat.random_quaternions(n)
        result_arrayapi = _quat.random_quaternions(n, xp=np)

        assert result.shape == (n, 4)
        assert result_arrayapi.shape == (n, 4)

    def test_unit_quaternions(self):
        """Test that generated quaternions are unit quaternions."""
        n = 100
        result = _quat.random_quaternions(n, xp=np)

        # Check that all quaternions have unit norm
        norms = np.sqrt(np.sum(result * result, axis=1))
        assert np.allclose(norms, np.ones(n), atol=1e-6)

    def test_nonnegative_real_part(self):
        """Test that all quaternions have nonnegative real part."""
        n = 100
        result = _quat.random_quaternions(n, xp=np)

        # Check that all real parts are nonnegative
        assert np.all(result[:, 0] >= 0)

    def test_dtype(self):
        """Test that the dtype is respected."""
        n = 10
        result = _quat.random_quaternions(n, xp=np)

        assert result.dtype == np.float64


class TestCopysign:
    """Test _copysign function."""

    def test_same_sign(self):
        """Test copysign when signs are the same."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0, 1.0])

        result = _quat._copysign(a, b)

        # When signs are same, result should be same as a
        assert np.allclose(result, a)

    def test_different_signs(self):
        """Test copysign when signs differ."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -1.0, -1.0])

        result = _quat._copysign(a, b)

        # When signs differ, result should be -a
        assert np.allclose(result, -a)

    def test_batch_copysign(self):
        """Test copysign with batch inputs."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[-1.0, 1.0], [1.0, -1.0]])

        result = _quat._copysign(a, b)

        assert np.allclose(result, np.array([[-1.0, 2.0], [3.0, -4.0]]))
