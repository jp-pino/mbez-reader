"""Tests for mbez_reader module."""

import numpy as np
import pytest
from pathlib import Path
from mbez_reader import create_fan_image, extract_pings


def test_create_fan_image_basic():
    """Test basic fan image creation with simple data."""
    # Create simple test data: 3 ranges x 3 beams
    raw_data = np.array(
        [
            [100, 150, 200],
            [110, 160, 210],
            [120, 170, 220],
        ],
        dtype=np.uint8,
    )

    # Bearings: -10°, 0°, 10° (in radians)
    bearings = np.array([-0.1745, 0.0, 0.1745])

    fan_image = create_fan_image(raw_data, bearings)

    # Check output shape
    assert fan_image.ndim == 2
    assert fan_image.shape[0] == raw_data.shape[0]  # Same height as ranges
    assert fan_image.shape[1] > raw_data.shape[1]  # Wider due to fan spread

    # Check dtype
    assert fan_image.dtype == np.uint8

    # Check that some pixels are non-zero (data was mapped)
    assert np.any(fan_image > 0)


def test_create_fan_image_output_dimensions():
    """Test that output dimensions are calculated correctly."""
    raw_data = np.ones((100, 50), dtype=np.uint8) * 128
    bearings = np.linspace(-1.0, 1.0, 50)  # ~114° FOV

    fan_image = create_fan_image(raw_data, bearings)

    # Height should match number of ranges
    assert fan_image.shape[0] == 100

    # Width should be approximately 2 * ranges * sin(max_bearing)
    expected_width = int(2 * 100 * np.sin(1.0))
    assert abs(fan_image.shape[1] - expected_width) < 10  # Allow small variance


def test_create_fan_image_preserves_values():
    """Test that the transformation preserves data values."""
    # Uniform data
    raw_data = np.ones((50, 30), dtype=np.uint8) * 200
    bearings = np.linspace(-0.5, 0.5, 30)

    fan_image = create_fan_image(raw_data, bearings)

    # All non-zero pixels should have value 200
    non_zero = fan_image[fan_image > 0]
    assert np.all(non_zero == 200)


def test_create_fan_image_zero_bearing():
    """Test with zero bearing (should create narrow vertical image)."""
    raw_data = np.ones((10, 1), dtype=np.uint8) * 100
    bearings = np.array([0.0])

    fan_image = create_fan_image(raw_data, bearings)

    # Should be very narrow
    assert fan_image.shape[1] <= 3
    assert np.any(fan_image == 100)


def test_create_fan_image_with_actual_data(tmp_path):
    """Test with realistic sonar data dimensions."""
    # Simulate typical sonar dimensions
    num_ranges = 649
    num_beams = 256

    raw_data = np.random.randint(0, 255, (num_ranges, num_beams), dtype=np.uint8)
    bearings = np.linspace(-1.134, 1.134, num_beams)  # ~130° FOV

    fan_image = create_fan_image(raw_data, bearings)

    # Verify output
    assert fan_image.shape[0] == num_ranges
    assert fan_image.dtype == np.uint8
    assert fan_image.shape[1] > num_beams  # Fan should be wider


@pytest.mark.skipif(
    not Path("../mbez/20241022_135915.oculus.mbez").exists(),
    reason="Example .mbez file not found",
)
def test_extract_pings_from_file():
    """Test extracting pings from actual .mbez file."""
    mbez_file = Path("../mbez/20241022_135915.oculus.mbez")

    # Extract first ping
    pings = list(extract_pings(mbez_file, start_index=0, count=1))

    assert len(pings) == 1
    assert hasattr(pings[0], "number_of_ranges")
    assert hasattr(pings[0], "number_of_beams")
    assert hasattr(pings[0], "ping_data")
    assert hasattr(pings[0], "bearings")


@pytest.mark.skipif(
    not Path("../mbez/20241022_135915.oculus.mbez").exists(),
    reason="Example .mbez file not found",
)
def test_extract_multiple_pings():
    """Test extracting multiple pings."""
    mbez_file = Path("../mbez/20241022_135915.oculus.mbez")

    # Extract 5 pings starting from index 10
    pings = list(extract_pings(mbez_file, start_index=10, count=5))

    assert len(pings) == 5

    # All should be valid pings
    for ping in pings:
        assert ping.number_of_ranges > 0
        assert ping.number_of_beams > 0


def test_extract_pings_insufficient_pings(tmp_path):
    """Test error handling when not enough pings available."""
    # This test would need a mock or small test file
    # Skipped for now as it requires test data
    pass
