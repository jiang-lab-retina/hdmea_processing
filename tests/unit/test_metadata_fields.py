"""
Unit tests for acquisition_rate and frame_time metadata fields.

Tests the priority chain extraction and frame_time computation.
"""

import pytest
import numpy as np

from hdmea.pipeline.runner import (
    validate_acquisition_rate,
    compute_frame_time,
    DEFAULT_ACQUISITION_RATE,
    MIN_TYPICAL_ACQUISITION_RATE,
    MAX_TYPICAL_ACQUISITION_RATE,
)


class TestValidateAcquisitionRate:
    """Tests for validate_acquisition_rate() function."""

    def test_valid_rate_returns_true(self):
        """Positive acquisition rate should return True."""
        assert validate_acquisition_rate(20000.0) is True
        assert validate_acquisition_rate(10000.0) is True
        assert validate_acquisition_rate(1.0) is True

    def test_none_returns_false(self):
        """None should return False."""
        assert validate_acquisition_rate(None) is False

    def test_zero_returns_false(self):
        """Zero should return False."""
        assert validate_acquisition_rate(0.0) is False

    def test_negative_returns_false(self):
        """Negative values should return False."""
        assert validate_acquisition_rate(-1.0) is False
        assert validate_acquisition_rate(-20000.0) is False

    def test_typical_range_no_warning(self, caplog):
        """Values in typical range should not log warnings."""
        caplog.clear()
        result = validate_acquisition_rate(20000.0)
        assert result is True
        assert "Unusual acquisition_rate" not in caplog.text

    def test_below_typical_range_logs_warning(self, caplog):
        """Values below 1000 Hz should log a warning but still return True."""
        import logging
        caplog.set_level(logging.WARNING)
        result = validate_acquisition_rate(500.0)
        assert result is True
        assert "Unusual acquisition_rate" in caplog.text

    def test_above_typical_range_logs_warning(self, caplog):
        """Values above 100000 Hz should log a warning but still return True."""
        import logging
        caplog.set_level(logging.WARNING)
        result = validate_acquisition_rate(200000.0)
        assert result is True
        assert "Unusual acquisition_rate" in caplog.text


class TestComputeFrameTime:
    """Tests for compute_frame_time() function."""

    def test_standard_20khz(self):
        """20 kHz should give 0.00005 seconds."""
        frame_time = compute_frame_time(20000.0)
        assert frame_time == pytest.approx(0.00005, rel=1e-9)

    def test_10khz(self):
        """10 kHz should give 0.0001 seconds."""
        frame_time = compute_frame_time(10000.0)
        assert frame_time == pytest.approx(0.0001, rel=1e-9)

    def test_1hz(self):
        """1 Hz should give 1.0 seconds."""
        frame_time = compute_frame_time(1.0)
        assert frame_time == pytest.approx(1.0, rel=1e-9)

    def test_100khz(self):
        """100 kHz should give 0.00001 seconds."""
        frame_time = compute_frame_time(100000.0)
        assert frame_time == pytest.approx(0.00001, rel=1e-9)

    def test_inverse_relationship(self):
        """frame_time * acquisition_rate should equal 1.0."""
        for rate in [1000.0, 10000.0, 20000.0, 50000.0, 100000.0]:
            frame_time = compute_frame_time(rate)
            assert frame_time * rate == pytest.approx(1.0, rel=1e-9)

    def test_zero_raises_value_error(self):
        """Zero acquisition_rate should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            compute_frame_time(0.0)

    def test_negative_raises_value_error(self):
        """Negative acquisition_rate should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            compute_frame_time(-20000.0)


class TestDefaultConstants:
    """Tests for module-level constants."""

    def test_default_acquisition_rate(self):
        """Default acquisition rate should be 20000 Hz."""
        assert DEFAULT_ACQUISITION_RATE == 20000.0

    def test_typical_range_bounds(self):
        """Typical range should be 1000-100000 Hz."""
        assert MIN_TYPICAL_ACQUISITION_RATE == 1000.0
        assert MAX_TYPICAL_ACQUISITION_RATE == 100000.0


class TestAcquisitionRatePriorityChain:
    """Tests for the priority chain: CMCR -> CMTR -> default."""

    def test_cmcr_takes_priority(self):
        """When CMCR provides rate, it should be used."""
        cmcr_rate = 25000.0
        cmtr_rate = 20000.0
        
        # Simulate priority chain logic
        rate = cmcr_rate if validate_acquisition_rate(cmcr_rate) else None
        if rate is None:
            rate = cmtr_rate if validate_acquisition_rate(cmtr_rate) else None
        if rate is None:
            rate = DEFAULT_ACQUISITION_RATE
        
        assert rate == cmcr_rate

    def test_cmtr_fallback_when_cmcr_missing(self):
        """When CMCR rate is None, CMTR should be used."""
        cmcr_rate = None
        cmtr_rate = 18000.0
        
        rate = cmcr_rate if validate_acquisition_rate(cmcr_rate) else None
        if rate is None:
            rate = cmtr_rate if validate_acquisition_rate(cmtr_rate) else None
        if rate is None:
            rate = DEFAULT_ACQUISITION_RATE
        
        assert rate == cmtr_rate

    def test_cmtr_fallback_when_cmcr_invalid(self):
        """When CMCR rate is invalid, CMTR should be used."""
        cmcr_rate = -1.0  # Invalid
        cmtr_rate = 18000.0
        
        rate = cmcr_rate if validate_acquisition_rate(cmcr_rate) else None
        if rate is None:
            rate = cmtr_rate if validate_acquisition_rate(cmtr_rate) else None
        if rate is None:
            rate = DEFAULT_ACQUISITION_RATE
        
        assert rate == cmtr_rate

    def test_default_when_both_missing(self):
        """When both CMCR and CMTR rates are None, default should be used."""
        cmcr_rate = None
        cmtr_rate = None
        
        rate = cmcr_rate if validate_acquisition_rate(cmcr_rate) else None
        if rate is None:
            rate = cmtr_rate if validate_acquisition_rate(cmtr_rate) else None
        if rate is None:
            rate = DEFAULT_ACQUISITION_RATE
        
        assert rate == DEFAULT_ACQUISITION_RATE

    def test_default_when_both_invalid(self):
        """When both CMCR and CMTR rates are invalid, default should be used."""
        cmcr_rate = 0.0  # Invalid
        cmtr_rate = -100.0  # Invalid
        
        rate = cmcr_rate if validate_acquisition_rate(cmcr_rate) else None
        if rate is None:
            rate = cmtr_rate if validate_acquisition_rate(cmtr_rate) else None
        if rate is None:
            rate = DEFAULT_ACQUISITION_RATE
        
        assert rate == DEFAULT_ACQUISITION_RATE

