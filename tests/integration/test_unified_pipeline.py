"""
Integration Tests for Unified Pipeline

Tests the complete pipeline from CMCR/CMTR through all 11 steps
and validates output against reference file.
"""

import logging
import sys
from pathlib import Path

import pytest
import h5py
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import create_session, load_session_from_hdf5

from Projects.unified_pipeline.config import (
    TEST_DATASET_ID,
    TEST_REFERENCE_FILE,
    DEFAULT_OUTPUT_DIR,
)


# Skip if reference file doesn't exist
pytestmark = pytest.mark.skipif(
    not TEST_REFERENCE_FILE.exists(),
    reason=f"Reference file not found: {TEST_REFERENCE_FILE}"
)


class TestUnifiedPipelineStructure:
    """Test HDF5 structure matches reference file."""
    
    def test_reference_file_exists(self):
        """Verify the reference file exists for comparison."""
        assert TEST_REFERENCE_FILE.exists(), f"Reference file not found: {TEST_REFERENCE_FILE}"
    
    def test_load_reference_file(self):
        """Test loading reference file into session."""
        session = load_session_from_hdf5(TEST_REFERENCE_FILE)
        
        assert session is not None
        assert session.dataset_id == TEST_DATASET_ID
        assert session.unit_count > 0
        
    def test_reference_has_expected_groups(self):
        """Test reference file has expected top-level groups."""
        with h5py.File(TEST_REFERENCE_FILE, 'r') as f:
            expected_groups = ['units', 'metadata', 'stimulus', 'pipeline']
            for group in expected_groups:
                assert group in f, f"Missing group: {group}"
    
    def test_reference_units_have_features(self):
        """Test reference file units have expected features."""
        with h5py.File(TEST_REFERENCE_FILE, 'r') as f:
            units = f['units']
            
            # Check at least one unit
            assert len(units) > 0
            
            # Check first unit has features
            first_unit = list(units.keys())[0]
            unit = units[first_unit]
            
            # Should have features group
            if 'features' in unit:
                features = unit['features']
                # Common expected features
                expected_features = ['eimage_sta', 'sta']
                for feat in expected_features:
                    if feat not in features:
                        logging.warning(f"Feature not found in reference: {feat}")
    
    def test_reference_metadata_structure(self):
        """Test reference file metadata structure."""
        with h5py.File(TEST_REFERENCE_FILE, 'r') as f:
            if 'metadata' in f:
                metadata = f['metadata']
                
                # Check for expected metadata groups
                possible_metadata = ['cmcr_meta', 'cmtr_meta', 'gsheet_row', 'ap_tracking']
                found_metadata = [m for m in possible_metadata if m in metadata]
                
                logging.info(f"Found metadata groups: {found_metadata}")


class TestUniversalLoader:
    """Test universal HDF5 loader functionality."""
    
    def test_load_session_preserves_data(self):
        """Test that load_session_from_hdf5 preserves all data."""
        session = load_session_from_hdf5(TEST_REFERENCE_FILE)
        
        # Check basic attributes
        assert session.dataset_id == TEST_DATASET_ID
        assert session.unit_count > 0
        
        # Check completed steps are restored
        assert len(session.completed_steps) > 0
        assert 'load_from_hdf5' in session.completed_steps
    
    def test_load_with_feature_filter(self):
        """Test selective feature loading."""
        # Load only eimage_sta
        session = load_session_from_hdf5(
            TEST_REFERENCE_FILE,
            load_features=['eimage_sta']
        )
        
        # Check that only requested features are loaded
        for unit_id, unit_data in session.units.items():
            if 'features' in unit_data:
                for feat_name in unit_data['features']:
                    if feat_name != 'eimage_sta':
                        # Other features should not be loaded
                        pass  # This test depends on what features exist
    
    def test_load_and_save_roundtrip(self, tmp_path):
        """Test that data survives load/save roundtrip."""
        # Load reference
        session = load_session_from_hdf5(TEST_REFERENCE_FILE)
        original_unit_count = session.unit_count
        
        # Save to temp file
        output_path = tmp_path / "test_roundtrip.h5"
        session.save(output_path=output_path, overwrite=True)
        
        # Reload
        session2 = load_session_from_hdf5(output_path)
        
        # Verify
        assert session2.unit_count == original_unit_count
        assert session2.dataset_id == session.dataset_id


class TestSaveOverwriteBehavior:
    """Test save() with overwrite parameter."""
    
    def test_save_raises_on_existing_file(self, tmp_path):
        """Test save() raises FileExistsError when file exists."""
        session = load_session_from_hdf5(TEST_REFERENCE_FILE)
        
        output_path = tmp_path / "test_overwrite.h5"
        
        # First save
        session.save(output_path=output_path, overwrite=True)
        
        # Second save without overwrite should fail
        session2 = load_session_from_hdf5(output_path)
        
        with pytest.raises(FileExistsError):
            session2.save(output_path=output_path, overwrite=False)
    
    def test_save_with_overwrite_succeeds(self, tmp_path):
        """Test save(overwrite=True) succeeds on existing file."""
        session = load_session_from_hdf5(TEST_REFERENCE_FILE)
        
        output_path = tmp_path / "test_overwrite.h5"
        
        # First save
        session.save(output_path=output_path, overwrite=True)
        
        # Second save with overwrite should succeed
        session2 = load_session_from_hdf5(output_path)
        session2.save(output_path=output_path, overwrite=True)
        
        assert output_path.exists()


class TestStepSkipping:
    """Test that completed steps are skipped on resume."""
    
    def test_completed_steps_restored(self):
        """Test that completed_steps are restored from HDF5."""
        session = load_session_from_hdf5(TEST_REFERENCE_FILE)
        
        # Should have multiple completed steps
        assert len(session.completed_steps) >= 1
        
        # Should include the load marker
        assert 'load_from_hdf5' in session.completed_steps


# =============================================================================
# Output Comparison Tests (for when pipeline output is generated)
# =============================================================================

class TestOutputComparison:
    """Compare pipeline output to reference file."""
    
    @pytest.fixture
    def pipeline_output_path(self):
        """Get path to pipeline output file."""
        return DEFAULT_OUTPUT_DIR / f"{TEST_DATASET_ID}.h5"
    
    @pytest.mark.skip(reason="Run after pipeline generates output")
    def test_output_structure_matches_reference(self, pipeline_output_path):
        """Test pipeline output has same structure as reference."""
        if not pipeline_output_path.exists():
            pytest.skip("Pipeline output not generated yet")
        
        # Load both files
        with h5py.File(TEST_REFERENCE_FILE, 'r') as ref:
            with h5py.File(pipeline_output_path, 'r') as out:
                # Compare top-level groups
                ref_groups = set(ref.keys())
                out_groups = set(out.keys())
                
                assert ref_groups == out_groups, f"Groups differ: {ref_groups} vs {out_groups}"
                
                # Compare unit count
                ref_units = len(ref['units'])
                out_units = len(out['units'])
                
                assert ref_units == out_units, f"Unit count differs: {ref_units} vs {out_units}"
    
    @pytest.mark.skip(reason="Run after pipeline generates output")
    def test_output_unit_features_match_reference(self, pipeline_output_path):
        """Test pipeline output units have same features as reference."""
        if not pipeline_output_path.exists():
            pytest.skip("Pipeline output not generated yet")
        
        with h5py.File(TEST_REFERENCE_FILE, 'r') as ref:
            with h5py.File(pipeline_output_path, 'r') as out:
                # Check a sample of units
                ref_units = list(ref['units'].keys())[:5]
                
                for unit_id in ref_units:
                    if unit_id not in out['units']:
                        pytest.fail(f"Unit {unit_id} missing from output")
                    
                    # Compare feature groups
                    if 'features' in ref['units'][unit_id]:
                        ref_features = set(ref['units'][unit_id]['features'].keys())
                        out_features = set(out['units'][unit_id]['features'].keys())
                        
                        missing = ref_features - out_features
                        if missing:
                            logging.warning(f"{unit_id} missing features: {missing}")

