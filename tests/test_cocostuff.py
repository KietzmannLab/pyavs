"""
Unit tests for COCO-Stuff integration in pyAVS.

Tests cover:
- Class definitions and integrity
- Thing/stuff classification
- Class name/ID mapping
- Annotation format validation
- Backward compatibility
- Edge cases

Author: pyAVS development team
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyavs.scenes.cocostuff_classes import (
    COCOSTUFF_CLASSES,
    THING_CLASS_INDICES,
    STUFF_CLASS_INDICES,
    MISSING_COCO_INDICES,
    get_class_name,
    get_class_id,
    is_thing_class,
    is_stuff_class,
    get_annotation_type,
    get_summary
)

from pyavs.scenes.objects import MSCOCO_CLASSES


class TestCOCOStuffClassDefinitions:
    """Test COCO-Stuff class definitions and constants."""

    def test_total_class_count(self):
        """Test that COCOSTUFF_CLASSES has exactly 183 classes."""
        assert len(COCOSTUFF_CLASSES) == 183, f"Expected 183 classes, got {len(COCOSTUFF_CLASSES)}"

    def test_thing_class_count(self):
        """Test that there are exactly 80 thing classes."""
        assert len(THING_CLASS_INDICES) == 80, f"Expected 80 thing classes, got {len(THING_CLASS_INDICES)}"

    def test_stuff_class_count(self):
        """Test that there are exactly 91 stuff classes."""
        assert len(STUFF_CLASS_INDICES) == 91, f"Expected 91 stuff classes, got {len(STUFF_CLASS_INDICES)}"

    def test_missing_coco_count(self):
        """Test that there are exactly 11 missing COCO indices."""
        assert len(MISSING_COCO_INDICES) == 11, f"Expected 11 missing indices, got {len(MISSING_COCO_INDICES)}"

    def test_first_class_unlabeled(self):
        """Test that index 0 is 'unlabeled'."""
        assert COCOSTUFF_CLASSES[0] == 'unlabeled', f"Index 0 should be 'unlabeled', got '{COCOSTUFF_CLASSES[0]}'"

    def test_first_thing_class(self):
        """Test that index 1 is 'person'."""
        assert COCOSTUFF_CLASSES[1] == 'person', f"Index 1 should be 'person', got '{COCOSTUFF_CLASSES[1]}'"

    def test_first_stuff_class(self):
        """Test that index 92 is 'banner' (first stuff class)."""
        assert COCOSTUFF_CLASSES[92] == 'banner', f"Index 92 should be 'banner', got '{COCOSTUFF_CLASSES[92]}'"

    def test_last_stuff_class(self):
        """Test that index 182 is 'wood' (last stuff class)."""
        assert COCOSTUFF_CLASSES[182] == 'wood', f"Index 182 should be 'wood', got '{COCOSTUFF_CLASSES[182]}'"

    def test_missing_indices_not_in_thing_classes(self):
        """Test that missing COCO indices are not in THING_CLASS_INDICES."""
        for idx in MISSING_COCO_INDICES:
            assert idx not in THING_CLASS_INDICES, f"Missing index {idx} should not be in THING_CLASS_INDICES"

    def test_thing_indices_range(self):
        """Test that thing class indices are in range [1, 91]."""
        for idx in THING_CLASS_INDICES:
            assert 1 <= idx <= 91, f"Thing class index {idx} out of range [1, 91]"

    def test_stuff_indices_range(self):
        """Test that stuff class indices are in range [92, 182]."""
        for idx in STUFF_CLASS_INDICES:
            assert 92 <= idx <= 182, f"Stuff class index {idx} out of range [92, 182]"

    def test_no_duplicate_classes(self):
        """Test that all class names are unique."""
        assert len(COCOSTUFF_CLASSES) == len(set(COCOSTUFF_CLASSES)), "Duplicate class names found"

    def test_thing_plus_stuff_equals_total(self):
        """Test that thing + stuff + missing + unlabeled = total."""
        expected_total = len(THING_CLASS_INDICES) + len(STUFF_CLASS_INDICES) + len(MISSING_COCO_INDICES) + 1  # +1 for unlabeled
        assert expected_total == len(COCOSTUFF_CLASSES), \
            f"Thing ({len(THING_CLASS_INDICES)}) + Stuff ({len(STUFF_CLASS_INDICES)}) + Missing ({len(MISSING_COCO_INDICES)}) + 1 != Total ({len(COCOSTUFF_CLASSES)})"


class TestClassNameIDMapping:
    """Test class name and ID mapping functions."""

    def test_get_class_name_valid(self):
        """Test get_class_name with valid indices."""
        assert get_class_name(0) == 'unlabeled'
        assert get_class_name(1) == 'person'
        assert get_class_name(92) == 'banner'
        assert get_class_name(182) == 'wood'

    def test_get_class_name_invalid(self):
        """Test get_class_name with invalid indices."""
        assert get_class_name(-1) == 'unknown'
        assert get_class_name(183) == 'unknown'
        assert get_class_name(999) == 'unknown'

    def test_get_class_id_valid(self):
        """Test get_class_id with valid class names."""
        assert get_class_id('unlabeled') == 0
        assert get_class_id('person') == 1
        assert get_class_id('banner') == 92
        assert get_class_id('wood') == 182

    def test_get_class_id_invalid(self):
        """Test get_class_id with invalid class names."""
        assert get_class_id('nonexistent') is None
        assert get_class_id('invalid_class') is None
        assert get_class_id('') is None

    def test_get_class_id_case_sensitive(self):
        """Test that get_class_id is case-sensitive."""
        assert get_class_id('Person') is None  # Should be 'person'
        assert get_class_id('PERSON') is None

    def test_roundtrip_name_id_mapping(self):
        """Test that name->ID->name roundtrip works for all classes."""
        for idx, name in enumerate(COCOSTUFF_CLASSES):
            class_id = get_class_id(name)
            assert class_id == idx, f"Roundtrip failed for '{name}': expected {idx}, got {class_id}"

            retrieved_name = get_class_name(class_id)
            assert retrieved_name == name, f"Roundtrip failed for ID {idx}: expected '{name}', got '{retrieved_name}'"


class TestThingStuffClassification:
    """Test thing/stuff classification functions."""

    def test_is_thing_class_valid_things(self):
        """Test is_thing_class with valid thing indices."""
        assert is_thing_class(1) is True  # person
        assert is_thing_class(3) is True  # car
        assert is_thing_class(62) is True  # chair

    def test_is_thing_class_stuff(self):
        """Test is_thing_class with stuff indices."""
        assert is_thing_class(92) is False  # banner
        assert is_thing_class(124) is False  # grass
        assert is_thing_class(182) is False  # wood

    def test_is_thing_class_missing(self):
        """Test is_thing_class with missing COCO indices."""
        for idx in MISSING_COCO_INDICES:
            assert is_thing_class(idx) is False, f"Missing index {idx} should not be classified as thing"

    def test_is_thing_class_unlabeled(self):
        """Test is_thing_class with unlabeled (index 0)."""
        assert is_thing_class(0) is False

    def test_is_stuff_class_valid_stuff(self):
        """Test is_stuff_class with valid stuff indices."""
        assert is_stuff_class(92) is True  # banner
        assert is_stuff_class(124) is True  # grass
        assert is_stuff_class(182) is True  # wood

    def test_is_stuff_class_things(self):
        """Test is_stuff_class with thing indices."""
        assert is_stuff_class(1) is False  # person
        assert is_stuff_class(3) is False  # car
        assert is_stuff_class(62) is False  # chair

    def test_is_stuff_class_unlabeled(self):
        """Test is_stuff_class with unlabeled (index 0)."""
        assert is_stuff_class(0) is False

    def test_mutually_exclusive_thing_stuff(self):
        """Test that no index is both thing and stuff."""
        for idx in range(183):
            if is_thing_class(idx):
                assert not is_stuff_class(idx), f"Index {idx} is both thing and stuff"
            if is_stuff_class(idx):
                assert not is_thing_class(idx), f"Index {idx} is both stuff and thing"

    def test_get_annotation_type_unlabeled(self):
        """Test get_annotation_type for unlabeled."""
        assert get_annotation_type(0) == 'unlabeled'

    def test_get_annotation_type_thing(self):
        """Test get_annotation_type for thing classes."""
        assert get_annotation_type(1) == 'thing'  # person
        assert get_annotation_type(3) == 'thing'  # car

    def test_get_annotation_type_stuff(self):
        """Test get_annotation_type for stuff classes."""
        assert get_annotation_type(92) == 'stuff'  # banner
        assert get_annotation_type(124) == 'stuff'  # grass

    def test_get_annotation_type_invalid(self):
        """Test get_annotation_type for invalid indices."""
        assert get_annotation_type(-1) == 'unknown'
        assert get_annotation_type(183) == 'unknown'
        assert get_annotation_type(999) == 'unknown'


class TestSummaryFunction:
    """Test the get_summary function."""

    def test_summary_keys(self):
        """Test that summary contains expected keys."""
        summary = get_summary()
        expected_keys = [
            'total_classes',
            'num_things',
            'num_stuff',
            'num_missing_coco',
            'thing_index_range',
            'stuff_index_range',
            'missing_coco_indices'
        ]
        for key in expected_keys:
            assert key in summary, f"Summary missing key: {key}"

    def test_summary_values(self):
        """Test that summary contains correct values."""
        summary = get_summary()
        assert summary['total_classes'] == 183
        assert summary['num_things'] == 80
        assert summary['num_stuff'] == 91
        assert summary['num_missing_coco'] == 11

    def test_summary_index_ranges(self):
        """Test that summary contains correct index ranges."""
        summary = get_summary()
        thing_min, thing_max = summary['thing_index_range']
        stuff_min, stuff_max = summary['stuff_index_range']

        assert thing_min == min(THING_CLASS_INDICES)
        assert thing_max == max(THING_CLASS_INDICES)
        assert stuff_min == min(STUFF_CLASS_INDICES)
        assert stuff_max == max(STUFF_CLASS_INDICES)

    def test_summary_missing_indices(self):
        """Test that summary contains correct missing indices."""
        summary = get_summary()
        assert summary['missing_coco_indices'] == MISSING_COCO_INDICES


class TestBackwardCompatibility:
    """Test backward compatibility with MSCOCO_CLASSES."""

    def test_mscoco_classes_count(self):
        """Test that MSCOCO_CLASSES has 80 classes."""
        assert len(MSCOCO_CLASSES) == 80, f"Expected 80 COCO classes, got {len(MSCOCO_CLASSES)}"

    def test_mscoco_classes_in_cocostuff(self):
        """Test that all MSCOCO classes exist in COCOSTUFF_CLASSES."""
        for coco_class in MSCOCO_CLASSES:
            assert coco_class in COCOSTUFF_CLASSES, f"COCO class '{coco_class}' not found in COCOSTUFF_CLASSES"

    def test_mscoco_classes_are_things(self):
        """Test that all MSCOCO classes are classified as things in COCO-Stuff."""
        for coco_class in MSCOCO_CLASSES:
            class_id = get_class_id(coco_class)
            assert class_id is not None, f"COCO class '{coco_class}' not found"
            assert is_thing_class(class_id), f"COCO class '{coco_class}' (ID {class_id}) not classified as thing"

    def test_thing_classes_superset_of_coco(self):
        """Test that THING_CLASS_INDICES contains all non-missing COCO classes."""
        # COCO classes are indices 1-90 (excluding missing)
        coco_indices = set(range(1, 91)) - set(MISSING_COCO_INDICES)
        thing_indices_set = set(THING_CLASS_INDICES)

        assert coco_indices == thing_indices_set, \
            f"THING_CLASS_INDICES doesn't match non-missing COCO indices"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_missing_indices_characteristics(self):
        """Test that missing indices have expected characteristics."""
        # Missing indices should be in range [1, 91] but not in THING_CLASS_INDICES
        for idx in MISSING_COCO_INDICES:
            assert 1 <= idx <= 91, f"Missing index {idx} out of expected range [1, 91]"
            assert idx not in THING_CLASS_INDICES, f"Missing index {idx} found in THING_CLASS_INDICES"

    def test_duplicate_class_suffixes(self):
        """Test that duplicate classes have proper suffixes."""
        # Known duplicates: desk/desk-stuff, door/door-stuff, mirror/mirror-stuff, window/window-other
        duplicate_pairs = [
            ('desk', 'desk-stuff'),
            ('door', 'door-stuff'),
            ('mirror', 'mirror-stuff'),
            ('window', 'window-other')
        ]

        for thing_name, stuff_name in duplicate_pairs:
            thing_id = get_class_id(thing_name)
            stuff_id = get_class_id(stuff_name)

            # Both should exist
            assert thing_id is not None, f"Thing class '{thing_name}' not found"
            assert stuff_id is not None, f"Stuff class '{stuff_name}' not found"

            # Thing should be in missing indices (these are missing in COCO)
            # Stuff should be in stuff class indices
            assert stuff_id in STUFF_CLASS_INDICES, f"'{stuff_name}' not classified as stuff"

    def test_boundary_indices(self):
        """Test boundary indices (0, 91, 92, 182, 183)."""
        # Index 0: unlabeled
        assert get_class_name(0) == 'unlabeled'
        assert get_annotation_type(0) == 'unlabeled'

        # Index 91: last potential thing index (hair brush - missing)
        assert get_class_name(91) == 'hair brush'
        assert 91 in MISSING_COCO_INDICES

        # Index 92: first stuff index
        assert get_class_name(92) == 'banner'
        assert is_stuff_class(92)

        # Index 182: last stuff index
        assert get_class_name(182) == 'wood'
        assert is_stuff_class(182)

        # Index 183: out of range
        assert get_class_name(183) == 'unknown'

    def test_all_indices_accounted_for(self):
        """Test that all indices 0-182 are accounted for."""
        accounted = set([0])  # unlabeled
        accounted.update(THING_CLASS_INDICES)
        accounted.update(STUFF_CLASS_INDICES)
        accounted.update(MISSING_COCO_INDICES)

        expected = set(range(183))
        assert accounted == expected, f"Not all indices accounted for. Missing: {expected - accounted}"

    def test_no_empty_class_names(self):
        """Test that no class name is empty."""
        for idx, name in enumerate(COCOSTUFF_CLASSES):
            assert len(name) > 0, f"Class at index {idx} has empty name"

    def test_class_name_format(self):
        """Test that class names follow expected format (lowercase, hyphens allowed)."""
        import re
        pattern = re.compile(r'^[a-z][a-z0-9\-]*$')

        for idx, name in enumerate(COCOSTUFF_CLASSES):
            assert pattern.match(name), f"Class '{name}' at index {idx} doesn't match expected format"


class TestAnnotationFormatValidation:
    """Test annotation format validation (integration test)."""

    def test_valid_cocostuff_annotation_ids(self):
        """Test that all valid COCO-Stuff IDs (0-182) are accepted."""
        # This would require importing FixationObjectChecker
        # For now, just test the classification functions
        for idx in range(183):
            annotation_type = get_annotation_type(idx)
            assert annotation_type in ['unlabeled', 'thing', 'stuff'], \
                f"Index {idx} has unexpected annotation type: {annotation_type}"

    def test_invalid_annotation_ids(self):
        """Test that invalid IDs are properly detected."""
        invalid_ids = [-1, 183, 200, 999]
        for idx in invalid_ids:
            assert get_annotation_type(idx) == 'unknown', f"Invalid ID {idx} not detected"


class TestSpecificClasses:
    """Test specific important classes."""

    def test_common_stuff_classes(self):
        """Test that common stuff classes are present."""
        common_stuff = ['sky-other', 'grass', 'wall-other', 'floor-other', 'tree', 'road', 'clouds', 'water-other']

        for class_name in common_stuff:
            class_id = get_class_id(class_name)
            assert class_id is not None, f"Common stuff class '{class_name}' not found"
            assert is_stuff_class(class_id), f"'{class_name}' not classified as stuff"

    def test_common_thing_classes(self):
        """Test that common thing classes are present."""
        common_things = ['person', 'car', 'chair', 'dog', 'cat', 'bird', 'bottle', 'cup']

        for class_name in common_things:
            class_id = get_class_id(class_name)
            assert class_id is not None, f"Common thing class '{class_name}' not found"
            assert is_thing_class(class_id), f"'{class_name}' not classified as thing"

    def test_person_is_first_thing(self):
        """Test that 'person' is at index 1 (first thing class)."""
        assert get_class_id('person') == 1
        assert get_class_name(1) == 'person'
        assert is_thing_class(1)

    def test_banner_is_first_stuff(self):
        """Test that 'banner' is at index 92 (first stuff class)."""
        assert get_class_id('banner') == 92
        assert get_class_name(92) == 'banner'
        assert is_stuff_class(92)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
