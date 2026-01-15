"""
COCO-Stuff class definitions and utilities.

This module provides constants and utilities for working with COCO-Stuff
annotations, which include 80 thing classes, 91 stuff classes, and 1 unlabeled class.

COCO-Stuff extends the COCO dataset with dense pixel-level annotations for amorphous
regions (stuff) like sky, grass, walls, water, etc. This provides comprehensive
scene segmentation for fixation object detection.

Class Structure:
- Index 0: unlabeled (background)
- Indices 1-91: Thing classes (80 actual classes with segmentation, 11 missing)
- Indices 92-182: Stuff classes (91 amorphous regions)
- Total: 183 classes (0-182)

References:
- COCO-Stuff paper: https://arxiv.org/abs/1612.03716
- GitHub: https://github.com/nightrome/cocostuff
- Labels: https://github.com/nightrome/cocostuff/blob/master/labels.md

Author: pyAVS development team
"""

from typing import Optional

# Complete list of 183 COCO-Stuff classes (indices 0-182)
# Index 0: unlabeled
# Indices 1-91: thing classes (80 actual classes, 11 missing indices)
# Indices 92-182: stuff classes (91 classes)
COCOSTUFF_CLASSES = [
    'unlabeled',         # 0
    'person',            # 1
    'bicycle',           # 2
    'car',               # 3
    'motorcycle',        # 4
    'airplane',          # 5
    'bus',               # 6
    'train',             # 7
    'truck',             # 8
    'boat',              # 9
    'traffic light',     # 10
    'fire hydrant',      # 11
    'street sign',       # 12 (missing in COCO)
    'stop sign',         # 13
    'parking meter',     # 14
    'bench',             # 15
    'bird',              # 16
    'cat',               # 17
    'dog',               # 18
    'horse',             # 19
    'sheep',             # 20
    'cow',               # 21
    'elephant',          # 22
    'bear',              # 23
    'zebra',             # 24
    'giraffe',           # 25
    'hat',               # 26 (missing in COCO)
    'backpack',          # 27
    'umbrella',          # 28
    'shoe',              # 29 (missing in COCO)
    'eye glasses',       # 30 (missing in COCO)
    'handbag',           # 31
    'tie',               # 32
    'suitcase',          # 33
    'frisbee',           # 34
    'skis',              # 35
    'snowboard',         # 36
    'sports ball',       # 37
    'kite',              # 38
    'baseball bat',      # 39
    'baseball glove',    # 40
    'skateboard',        # 41
    'surfboard',         # 42
    'tennis racket',     # 43
    'bottle',            # 44
    'plate',             # 45 (missing in COCO)
    'wine glass',        # 46
    'cup',               # 47
    'fork',              # 48
    'knife',             # 49
    'spoon',             # 50
    'bowl',              # 51
    'banana',            # 52
    'apple',             # 53
    'sandwich',          # 54
    'orange',            # 55
    'broccoli',          # 56
    'carrot',            # 57
    'hot dog',           # 58
    'pizza',             # 59
    'donut',             # 60
    'cake',              # 61
    'chair',             # 62
    'couch',             # 63
    'potted plant',      # 64
    'bed',               # 65
    'mirror',            # 66 (missing in COCO)
    'dining table',      # 67
    'window',            # 68 (missing in COCO)
    'desk',              # 69 (missing in COCO)
    'toilet',            # 70
    'door',              # 71 (missing in COCO)
    'tv',                # 72
    'laptop',            # 73
    'mouse',             # 74
    'remote',            # 75
    'keyboard',          # 76
    'cell phone',        # 77
    'microwave',         # 78
    'oven',              # 79
    'toaster',           # 80
    'sink',              # 81
    'refrigerator',      # 82
    'blender',           # 83 (missing in COCO)
    'book',              # 84
    'clock',             # 85
    'vase',              # 86
    'scissors',          # 87
    'teddy bear',        # 88
    'hair drier',        # 89
    'toothbrush',        # 90
    'hair brush',        # 91 (missing in COCO)
    'banner',            # 92 - START OF STUFF CLASSES
    'blanket',           # 93
    'branch',            # 94
    'bridge',            # 95
    'building-other',    # 96
    'bush',              # 97
    'cabinet',           # 98
    'cage',              # 99
    'cardboard',         # 100
    'carpet',            # 101
    'ceiling-other',     # 102
    'ceiling-tile',      # 103
    'cloth',             # 104
    'clothes',           # 105
    'clouds',            # 106
    'counter',           # 107
    'cupboard',          # 108
    'curtain',           # 109
    'desk-stuff',        # 110
    'dirt',              # 111
    'door-stuff',        # 112
    'fence',             # 113
    'floor-marble',      # 114
    'floor-other',       # 115
    'floor-stone',       # 116
    'floor-tile',        # 117
    'floor-wood',        # 118
    'flower',            # 119
    'fog',               # 120
    'food-other',        # 121
    'fruit',             # 122
    'furniture-other',   # 123
    'grass',             # 124
    'gravel',            # 125
    'ground-other',      # 126
    'hill',              # 127
    'house',             # 128
    'leaves',            # 129
    'light',             # 130
    'mat',               # 131
    'metal',             # 132
    'mirror-stuff',      # 133
    'moss',              # 134
    'mountain',          # 135
    'mud',               # 136
    'napkin',            # 137
    'net',               # 138
    'paper',             # 139
    'pavement',          # 140
    'pillow',            # 141
    'plant-other',       # 142
    'plastic',           # 143
    'platform',          # 144
    'playingfield',      # 145
    'railing',           # 146
    'railroad',          # 147
    'river',             # 148
    'road',              # 149
    'rock',              # 150
    'roof',              # 151
    'rug',               # 152
    'salad',             # 153
    'sand',              # 154
    'sea',               # 155
    'shelf',             # 156
    'sky-other',         # 157
    'skyscraper',        # 158
    'snow',              # 159
    'solid-other',       # 160
    'stairs',            # 161
    'stone',             # 162
    'straw',             # 163
    'structural-other',  # 164
    'table',             # 165
    'tent',              # 166
    'textile-other',     # 167
    'towel',             # 168
    'tree',              # 169
    'vegetable',         # 170
    'wall-brick',        # 171
    'wall-concrete',     # 172
    'wall-other',        # 173
    'wall-panel',        # 174
    'wall-stone',        # 175
    'wall-tile',         # 176
    'wall-wood',         # 177
    'water-other',       # 178
    'waterdrops',        # 179
    'window-blind',      # 180
    'window-other',      # 181
    'wood'               # 182
]

# Indices that are missing in COCO (no segmentation annotations)
# These thing classes exist in COCO-Stuff but lack instance segmentations in original COCO
MISSING_COCO_INDICES = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]

# Thing class indices (1-91, excluding missing)
THING_CLASS_INDICES = [i for i in range(1, 92) if i not in MISSING_COCO_INDICES]

# Stuff class indices (92-182)
STUFF_CLASS_INDICES = list(range(92, 183))

# Classes that exist in both COCO and COCO-Stuff with suffix disambiguation
# In COCO-Stuff, these have "-stuff" or "-other" suffix to distinguish from thing versions
DUPLICATE_CLASSES = {
    'desk-stuff': 'desk',      # desk-stuff (110) vs desk (69)
    'door-stuff': 'door',      # door-stuff (112) vs door (71)
    'mirror-stuff': 'mirror',  # mirror-stuff (133) vs mirror (66)
    'window-other': 'window'   # window-other (181) vs window (68)
}


def get_class_name(class_id: int) -> str:
    """
    Get class name from COCO-Stuff class ID.

    Parameters
    ----------
    class_id : int
        COCO-Stuff class ID (0-182)

    Returns
    -------
    str
        Class name, or 'unknown' if ID is out of range

    Examples
    --------
    >>> get_class_name(0)
    'unlabeled'
    >>> get_class_name(1)
    'person'
    >>> get_class_name(92)
    'banner'
    >>> get_class_name(182)
    'wood'
    """
    if 0 <= class_id < len(COCOSTUFF_CLASSES):
        return COCOSTUFF_CLASSES[class_id]
    return 'unknown'


def get_class_id(class_name: str) -> Optional[int]:
    """
    Get COCO-Stuff class ID from class name.

    Parameters
    ----------
    class_name : str
        Class name (e.g., 'person', 'sky-other')

    Returns
    -------
    int or None
        Class ID (0-182), or None if name not found

    Examples
    --------
    >>> get_class_id('person')
    1
    >>> get_class_id('banner')
    92
    >>> get_class_id('nonexistent')
    None
    """
    try:
        return COCOSTUFF_CLASSES.index(class_name)
    except ValueError:
        return None


def is_thing_class(class_id: int) -> bool:
    """
    Check if class ID represents a thing class.

    Thing classes are countable objects with defined boundaries (1-91, excluding missing).

    Parameters
    ----------
    class_id : int
        COCO-Stuff class ID

    Returns
    -------
    bool
        True if class is a thing, False otherwise

    Examples
    --------
    >>> is_thing_class(1)  # person
    True
    >>> is_thing_class(92)  # banner (stuff)
    False
    >>> is_thing_class(0)  # unlabeled
    False
    """
    return class_id in THING_CLASS_INDICES


def is_stuff_class(class_id: int) -> bool:
    """
    Check if class ID represents a stuff class.

    Stuff classes are amorphous regions without defined boundaries (92-182).

    Parameters
    ----------
    class_id : int
        COCO-Stuff class ID

    Returns
    -------
    bool
        True if class is stuff, False otherwise

    Examples
    --------
    >>> is_stuff_class(92)  # banner
    True
    >>> is_stuff_class(182)  # wood
    True
    >>> is_stuff_class(1)  # person (thing)
    False
    """
    return class_id in STUFF_CLASS_INDICES


def get_annotation_type(class_id: int) -> str:
    """
    Get annotation type: 'thing', 'stuff', 'unlabeled', or 'unknown'.

    Parameters
    ----------
    class_id : int
        COCO-Stuff class ID

    Returns
    -------
    str
        Annotation type

    Examples
    --------
    >>> get_annotation_type(0)
    'unlabeled'
    >>> get_annotation_type(1)
    'thing'
    >>> get_annotation_type(92)
    'stuff'
    >>> get_annotation_type(999)
    'unknown'
    """
    if class_id == 0:
        return 'unlabeled'
    elif is_thing_class(class_id):
        return 'thing'
    elif is_stuff_class(class_id):
        return 'stuff'
    return 'unknown'


def get_summary() -> dict:
    """
    Get summary statistics about COCO-Stuff classes.

    Returns
    -------
    dict
        Dictionary with class counts and index ranges

    Examples
    --------
    >>> summary = get_summary()
    >>> summary['total_classes']
    183
    >>> summary['num_things']
    80
    >>> summary['num_stuff']
    91
    """
    return {
        'total_classes': len(COCOSTUFF_CLASSES),
        'num_things': len(THING_CLASS_INDICES),
        'num_stuff': len(STUFF_CLASS_INDICES),
        'num_missing_coco': len(MISSING_COCO_INDICES),
        'thing_index_range': (min(THING_CLASS_INDICES), max(THING_CLASS_INDICES)),
        'stuff_index_range': (min(STUFF_CLASS_INDICES), max(STUFF_CLASS_INDICES)),
        'missing_coco_indices': MISSING_COCO_INDICES
    }


# Validate class list integrity on module import
assert len(COCOSTUFF_CLASSES) == 183, f"Expected 183 classes, got {len(COCOSTUFF_CLASSES)}"
assert len(THING_CLASS_INDICES) == 80, f"Expected 80 thing classes, got {len(THING_CLASS_INDICES)}"
assert len(STUFF_CLASS_INDICES) == 91, f"Expected 91 stuff classes, got {len(STUFF_CLASS_INDICES)}"
assert COCOSTUFF_CLASSES[0] == 'unlabeled', "Index 0 must be 'unlabeled'"
assert COCOSTUFF_CLASSES[1] == 'person', "Index 1 must be 'person'"
assert COCOSTUFF_CLASSES[92] == 'banner', "Index 92 must be 'banner' (first stuff class)"
assert COCOSTUFF_CLASSES[182] == 'wood', "Index 182 must be 'wood' (last stuff class)"
