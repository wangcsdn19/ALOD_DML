"""
Unit tests for pseudo-label quality threshold logic.

Tests cover:
  - ``compute_quality_thresholds``: validation and default handling
  - ``filter_invalid``: core filtering primitive used in the two-stage filter

Run with:
    python -m pytest tests/test_quality_thresholds.py -v
"""

import sys
import os
import types
import importlib.util

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Import modules under test directly by file path to avoid heavy dependencies
# ---------------------------------------------------------------------------


_REPO = os.path.join(os.path.dirname(__file__), '..')

# ---------------------------------------------------------------------------
# Provide minimal stubs for heavy deps before loading source modules
# ---------------------------------------------------------------------------
import types as _types


def _load_module_from_path(name, path):
    """Load a Python source file as a module without running its full imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub(dotted, attrs=None):
    """Create and register a stub module for ``dotted``, setting ``attrs``."""
    parts = dotted.split('.')
    for i in range(1, len(parts) + 1):
        full = '.'.join(parts[:i])
        if full not in sys.modules:
            parent_name = '.'.join(parts[:i - 1])
            mod = _types.ModuleType(full)
            if parent_name and parent_name in sys.modules:
                setattr(sys.modules[parent_name], parts[i - 1], mod)
            sys.modules[full] = mod
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[dotted], k, v)
    return sys.modules[dotted]


# BitmapMasks stub (used only in filter_invalid's mask branch, which our tests
# don't exercise – a real BitmapMasks would require additional logic)
class _BitmapMasks:
    def __init__(self, masks, height, width):
        self.masks = masks
        self.height = height
        self.width = width

_stub('mmdet')
_stub('mmdet.core')
_stub('mmdet.core.mask')
_stub('mmdet.core.mask.structures', {'BitmapMasks': _BitmapMasks})
_stub('mmrotate')
_stub('mmrotate.core')
_stub('mmrotate.core.bbox')
_stub('mmrotate.core.bbox.transforms', {
    'poly2obb_le90': lambda x: x,
    'obb2poly_le90': lambda x: x,
})
_stub('sklearn')
_stub('sklearn.mixture')

# ---------------------------------------------------------------------------
# Import modules under test directly by file path
# ---------------------------------------------------------------------------

_bbox_utils = _load_module_from_path(
    '_test_bbox_utils',
    os.path.join(_REPO, 'src', 'models', 'utils', 'bbox_utils.py'),
)
filter_invalid = _bbox_utils.filter_invalid


# -- dml_alod: pull out only the pure helpers we want to test ----------------
# We extract compute_quality_thresholds by evaluating just the relevant snippet
# (the function is self-contained and only uses builtins + getattr).
_dml_src = os.path.join(_REPO, 'src', 'models', 'dml_alod.py')
_dml_globals = {'__name__': '_test_dml_alod', '__builtins__': __builtins__}

with open(_dml_src) as _f:
    _lines = _f.readlines()

# Extract the DEFAULT_TAU_* constant lines and the compute_quality_thresholds
# function.  We collect the function body by tracking indentation: the function
# ends when we see a new top-level (non-indented, non-blank) definition after
# the first line of the function body.
_snippet_lines = []
_in_func = False
_past_first_body_line = False

for _line in _lines:
    stripped = _line.strip()
    if stripped.startswith('DEFAULT_TAU_H') or stripped.startswith('DEFAULT_TAU_L'):
        _snippet_lines.append(_line)
        continue
    if stripped.startswith('def compute_quality_thresholds('):
        _in_func = True
        _past_first_body_line = False
        _snippet_lines.append(_line)
        continue
    if _in_func:
        is_top_level_def = (
            stripped
            and not stripped.startswith('#')
            and not _line.startswith(' ')
            and not _line.startswith('\t')
            and (stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('@'))
        )
        if _past_first_body_line and is_top_level_def:
            _in_func = False
        else:
            _snippet_lines.append(_line)
            if stripped:
                _past_first_body_line = True

exec(''.join(_snippet_lines), _dml_globals)
compute_quality_thresholds = _dml_globals['compute_quality_thresholds']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Cfg:
    """Minimal config object."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _boxes(scores):
    """Build a (N, 6) bbox tensor where columns 2,3 are w,h=1 and column 5 is the score."""
    n = len(scores)
    boxes = torch.zeros(n, 6)
    boxes[:, 2] = 1.0  # width
    boxes[:, 3] = 1.0  # height
    boxes[:, 5] = torch.tensor(scores, dtype=torch.float32)
    return boxes


# ---------------------------------------------------------------------------
# Tests for compute_quality_thresholds
# ---------------------------------------------------------------------------

class TestComputeQualityThresholds:
    def test_reads_from_cfg(self):
        cfg = _Cfg(pseudo_label_real_score_thr=0.8,
                   pseudo_label_initial_score_thr=0.2)
        tau_h, tau_l = compute_quality_thresholds(cfg)
        assert tau_h == 0.8
        assert tau_l == 0.2

    def test_defaults_when_attrs_missing(self):
        cfg = _Cfg()  # no threshold attrs
        tau_h, tau_l = compute_quality_thresholds(cfg)
        assert tau_h == 0.9   # DEFAULT_TAU_H
        assert tau_l == 0.0   # DEFAULT_TAU_L

    def test_tau_l_zero_is_valid(self):
        cfg = _Cfg(pseudo_label_real_score_thr=0.9,
                   pseudo_label_initial_score_thr=0.0)
        tau_h, tau_l = compute_quality_thresholds(cfg)
        assert tau_l == 0.0

    def test_equal_thresholds_valid(self):
        cfg = _Cfg(pseudo_label_real_score_thr=0.5,
                   pseudo_label_initial_score_thr=0.5)
        tau_h, tau_l = compute_quality_thresholds(cfg)
        assert tau_h == tau_l == 0.5

    def test_tau_l_greater_than_tau_h_raises(self):
        cfg = _Cfg(pseudo_label_real_score_thr=0.3,
                   pseudo_label_initial_score_thr=0.8)
        with pytest.raises(ValueError, match="tau_l"):
            compute_quality_thresholds(cfg)

    def test_tau_h_out_of_range_raises(self):
        cfg = _Cfg(pseudo_label_real_score_thr=1.5,
                   pseudo_label_initial_score_thr=0.0)
        with pytest.raises(ValueError, match="tau_h"):
            compute_quality_thresholds(cfg)

    def test_tau_l_negative_raises(self):
        cfg = _Cfg(pseudo_label_real_score_thr=0.9,
                   pseudo_label_initial_score_thr=-0.1)
        with pytest.raises(ValueError, match="tau_l"):
            compute_quality_thresholds(cfg)


# ---------------------------------------------------------------------------
# Tests for filter_invalid (core filtering primitive)
# ---------------------------------------------------------------------------

class TestFilterInvalid:
    def test_basic_score_filter(self):
        scores = [0.1, 0.5, 0.91, 0.95]
        boxes = _boxes(scores)
        labels = torch.tensor([0, 1, 2, 3])

        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 2  # 0.91, 0.95 survive
        assert out_labels.tolist() == [2, 3]

    def test_tau_l_zero_keeps_all_nonzero(self):
        """At tau_l=0.0 all positive-score boxes are kept."""
        scores = [0.01, 0.5, 0.99]
        boxes = _boxes(scores)
        labels = torch.tensor([0, 1, 2])

        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.0
        )
        assert out_boxes.shape[0] == 3

    def test_all_boxes_below_threshold(self):
        """When nothing passes, output tensors are empty."""
        scores = [0.1, 0.2, 0.3]
        boxes = _boxes(scores)
        labels = torch.tensor([0, 1, 2])

        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 0
        assert out_labels.shape[0] == 0

    def test_all_boxes_above_threshold(self):
        """When all pass, nothing is removed."""
        scores = [0.91, 0.95, 0.99]
        boxes = _boxes(scores)
        labels = torch.tensor([0, 1, 2])

        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 3

    def test_all_identical_scores_at_boundary(self):
        """Scores exactly equal to thr are NOT kept (strict > comparison)."""
        scores = [0.9, 0.9, 0.9]
        boxes = _boxes(scores)
        labels = torch.tensor([0, 1, 2])

        out_boxes, _, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 0

    def test_negation_trick_strict_greater(self):
        """filter_invalid uses strict > ; only 0.91 passes thr=0.9."""
        scores = [0.89, 0.90, 0.91]
        boxes = _boxes(scores)
        labels = torch.tensor([0, 1, 2])
        tau_h = 0.9

        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=tau_h
        )
        # Only 0.91 > 0.9
        assert out_boxes.shape[0] == 1
        assert out_labels.tolist() == [2]

    def test_empty_input(self):
        """Empty tensor input should return empty tensors without error."""
        boxes = torch.zeros(0, 6)
        labels = torch.zeros(0, dtype=torch.long)

        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 0
        assert out_labels.shape[0] == 0

    def test_single_box_passes(self):
        boxes = _boxes([0.95])
        labels = torch.tensor([0])
        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 1

    def test_single_box_fails(self):
        boxes = _boxes([0.5])
        labels = torch.tensor([0])
        out_boxes, out_labels, _ = filter_invalid(
            bbox=boxes, label=labels, score=boxes[:, 5], thr=0.9
        )
        assert out_boxes.shape[0] == 0

    def test_no_score_no_label_no_mask(self):
        """When score/label/mask are None only min_size filter applies."""
        boxes = torch.tensor([[0, 0, 5, 5, 0, 1.0],
                               [0, 0, 1, 1, 0, 0.5]], dtype=torch.float32)
        out_boxes, out_label, out_mask = filter_invalid(
            bbox=boxes, thr=0.0, min_size=2
        )
        # Only the 5×5 box survives the min_size filter
        assert out_boxes.shape[0] == 1
        assert out_label is None
        assert out_mask is None

