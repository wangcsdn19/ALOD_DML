# Teacher Pseudo-Label Quality Thresholds in ALOD_DML

## Overview

The semi-supervised learning pipeline uses a **teacher–student** architecture where the teacher
model generates pseudo-labels for unlabeled images. Two confidence-score thresholds gate the
quality of pseudo-labels admitted to the student training:

| Symbol | Config key | Default | Role |
|--------|-----------|---------|------|
| `tau_h` | `pseudo_label_real_score_thr` | **0.9** | High-quality threshold – only boxes with score ≥ τ_h are used as supervised pseudo-labels |
| `tau_l` | `pseudo_label_initial_score_thr` | **0.0** | Low-quality threshold – coarse pre-filter; effectively disabled at the default value of 0.0 |

---

## Files & Locations

| File | Line(s) | What is defined |
|------|---------|-----------------|
| `configs/K_ALOD_dotav1/1_ALOD_dotav1.py` | 40–41 | Config values for `tau_h` / `tau_l` |
| `src/models/dml_alod.py` | 46–51 | Module-level constant defaults (`DEFAULT_TAU_H`, `DEFAULT_TAU_L`) |
| `src/models/dml_alod.py` | `compute_quality_thresholds()` | **Encapsulated** helper that returns `(tau_h, tau_l)` with validation |
| `src/models/dml_alod.py` | `extract_teacher_info()` | Applies `tau_l` then `tau_h` in a two-stage filter |
| `src/models/dml_alod.py` | `_build_img_pseudo_bank()` | Uses `img_pseudo_thr` (τ_patch = 0.7) for CutMix patch library |
| `src/models/utils/bbox_utils.py` | `filter_invalid()` | Core per-box filtering primitive |
| `src/models/utils/bbox_utils.py` | `get_adaptive_cls_base_thr()` | Per-class percentile-based adaptive threshold [0.5, 0.9] |
| `src/models/utils/bbox_utils.py` | `get_adaptive_thr_gmm()` | GMM-based adaptive threshold [0.2, 0.6] |

---

## Threshold Roles in Detail

### τ_h — High-quality threshold (`pseudo_label_real_score_thr = 0.9`)

Used to admit **high-confidence** teacher predictions as supervised pseudo-labels.

**Application (two-pass filter in `extract_teacher_info`):**

```
Stage 1 (lower bound):  keep boxes where score > τ_l   (≈ keep all at τ_l = 0.0)
Stage 2 (upper bound):  keep boxes where score < τ_h   (inverted: score ≥ τ_h retained)
Stage 3 (exact):        keep boxes where score > τ_h   (final confirmation)
```

After all three stages only boxes with `score ≥ τ_h` survive.  These become the
**supervised pseudo-label targets** for the student's detection head.

Additionally, boxes with `score ≥ img_pseudo_thr` (default 0.7) are used to build
the **CutMix patch library** (`_build_img_pseudo_bank`).

### τ_l — Low-quality threshold (`pseudo_label_initial_score_thr = 0.00`)

Acts as a coarse initial filter.  At the default value of 0.0 it discards only
zero-score boxes.  Raising it (e.g. to 0.3) would pre-filter noisy low-confidence
detections before the expensive `merge_rboxes` step.

### Additional Related Thresholds

| Variable | Config key | Default | Purpose |
|----------|-----------|---------|---------|
| `img_pseudo_thr` | `img_pseudo_thr` | 0.7 | CutMix patch library – boxes above this score are cropped as patch candidates |
| `mask_thr` | `mask_thr` | 0.2 | CutMix augmentation – boxes below this score are filled with gray instead of real patches |
| `mining_th_score` | `mining_th_score` | 0.6 | Mining quality gate for pseudo-label candidates |

---

## Type / Update Mechanism

Both `tau_h` and `tau_l` are **fixed constants** (static floats read once from the
config at construction time). They do **not** change during training.

```python
# pseudo-code for compute_quality_thresholds
def compute_quality_thresholds(train_cfg):
    tau_h = train_cfg.pseudo_label_real_score_thr     # fixed float, default 0.9
    tau_l = train_cfg.pseudo_label_initial_score_thr  # fixed float, default 0.0
    return tau_h, tau_l
```

For **per-class adaptive thresholds** (not the primary pseudo-label gate) see:
- `get_adaptive_cls_base_thr()` – maintains a sliding window of per-class scores
  and updates the threshold every iteration using the `percent`-th percentile,
  clipped to `[min=0.5, max=0.9]`.
- `get_adaptive_thr_gmm()` – same sliding window, threshold determined by
  fitting a 2-component GMM to the score distribution and picking the valley
  between the two components, clipped to `[min=0.2, max=0.6]`.

---

## Configuration

Edit `configs/K_ALOD_dotav1/1_ALOD_dotav1.py`:

```python
semi_wrapper = dict(
    type="DML_ALOD",
    train_cfg=dict(
        # ── Pseudo-label quality thresholds ──────────────────────────────────
        # τ_h: high-quality threshold. Only teacher predictions with
        #      confidence score ≥ tau_h are used as supervised pseudo-labels.
        #      Range: (0, 1]. Recommended: 0.8 – 0.95.
        pseudo_label_real_score_thr=0.9,

        # τ_l: low-quality threshold. Initial coarse pre-filter.
        #      At 0.0 (default) it is effectively disabled.
        #      Raise to ~0.2–0.4 to skip obviously noisy detections early.
        pseudo_label_initial_score_thr=0.00,
        ...
    ),
)
```

---

## Training Log

At every iteration `extract_teacher_info` logs the active thresholds at `DEBUG`
level via `log_every_n`:

```
[DML_ALOD] extract_teacher_info: tau_high=0.900, tau_low=0.000
```

This is emitted once every 50 iterations by default (controlled by the `n`
parameter of `log_every_n`).  Increase the frequency by passing `n=1` or
lower to track threshold changes more closely when experimenting.

---

## Default Values Summary

| Config key | Default | Source |
|-----------|---------|--------|
| `pseudo_label_real_score_thr` | 0.9 | `configs/K_ALOD_dotav1/1_ALOD_dotav1.py:40` |
| `pseudo_label_initial_score_thr` | 0.0 | `configs/K_ALOD_dotav1/1_ALOD_dotav1.py:41` |
| `img_pseudo_thr` | 0.7 | `dml_alod.py` `thr_after_cali` constant |
| `mask_thr` | 0.2 | `configs/K_ALOD_dotav1/1_ALOD_dotav1.py:55` |
| `mining_th_score` | 0.6 | `configs/K_ALOD_dotav1/1_ALOD_dotav1.py:45` |
