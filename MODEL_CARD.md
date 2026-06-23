# Model Card — Osteoporosis Dental X-ray Classifier

## Model
- Architecture: EfficientNetB0 backbone (ImageNet) + GAP + dense head, softmax(3)
- Backbone training: two-phase (head @ 0.001, then 12 top-block layers @ 1e-05, BN frozen); BatchNorm frozen; flips-only augmentation
- Exposure normalisation: clahe; patch sub-sampling STRIDE=2
- File: osteo_efficientnetb0.keras
- SHA-256: c31408c4a110b09dade5d425e5807a4273f069e9e5e02081f74d032bfe27e82e

## Data
- Mendeley dental periapical radiograph patches, 100x100, 3 classes
- 13 sources (Normal=3,
  Osteopenia=6,
  Osteoporosis=4); one class per source
- Deployment model trained on 11 sources; ['roiant5', 'roiant2'] held out for val_loss

## Evaluation (Leave-One-Source-Out, source-level)
- macro-F1: 0.5152   accuracy: 0.5385 (7/13)
- Brightness baseline (same protocol): macro-F1 0.5333, acc 0.5385
- Per-class F1: Normal 0.400, Osteopenia 0.600, Osteoporosis 0.545

## Deployment status
- RESEARCH PROTOTYPE -- NOT a medical device and NOT fit for clinical or production
  diagnostic use. Under patient-independent (LOSO) evaluation it performs at/around a
  one-feature brightness baseline with Normal recall near zero; screening real patients
  with it would produce unsafe, unvalidated osteoporosis calls.

## Limitations
- Only 3 Normal sources -> Normal generalisation is severely data-limited (recall ~0).
- 10 training sources per LOSO fold -> high variance; treat single-run differences as noise.
- A brightness-only baseline matches/beats the CNN -> much of the separability is likely an
  acquisition/exposure artifact that will not transfer to new scanners.
- A deployable model would need hundreds of DEXA-confirmed patients across multiple sites,
  standardised acquisition, calibration, and prospective external validation.

Seed=42, TensorFlow=2.19.0, op-determinism=True.
