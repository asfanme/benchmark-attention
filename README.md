## Experiment Notebooks and Naming Convention

The repository contains ten Jupyter notebooks representing five model
configurations evaluated under two cross-validation protocols.

The original notebook filenames are retained to preserve consistency with
the completed experiments and stored result directories. In the manuscript,
the experiments are referred to as E1--E10.

### Experimental mapping

| Manuscript ID | Notebook file | Validation protocol | Model configuration |
|---|---|---|---|
| E1 | `exp0 convnext-tiny+focal loss.ipynb` | Image-level Stratified 5-Fold | ConvNeXt-Tiny baseline with Focal Loss |
| E2 | `exp1 convnext-tiny+SE attention + focal loss.ipynb` | Image-level Stratified 5-Fold | ConvNeXt-Tiny with SE-based Channel Attention and Focal Loss |
| E3 | `exp2 convnext-tiny+Spatial attention + focal loss.ipynb` | Image-level Stratified 5-Fold | ConvNeXt-Tiny with Spatial Attention and Focal Loss |
| E4 | `exp3 convnext-tiny+CBAM + focal loss.ipynb` | Image-level Stratified 5-Fold | ConvNeXt-Tiny with CBAM and Focal Loss |
| E5 | `exp4 convnext-tiny+CA + focal loss.ipynb` | Image-level Stratified 5-Fold | ConvNeXt-Tiny with Coordinate Attention and Focal Loss |
| E6 | `exp0 convnext-tiny+focal loss_GroupKFold.ipynb` | Lesion-level Group 5-Fold | ConvNeXt-Tiny baseline with Focal Loss |
| E7 | `exp1 convnext-tiny+SE attention + focal loss_GroupKFold.ipynb` | Lesion-level Group 5-Fold | ConvNeXt-Tiny with SE-based Channel Attention and Focal Loss |
| E8 | `exp2 convnext-tiny+Spatial attention + focal loss_GroupKFold.ipynb` | Lesion-level Group 5-Fold | ConvNeXt-Tiny with Spatial Attention and Focal Loss |
| E9 | `exp3 convnext-tiny+CBAM + focal loss_GroupKFold.ipynb` | Lesion-level Group 5-Fold | ConvNeXt-Tiny with CBAM and Focal Loss |
| E10 | `exp4 convnext-tiny+CA + focal loss_GroupKFold.ipynb` | Lesion-level Group 5-Fold | ConvNeXt-Tiny with Coordinate Attention and Focal Loss |

### Naming notes

- `exp0` denotes the attention-free ConvNeXt-Tiny baseline.
- `exp1` denotes the SE-based Channel Attention configuration.
- `exp2` denotes the Spatial Attention configuration.
- `exp3` denotes the Convolutional Block Attention Module (CBAM).
- `exp4` denotes the Coordinate Attention configuration.
- Files without the `_GroupKFold` suffix use image-level Stratified
  five-fold cross-validation.
- Files with the `_GroupKFold` suffix use lesion-level Group five-fold
  cross-validation, with `lesion_id` used as the grouping variable.
- In the `exp4` filenames, the abbreviation `CA` refers to
  **Coordinate Attention**, not Channel Attention.
- The SE block in `exp1` is reported in the manuscript under the broader
  term **Channel Attention**.

The two validation protocols use the same model configurations and training
settings. Their primary difference is the data-partitioning strategy.
