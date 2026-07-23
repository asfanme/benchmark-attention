# Benchmarking Attention Mechanisms in ConvNeXt-Tiny

Official implementation and reproducibility resources for benchmarking
Channel, Spatial, CBAM, and Coordinate Attention on ConvNeXt-Tiny for
HAM10000 skin-lesion classification under image-level stratified and
lesion-level group cross-validation.

## Overview

This repository contains the experiment notebooks, statistical-analysis
scripts, compact result summaries, and environment specifications used in
the associated study.
## Dataset Preparation

This project uses the **HAM10000 (Human Against Machine with 10,000 Training Images)** dermoscopic image dataset.

The dataset is not included in this repository. It must be downloaded separately from the official Harvard Dataverse repository:

- **Official dataset:** [HAM10000 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **Dataset DOI:** [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)

Please review the dataset description, citation requirements, and terms of use before downloading or redistributing the data.

### Expected Directory Structure

For this project, the HAM10000 images are organized into seven diagnostic class folders. The metadata CSV file is placed in the same root directory.

```text
data/
└── HAM10000/
    ├── akiec/
    │   └── *.jpg
    ├── bcc/
    │   └── *.jpg
    ├── bkl/
    │   └── *.jpg
    ├── df/
    │   └── *.jpg
    ├── mel/
    │   └── *.jpg
    ├── nv/
    │   └── *.jpg
    ├── vasc/
    │   └── *.jpg
    └── HAM10000_metadata.csv
```

### Diagnostic Classes

| Folder | Diagnostic category |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

### Metadata

The `HAM10000_metadata.csv` file provides metadata associated with each image. The main fields used in this project are:

- `image_id`: unique image identifier;
- `dx`: diagnostic class label;
- `lesion_id`: identifier shared by images belonging to the same lesion.

The class folders are used to locate image files, while the metadata file provides the `lesion_id` required for lesion-level group cross-validation.

### Dataset Path Configuration

Update the dataset path in the notebooks when the dataset is stored in a different directory.

```python
from pathlib import Path

DATA_DIR = Path("data/HAM10000")
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]
```

### Loading the Metadata

```python
import pandas as pd

metadata = pd.read_csv(METADATA_PATH)

required_columns = {"image_id", "dx", "lesion_id"}
missing_columns = required_columns.difference(metadata.columns)

if missing_columns:
    raise ValueError(
        f"Missing required metadata columns: {sorted(missing_columns)}"
    )

print("Metadata records:", len(metadata))
print("Unique lesions:", metadata["lesion_id"].nunique())
print(metadata["dx"].value_counts())
```

### Loading Images from the Seven Class Folders

```python
image_records = []

for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class directory was not found: {class_dir}"
        )

    image_paths = sorted(
        list(class_dir.glob("*.jpg"))
        + list(class_dir.glob("*.jpeg"))
        + list(class_dir.glob("*.png"))
    )

    for image_path in image_paths:
        image_records.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "folder_label": class_name,
            }
        )

images_df = pd.DataFrame(image_records)

print("Discovered images:", len(images_df))
print(images_df["folder_label"].value_counts())
```

### Combining Image Paths and Metadata

```python
dataset_df = images_df.merge(
    metadata,
    on="image_id",
    how="left",
    validate="one_to_one",
)

missing_metadata = dataset_df["lesion_id"].isna().sum()

if missing_metadata > 0:
    raise ValueError(
        f"{missing_metadata} images could not be matched with metadata."
    )

label_mismatches = dataset_df[
    dataset_df["folder_label"] != dataset_df["dx"]
]

if not label_mismatches.empty:
    raise ValueError(
        f"Found {len(label_mismatches)} images whose folder labels "
        "do not match the metadata labels."
    )

print(dataset_df.head())
```

> **Important:** HAM10000 may contain multiple images associated with the same `lesion_id`. The lesion-level experiments therefore use `lesion_id` as the grouping variable to prevent images of the same lesion from appearing in both training and validation folds.

### Dataset Citation

When using HAM10000, please cite the original dataset:

```text
Tschandl, P., Rosendahl, C., and Kittler, H. (2018).
The HAM10000 dataset, a large collection of multi-source dermatoscopic
images of common pigmented skin lesions.
Harvard Dataverse.
https://doi.org/10.7910/DVN/DBW86T
```
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
