# Parking Lot Occupancy from Satellite Imagery

Segment parking spaces in aerial/satellite images and track occupancy over time, using Walmart locations as a real-world test case.

---

## Project Flow

```
 1. Coordinates          2. Image Download       3. EDA
 walmart_coordenates  →  parking_image_         →  eda_pklot
 .ipynb                  colection.ipynb            .ipynb
      │                        │                        │
      │                        ▼                        ▼
      │                 Data/walmart_test/       Data/raw/ (PKLot)
      │
      ▼
 4. Exploratory Model                    5. Inference
 exploratory_model.ipynb        →        inference.ipynb
 trains ResNet34 U-Net                   runs on walmart_test
 on PKLot dataset                        images, tracks occupancy
      │
      ▼
 checkpoints/resnet34_unet_best.pth

 6. Fine-Tuning Model
 fine_tuning_model.ipynb
 loads resnet34_unet_best.pth  →  fine-tunes on Data/walmart-1/
 tests on walmart_test satellite images
 saves checkpoints/resnet34_unet_walmart_ft.pth
```

---

## Notebooks

### `notebooks/data_collection/`

| Notebook | What it does |
|---|---|
| `walmart_coordenates.ipynb` | Loads the Walmart US store CSV (~4 500 stores), samples 2 stores per state, and manually pins the **parking lot** coordinates (not store entrance). Exports `Data/walmart_coordinates.csv`. |
| `parking_image_colection.ipynb` | Uses **Google Earth Engine** (NAIP imagery) to download satellite images for a selected Walmart location across multiple years (2010–2022). These are the **exploratory** images used to test the model — not labelled training data. Saves to `Data/walmart_test/`. |
| `download_fine_tuning.ipynb` | Downloads the labelled Walmart parking dataset from Roboflow (`Data/walmart-1/`), which will be used for fine-tuning. |

### `notebooks/eda/`

| Notebook | What it does |
|---|---|
| `eda_pklot.ipynb` | Explores the PKLot dataset: class balance (empty vs occupied), temporal coverage (Sept 2012 – Apr 2013), hourly occupancy patterns, bounding box size distributions, and sample image visualisations. |

### `notebooks/model/`

| Notebook | What it does |
|---|---|
| `exploratory_model.ipynb` | Trains a **ResNet34 U-Net** on the PKLot dataset. Converts COCO bounding-box annotations into dense pixel masks at load time. Runs 30 epochs with early stopping. Best checkpoint saved to `checkpoints/resnet34_unet_best.pth`. |
| `fine_tuning_model.ipynb` | Loads `resnet34_unet_best.pth` and fine-tunes on the **walmart-1** Roboflow dataset (YOLO format). Two-phase strategy: encoder frozen first, then full unfreeze with differential LR. Includes a side-by-side comparison of the exploratory vs fine-tuned model on the satellite test images. Saves `checkpoints/resnet34_unet_walmart_ft.pth`. |

### `notebooks/inference.ipynb`

Loads `resnet34_unet_best.pth` and runs it on the Walmart satellite images downloaded in step 2. Outputs per-image segmentation overlays and an **occupancy rate trend** across years. Also includes Test-Time Augmentation (TTA) experiments (single-angle and combined) to improve predictions on real-world imagery.

---

## Datasets

| Path | Format | Description |
|---|---|---|
| `Data/raw/` | COCO JSON | PKLot dataset — 8 691 train / 2 483 val / 1 242 test images, ~711k annotations |
| `Data/walmart_test/` | PNG | Satellite images of one Walmart parking lot, 2010–2022 (exploratory, unlabelled) |
| `Data/walmart-1/` | YOLO | Labelled Walmart parking lot images from Roboflow — classes: `car`, `free_space` |

---

## Model

**Architecture:** ResNet-34 encoder + U-Net decoder (via `segmentation-models-pytorch`)

| Setting | Value |
|---|---|
| Input | 256 × 256 RGB, ImageNet-normalised |
| Output | 3-class pixel mask: `background` / `space-empty` / `space-occupied` |
| Loss | CrossEntropyLoss |
| Metric | Mean IoU |
| Best val mIoU | 0.921 |
| Test mIoU | 0.906 (pixel acc 97.5%) |

Training runs on a 5% subsample of PKLot for speed; set `SUBSET_FRACTION = 1.0` in the notebook for full training.

---

## Fine-Tuning Details (`fine_tuning_model.ipynb`)

The exploratory model was trained on **PKLot** (university parking lots, fixed camera). Fine-tuning on `walmart-1` adapts it to Walmart-specific parking layouts.

**Class mapping** from YOLO to the 3-class model head:

| walmart-1 YOLO | id | → model class | id |
|---|---|---|---|
| `car` | 0 | `space-occupied` | 2 |
| `free_space` | 1 | `space-empty` | 1 |

**Two-phase training** (small dataset = 40 images, so catastrophic forgetting is a real risk):

| Phase | Encoder | Decoder/Head | LR |
|---|---|---|---|
| 1 (20 epochs) | Frozen | Training | `1e-3` |
| 2 (15 epochs) | Unfrozen | Training | `5e-6` / `5e-5` (differential) |

**Output checkpoint:** `checkpoints/resnet34_unet_walmart_ft.pth`

---

## Setup

```bash
uv sync          # install dependencies from pyproject.toml / uv.lock
```

Requires a Google Earth Engine project to run `parking_image_colection.ipynb`. Authenticate with:

```bash
earthengine authenticate
```
