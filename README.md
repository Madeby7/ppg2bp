# Blood Pressure Estimation from PPG Signals

## Project Overview

This project implements a deep learning approach to estimate blood pressure (Systolic Blood Pressure - SBP and Diastolic Blood Pressure - DBP) from Photoplethysmography (PPG) signals. The workflow is divided into two main stages, each implemented in a dedicated Jupyter notebook.

## Table of Contents

- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1: Data Preparation & Preprocessing](#stage-1-data-preparation--preprocessing)
- [Stage 2: Model Training, Evaluation & Analysis](#stage-2-model-training-evaluation--analysis)
- [Data Organization](#data-organization)
- [File Structure](#file-structure)
- [How to Reproduce](#how-to-reproduce)
- [Experimental Setup, Architecture & Results](#experimental-setup-architecture--results)
- [Notes](#notes)

## Project Structure

```
├── segment_to_cycle_loader.ipynb                           # Stage 1: Data preprocessing
├── three_channel_ppg_peak2peak_subjects.ipynb              # Stage 2: Model training & analysis
├── data/
│   ├── dataframes/                                         # Processed DataFrames from Stage 1
│   └── preprocessing_ready_set/                            # Pre-processed datasets for direct use
├── models/                                                 # Saved model checkpoints
├── results/                                                # Evaluation outputs
└── README.md
```

## Pipeline Overview

1. **Data Preparation & First Stage Preprocessing** → `segment_to_cycle_loader.ipynb`
2. **Second Stage Preprocessing, Model Training, Evaluation & Analysis** → `three_channel_ppg_peak2peak_subjects.ipynb`

---

## Stage 1: Data Preparation & Preprocessing

**Notebook:** `segment_to_cycle_loader.ipynb`

### 1.1. Segment Loading
- **Input:** Segmented PPG/ABP data from `.pkl` files, organized by subject and segment
- **Function:** `load_segments_from_directory`
- **Operations:**
  - Apply length filters (min/max duration)
  - Optimize datatypes for memory efficiency
  - Merge segments into dictionary: `segments_by_subject_merged`

### 1.2. Signal Filtering
- **Purpose:** Remove noise and baseline drift from PPG signals
- **Method:** Butterworth 4th-order bandpass, 0.5–12 Hz
- **Sampling rate:** PPG `fs = 125 Hz`
- **Function:** `apply_filter_to_segments`
- **Output:** Filtered PPG signals replacing original data

### 1.3. Feature Extraction & HRV Analysis
- **Tool:** NeuroKit2 library
- **Features:** Heart Rate Variability (HRV) metrics from PPG signals
- **Storage:** Extended segments with `data` (DataFrame) and `info` (metadata)

### 1.4. Quality Filtering
- **Metric:** Mean `PPG_Quality` score
- **Threshold:** 0.92 (configurable)
- **Action:** Remove segments failing quality check

### 1.5. RR Interval Validation
- **Criteria:** Physiological plausibility of RR intervals
  - Range: 0.33 s ≤ RR ≤ 1.5 s
  - Validity: At least 80% valid intervals
- **Output:** `cleaned_segments_by_subject`

### 1.6. Bottom Detection
- **Purpose:** Identify valley (bottom) indices in PPG waveform between peaks
- **Storage:** Indices stored in segment's `info` dictionary

### 1.7. Visualization
- **Features:** Plot ABP and PPG signals with marked peaks and bottoms
- **Purpose:** Visual inspection and quality assessment

### 1.8. Beat Extraction
- **Function:** `extract_beats_with_raw_and_norm`
- **Process:**
  - Extract individual beats (peak-to-peak windows)
  - Resample PPG windows to fixed length (120 samples)
  - Extract SBP (max) and DBP (min) from corresponding ABP window
  - Store raw ABP waveform (optional)
- **Output Columns:** `ppg_norm_120`, `ppg_raw_120`, `sbp`, `dbp`, `segment_id`, `abp_raw`

### 1.9. Data Persistence
- **Format:** Pickle file in `data/dataframes/` directory
- **Naming:** Encodes number of subjects, segments, and rows

---

## Stage 2: Model Training, Evaluation & Analysis

**Notebook:** `three_channel_ppg_peak2peak_subjects.ipynb`

### 2.1. Data Loading
- Load processed DataFrame from Stage 1
- Support for concatenating multiple DataFrames

### 2.2. Outlier Filtering
- **Method:** Remove rows with mean ABP outside specified confidence interval
- **Purpose:** Reduce impact of outliers on model training

### 2.3. Per-Subject Trimming
- **Target:** Fixed number of windows per subject (e.g., 1000–1001)
- **Purpose:** Ensure balanced representation across subjects

### 2.4. Blood Pressure Categorization
- **Categories:** Normal, Elevated, Stage 1, Stage 2, etc.
- **Method:** Custom classification rules
- **Analysis:** Class balance visualization and statistics

### 2.5. Data Splitting
- **Strategy:** Subject-wise splitting to prevent data leakage
- **Splits:** Train, Validation, Test sets
- **Post-processing:** Trim splits to match target class distribution

### 2.6. Data Preparation for Modeling
- **Structure:**
  - `ppg_train`, `ppg_val`, `ppg_test`: Raw PPG windows
  - `abp_train`, `abp_val`, `abp_test`: [SBP, DBP] pairs
- **Cleaning:** Remove NaN values
- **Randomization:** Shuffle with fixed seeds while maintaining alignment

### 2.7. 3-Channel PPG Representation
- **Channels:**
  - PPG: Original signal
  - VPG: First derivative (Velocity PPG)
  - APG: Second derivative (Acceleration PPG)
- **Output Shape:** `(N, 3, 120)`

### 2.8. Normalization
- **Method:** Z-score normalization
- **Scope:** Each channel normalized independently

### 2.9. PyTorch Integration
- **Dataset:** Custom `PPGABPDataset` class
- **DataLoader:** Efficient batching for training and evaluation

### 2.10. Model Architecture
```python
PPGtoABPRegressor:
├── Input: 3-channel PPG tensor (3, 120)
├── Conv1D layers with BatchNorm and ReLU
├── Dropout for regularization
├── Flatten and Linear layers
└── Output: 2 values (SBP, DBP)
```

### 2.11. Training Configuration
- **Optimizer:** Adam
- **Loss Function:** MAE (L1 Loss)
- **Monitoring:** Training and validation loss tracking

### 2.12. Evaluation Metrics
- **Visualizations:**
  - MAE distribution histograms
  - Bland–Altman plots (MAP, SBP, DBP)
  - Scatter plots (predicted vs. true)
- **Calibration:** Optional global linear calibration for bias correction

### 2.13. Output Management
- **Models:** Saved to `models/` directory
- **Results:** Predictions and evaluations saved to `results/`

---

## Data Organization

### Data Folders

#### `data/dataframes/`
Contains processed DataFrames generated by Stage 1 preprocessing:
- **Format:** `.pkl` files with encoded naming convention
- **Naming Pattern:** `df_subject_{N}_segment_{M}_row_{R}_peak_by_peak_120_sampled.pkl`
  - `N`: Number of unique subjects
  - `M`: Number of segments processed
  - `R`: Total number of beat windows/rows
- **Example:** `df_subject_5_segment_5_row_8721_peak_by_peak_120_sampled.pkl`

#### `data/preprocessing_ready_set/`
Contains pre-processed datasets ready for immediate use in Stage 2 (the results mentioned dataframe for the validation proceed from that dataset):
- **Purpose:** Skip time-intensive Stage 1 preprocessing
- **Not Available Dataset:** `df_137_sampled_peak_by_peak_rows_fixed_1000.pkl`
  - 137 subjects with balanced sampling
  - Fixed 1000 beat windows per subject
  - **Please contact with author for using that dataset** 
- **Usage:** Load directly in Stage 2 notebook for model training


---

## File Structure

```
├── segment_to_cycle_loader.ipynb                           # Stage 1: Data preprocessing
├── three_channel_ppg_peak2peak_subjects.ipynb  # Stage 2: Model training & analysis
├── data/
│   ├── dataframes/                                         # Generated by Stage 1
│   │   └── df_subject_5_segment_5_row_8721_peak_by_peak_120_sampled.pkl
│   └── preprocessing_ready_set/                            # Pre-processed datasets
│       └─ about.txt                                       # Dataset information
|
├── models/                                                 # PyTorch model checkpoints
├── results/                                                # Evaluation outputs and predictions
└── README.md
```

### Note about Raw Data Structure
The segmentation and preprocessing pipeline expects local subject dataframes under `saved_subjects_30/31/32` folders (not included in this export). To keep the repo lightweight, the first stage of data preparation is illustrated with toy segments chosen from `saved_subjects_32` (see [toy_segments/](toy_segments/)). It is possible to use the pre-processed dataset in [data/dataframes/](data/dataframes/) to start training/validation quickly, but instead of running on only 5 segmented subjects, we recommend running the full [segment_to_cycle_loader.ipynb](segment_to_cycle_loader.ipynb) pipeline, which produces a **19-subject dataframe for downstream use** and gives a complete picture of how the dataframes are constructed.

## How to Reproduce

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Required packages (please consider requirements.txt)

### Steps

#### Quick Start (Using Pre-processed Toy Dataframe) - Not Recommended
1. **Load Pre-processed Dataset**
   ```python
   import pandas as pd
   df = pd.read_pickle('data/dataframes/df_subject_5_segment_5_row_8721_peak_by_peak_rows_120_sampled.pkl')
   ```

2. **Run Stage 2 - Model Training & Analysis**
   ```bash
   jupyter notebook three_channel_ppg_peak2peak_subjects.ipynb
   ```

#### Full Pipeline (For Custom Preprocessing)
1. **Prepare Raw Data**
   - Place raw segment `.pkl` files in appropriate directories

2. **Run Stage 1 - Data Preprocessing**
   ```bash
   jupyter notebook segment_to_cycle_loader.ipynb
   ```
   - Follow notebook cells sequentially
   - Outputs processed DataFrame to `data/dataframes/`

3. **Run Stage 2 - Model Training & Analysis**
   ```bash
   jupyter notebook three_channel_ppg_peak2peak_subjects.ipynb
   ```
   - Load processed data from Stage 1
   - Train model and generate evaluations

4. **Review Outputs**
   - Check `models/` for saved model checkpoints
   - Review `results/` for predictions and analysis

---

## Experimental Setup, Architecture & Results

The numbers below were obtained by training the `PPGtoABPRegressor` on the 19-subject peak-to-peak dataset and evaluating on the held-out test split. They correspond to the saved checkpoint [models/ppg_to_abp_cnn_subject_19_segment_19_fixed_row_1000.pth](models/ppg_to_abp_cnn_subject_19_segment_19_fixed_row_1000.pth) and the predictions pickle [results/ppg_abp_predictions_peak_to_peak_raw_19_subject_19_segment_subject_analyze.pkl](results/ppg_abp_predictions_peak_to_peak_raw_19_subject_19_segment_subject_analyze.pkl).

### Experimental Setup

| Aspect | Value |
|---|---|
| Model | `PPGtoABPRegressor` (5 conv blocks + 1×1 conv + FC) |
| Input | `(128, 3, 120)` — (batch, channels, samples) |
| Output | 2 neurons — `[SBP, DBP]` mmHg |
| Batch size | 128 |
| Learning rate | 1e-4 |
| Epochs | 60 |
| Loss | L1 (MAE) |
| Train / Val / Test | 86,584 / 11,697 / 21,636 |
| Subjects | 19 |
| Normalization | Z-score per channel (PPG only) |
| Test MAE (pre-calib) | 16.60 mmHg (SBP) |
| Test MAE (post-calib) | 15.86 mmHg (SBP) |
| Calibration | `SBP_cal = 0.5840 · SBP_pred + 51.92` |
| PPG sampling rate | 125 Hz |
| Filter | Butterworth 4th order, 0.5–12 Hz |
| Quality threshold | 0.92 |
| RR range | 0.33–1.5 s |

### Model Architecture

Input shape: `(batch, 3, 120)` &nbsp;·&nbsp; Output: 2 values `[SBP, DBP]` in mmHg.

| # | Layer | In → Out ch | Kernel | Pad | Extras |
|---|---|---|---|---|---|
| 1 | Conv1d | 3 → 16 | 15 | 7 | BN, ReLU |
| 2 | Conv1d | 16 → 32 | 15 | 7 | BN, ReLU, Dropout 0.2 |
| 3 | Conv1d | 32 → 64 | 15 | 7 | BN, ReLU |
| 4 | Conv1d | 64 → 32 | 15 | 7 | BN, ReLU, Dropout 0.4 |
| 5 | Conv1d | 32 → 16 | 15 | 7 | BN, ReLU |
| 6 | Conv1d | 16 → 1  | 1  | 0 | channel collapse |
| 7 | Flatten + Linear (120 → 2) | — | — | — | → `[SBP, DBP]` |

Padding = 7 with kernel = 15 preserves the temporal length, so the Linear head sees exactly `input_length = 120`.

### Results

Test set, SBP (n = 21,636 windows):

| Metric | Pre-calibration | Post-calibration |
|---|---|---|
| Bias (mean error) | −0.43 mmHg | ~0 mmHg |
| SD of error | 21.16 mmHg | 19.81 mmHg |
| **MAE** | **16.60 mmHg** | **15.86 mmHg** |
| n (windows) | 21,636 | 21,636 |

Global linear calibration fit on the test set:

```
SBP_cal = 0.5840 · SBP_pred + 51.92
```

### Reproducing these results

1. Load the pre-processed 137-subject / 1000-rows-per-subject dataframe (request access — see [data/preprocessing_ready_set/about.txt](data/preprocessing_ready_set/about.txt)) **or** regenerate it via [segment_to_cycle_loader.ipynb](segment_to_cycle_loader.ipynb) on the 19-subject set.
2. Open [three_channel_ppg_peak2peak_subjects.ipynb](three_channel_ppg_peak2peak_subjects.ipynb) and run the cells end-to-end with the configuration above (batch 128, LR 1e-4, 60 epochs, L1 loss).
3. The notebook saves the checkpoint under [models/](models/) and the per-window predictions under [results/](results/); calibration and the Bland–Altman / scatter plots are produced in the evaluation cells.

---

## Notes

### Important Considerations
- All processing steps maintain **subject-wise separation** to prevent data leakage
- Pipeline is **modular** - parameters can be adjusted for different datasets
- **Reproducibility** ensured through fixed random seeds
- Memory optimization techniques used for large datasets

### Configuration Options
- Quality thresholds can be adjusted based on dataset characteristics
- Window lengths and model architecture are configurable
- BP categorization rules can be customized for different clinical standards

### Performance Tips
- Use CUDA-enabled GPU for faster training
- Adjust batch sizes based on available memory
- Consider data augmentation for smaller datasets
- **Recommended:** Use pre-processed data from `preprocessing_ready_set/` for faster iteration, then retrain the model architecture to reproduce the numbers reported in the [Experimental Setup, Architecture & Results](#experimental-setup-architecture--results) section.

### Data Validation
- Pre-processed datasets in `preprocessing_ready_set/` have been validated
- Detailed results and validation metrics are reported in the evaluation part of [three_channel_ppg_peak2peak_subjects.ipynb](three_channel_ppg_peak2peak_subjects.ipynb)
- Custom processed data should be validated against known benchmarks

---
