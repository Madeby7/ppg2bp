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
- [Dependencies](#dependencies)
- [Notes](#notes)

## Project Structure

```
├── segment_to_cycle_loader.ipynb                         # Stage 1: Data preprocessing
├── three_channel_ppg_peak2peak_subjects.ipynb  # Stage 2: Model training & analysis
├── data/
│   ├── dataframes/                                         # Processed DataFrames from Stage 1
│   └── preprocessing_ready_set/                            # Pre-processed datasets for direct use
├── models/                                                 # Saved model checkpoints
├── results/                                                # Evaluation outputs
└── README.md
```

## Pipeline Overview

1. **Data Preparation & First Stage Preprocessing** → `segment_to_cycle_loader.ipynb`
2. **Second Stage Preprocessing, Model Training, Evaluation & Analysis** → 
`three_channel_ppg_peak2peak_subjects.ipynb`

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
- **Method:** Bandpass filter application
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
  - Range: 0.4s ≤ RR ≤ 1.5s
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
Contains pre-processed datasets ready for immediate use in Stage 2:
- **Purpose:** Skip time-intensive Stage 1 preprocessing
- **Available Dataset:** `df_137_sampled_peak_by_peak_rows_fixed_1000.pkl`
  - 137 subjects with balanced sampling
  - Fixed 1000 beat windows per subject
  - Results validation available through collaborators
- **Usage:** Load directly in Stage 2 notebook for model training

### Data Access Options

**Option 1: Full Pipeline (Recommended for customization)**
```python
# Run complete preprocessing pipeline
jupyter notebook segment_to_cycle_loader.ipynb
# Then proceed to model training
jupyter notebook model_checkpoint_3_channel_peak_to_peak_loaded_data_subject_analze.ipynb
```

**Option 2: Use Pre-processed Data (Recommended for quick start)**
```python
# Load pre-processed dataset directly in Stage 2
df = pd.read_pickle('data/preprocessing_ready_set/df_137_sampled_peak_by_peak_rows_fixed_1000.pkl')
```

---

## File Structure

```
├── segment_to_cycle_loader.ipynb                           # Stage 1: Data preprocessing
├── model_checkpoint_3_channel_peak_to_peak_loaded_data_subject_analze.ipynb  # Stage 2: Model training & analysis
├── data/
│   ├── dataframes/                                         # Generated by Stage 1
│   │   └── df_subject_5_segment_5_row_8721_peak_by_peak_120_sampled.pkl
│   └── preprocessing_ready_set/                            # Pre-processed datasets
│       ├── about.txt                                       # Dataset information
│       └── df_137_sampled_peak_by_peak_rows_fixed_1000.pkl # Ready-to-use dataset
├── models/                                                 # PyTorch model checkpoints
├── results/                                                # Evaluation outputs and predictions
└── README.md
```

### Note about Raw Data Structure 
The segmentation and preprocessing pipeline considers local subject dataframes under `saved_subjects_30/31/32` folders (not included in this export). Due to the extensive data processing requirements, it's recommended to use the pre-processed dataset in `data/preprocessing_ready_set/` instead of running the complete `segment_to_cycle_loader.ipynb` pipeline, which can take several hours.

## How to Reproduce

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Required packages (see Dependencies)

### Steps

#### Quick Start (Using Pre-processed Data)
1. **Load Pre-processed Dataset**
   ```python
   import pandas as pd
   df = pd.read_pickle('data/preprocessing_ready_set/df_137_sampled_peak_by_peak_rows_fixed_1000.pkl')
   ```

2. **Run Stage 2 - Model Training & Analysis**
   ```bash
   jupyter notebook model_checkpoint_3_channel_peak_to_peak_loaded_data_subject_analze.ipynb
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
   jupyter notebook model_checkpoint_3_channel_peak_to_peak_loaded_data_subject_analze.ipynb
   ```
   - Load processed data from Stage 1
   - Train model and generate evaluations

4. **Review Outputs**
   - Check `models/` for saved model checkpoints
   - Review `results/` for predictions and analysis


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
- **Recommended:** Use pre-processed data from `preprocessing_ready_set/` for faster iteration

### Data Validation
- Pre-processed datasets in `preprocessing_ready_set/` have been validated
- Contact collaborators for detailed results and validation metrics
- Custom processed data should be validated against known benchmarks

---