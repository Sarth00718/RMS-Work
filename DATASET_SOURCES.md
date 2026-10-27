# Kaggle dataset sources

This document tracks the datasets used in this repository, their Kaggle sources, local directory locations, and key specifications.

---

## 1. APTOS 2019 blindness detection

- **Kaggle competition URL:** https://www.kaggle.com/c/aptos2019-blindness-detection
- **Provider:** Asia Pacific Tele-Ophthalmology Society (APTOS) & Aravind Eye Hospital, India
- **Local repository paths:**
  - `aptos_dataset/` (contains `train_images/`, `val_images/`, `test_images/`, `train_1.csv`, `valid.csv`, `test.csv`)
  - `archive.zip` (8.6 GB archive containing full image splits and test metadata)
  - `project_aptos/` (standalone reproduction pipeline and evaluation scripts)
- **Data type:** Retinal fundus photographs (color PNG images)
- **Total sample count:** 3,662 labeled images partitioned into:
  - Train: 2,930 images (80%)
  - Validation: 366 images (10%)
  - Test: 366 images (10%)
- **Target classes:** 5-stage International Clinical Diabetic Retinopathy (ICDR) scale
  - Grade 0: No DR
  - Grade 1: Mild
  - Grade 2: Moderate
  - Grade 3: Severe
  - Grade 4: Proliferative DR
- **Binary framing:** Grade 0 (No DR) vs. Grades 1--4 (Referable DR)

---

## 2. IDRiD (Indian diabetic retinopathy image dataset)

- **Kaggle dataset URLs:**
  - https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset
  - https://www.kaggle.com/datasets/prashant268/idrid-dataset
- **Primary source:** IEEE DataPort (DOI: 10.21227/H25W98)
- **Contributors:** Prasanna Porwal, Samiksha Pachade, Manesh Kokare (Eye clinic, Nanded, Maharashtra, India)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Local repository paths:**
  - `B. Disease Grading/` (unpacked directory containing `1. Original Images/` and `2. Groundtruths/`)
  - `B. Disease Grading.zip` (198 MB original archive)
  - `project_idrid/` (standalone pipeline for IDRiD federated training)
- **Data type:** Retinal fundus photographs (.jpg, centered on macula and optic disc)
- **Total sample count:** 516 images
  - Training set: 413 images (`a. IDRiD_Disease Grading_Training Labels.csv`)
  - Testing set: 103 images (`b. IDRiD_Disease Grading_Testing Labels.csv`)
- **Target labels:**
  - Retinopathy grade: 0 to 4 (ICDR scale)
  - Risk of macular edema: 0 to 2

---

## 3. Messidor diabetic retinopathy features

- **Kaggle dataset URLs:**
  - https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen
- **Original source:** UCI Machine Learning Repository (Messidor Diabetic Retinopathy features extracted by Antal and Hajdu)
- **Local repository paths:**
  - `project_messidor/messidor_features.csv`
  - `project_messidor/` (NumPy/scikit-learn federated learning pipeline and analytical Rényi-DP accountant)
- **Data type:** Tabular clinical records (engineered biomarkers extracted from retinal images)
- **Sample count:** 1,151 samples (1,146 unique records after dropping 5 duplicate rows)
  - Train: 802 records (70%)
  - Validation: 172 records (15%)
  - Test: 172 records (15%)
- **Feature dimensionality:** 19 raw numeric attributes expanded to 28 features (microaneurysm counts, exudate severity, macula center distance, optic disc diameter, AM/FM classification)
- **Target:** Binary risk classification (`Class` 0 = no signs of DR, 1 = signs of DR)

---

## Quick command line access

If you have the Kaggle API installed (`pip install kaggle`) and your `kaggle.json` credentials in `~/.kaggle/`, you can download or verify the datasets with:

```bash
# APTOS 2019 competition dataset
kaggle competitions download -c aptos2019-blindness-detection

# IDRiD dataset mirror
kaggle datasets download -d mariaherrerot/idrid-dataset

# Messidor features dataset
kaggle datasets download -d abdallahwagih/messidor-features
```
