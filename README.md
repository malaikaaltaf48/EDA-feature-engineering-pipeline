# EDA & Feature Engineering Pipeline

This is Project 1. The goal of this project was to take raw, messy data and transform it into a clean, mathematically sound dataset ready for machine learning.

## What This Project Does

I used the Kaggle House Prices dataset, which had a lot of missing values, several extreme outliers, and wasn't directly usable for ML models. I built a full pipeline to handle all of that — from fixing missing values to engineering new features.

## Dataset
[House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## Steps I Followed

### 1. Handled Missing Values
First, I checked which columns had missing values and how much. For some columns like `PoolQC` or `GarageType`, a missing value didn't actually mean the data was missing — it meant that feature simply didn't exist for that house (e.g., no pool). So I filled those with "None" instead of dropping them. For `LotFrontage`, I used group-wise median imputation (grouped by `Neighborhood`), since homes in the same area tend to have similar lot sizes.

### 2. Detected & Treated Outliers
I used the IQR method to detect outliers (`Q1 - 1.5*IQR` to `Q3 + 1.5*IQR`). Instead of deleting rows with outliers, I used `numpy.clip()` to cap them at the boundary values — this keeps the dataset volume intact while still neutralizing the effect of extreme values.

### 3. Engineered New Features
I created 5 new predictive features:
- `TotalSF` — combined basement + 1st floor + 2nd floor square footage
- `HouseAge` — age of the house at time of sale
- `TotalBathrooms` — weighted combination of full and half bathrooms
- `RemodAge` — years since the last remodel
- `IsNew` — binary flag for newly built homes

### 4. Encoded Categorical Columns
I used One-Hot Encoding instead of Label Encoding, since Label Encoding introduces a false mathematical distance between categories (e.g., if "Tokyo" is assigned 3 and "London" is assigned 1, the model would incorrectly interpret Tokyo as "3x more" than London).

### 5. Removed Multicollinearity
I built a correlation matrix to find feature pairs with correlation above 0.80. For each pair, I dropped the feature that correlated less strongly with `SalePrice`, to remove redundant information from the dataset.

## What I Learned
Data cleaning isn't just about removing junk — every decision (like filling vs. dropping a missing value) needs statistical reasoning behind it. The cleaner and more meaningful the data, the better a model can learn from it.

## Tools Used
Pandas, NumPy, Matplotlib, Seaborn, SciPy
