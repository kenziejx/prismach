
# %%
import pandas as pd
import matplotlib.pyplot as plt
import polars
import openpyxl
from ema_workbench.analysis import prim

# %% Load data
global_data = polars.read_excel(
    '_Scenario_Compass_Initiative_Data/SCI-2025_v1.0_pathways_ensemble_global.xlsx',
    sheet_name='data'
)

# %% Set multi-index: each (Model, Scenario) pair = one model run
pandas_df = global_data.to_pandas()
pandas_df.set_index(['Model', 'Scenario'], inplace=True)

# Year columns available in the dataset (all numeric-named columns)
all_year_cols = [c for c in pandas_df.columns if str(c).isdigit()]

# %% --- Configuration ---
# Select which years to use as predictors
predictor_years = ['2050']
# Select which years to use as the outcome (single year)
outcome_year = '2070'
# Variable filter for predictors (str.contains match)
predictor_filter = 'Primary Energy|'
# Exact variable name for the outcome
outcome_variable = 'Climate Assessment|Harmonized|Emissions|CO2|Energy and Industrial Processes'

# %% --- Build X (predictors) ---
# Filter rows whose Variable contains the predictor string
inputs_filtered = pandas_df[
    pandas_df['Variable'].str.contains(predictor_filter, case=False, regex=False, na=False)
]

# Keep only the year columns we want plus the Variable label
inputs_cols = [c for c in pandas_df.columns if str(c) in predictor_years]
inputs_subset = inputs_filtered[['Variable'] + inputs_cols]

# Pivot to wide format: one row per (Model, Scenario), one col per (year, Variable)
# Result columns are a MultiIndex: (year, Variable_name)
X_df = inputs_subset.reset_index().pivot_table(
    index=['Model', 'Scenario'],
    columns='Variable',
    values=inputs_cols
)
# Flatten column MultiIndex to "year|Variable" strings for readability
X_df.columns = [f"{year}|{var}" for year, var in X_df.columns]

# %% --- Build Y (outcome) ---
outputs_filtered = pandas_df[pandas_df['Variable'] == outcome_variable]
Y_series = outputs_filtered[outcome_year]  # Series indexed by (Model, Scenario)
Y_series.name = outcome_variable

# %% --- Align X and Y on common model runs ---
common_index = X_df.index.intersection(Y_series.index)
# Align on common runs, remove columns that are all zero, drop rows with NaNs, then align Y
X_df = X_df.loc[common_index]
# %%
# Drop columns where every value is exactly 0
X_df = X_df.loc[:, ~(X_df == 0).all(axis=0)]
# Drop columns where every value is NaN
X_df = X_df.dropna(axis=1, how='all')
Y_series = Y_series.loc[X_df.index]

print(f"Model runs (rows): {len(X_df)}")
print(f"Predictor features (cols): {X_df.shape[1]}")
print(f"X shape: {X_df.shape}, Y shape: {Y_series.shape}")
print(f"\nX columns (first 5): {list(X_df.columns[:5])}")

# %% --- Convert to arrays for EMA workbench ---
X = X_df.values          # 2D numpy array: (n_runs, n_features)
y = Y_series.values       # 1D numpy array: (n_runs,)

# %% --- PRIM analysis ---
# threshold and threshold_type depend on your outcome variable's units/scale
threshold = y.mean()      # example: split above/below mean
y_prim = (y > threshold).astype(bool)  # binary outcome for PRIM: 1 if above threshold, else 0
prim_alg = prim.Prim(X_df, y_prim)

box1 = prim_alg.find_box()
box1.show_tradeoff()
plt.show()

# %%
