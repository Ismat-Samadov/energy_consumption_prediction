# Auto-generated from energy_prediction_analysis.ipynb

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import kagglehub

# Sklearn - Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Sklearn - Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge

# Sklearn - Tree Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                               GradientBoostingRegressor, AdaBoostRegressor,
                               VotingRegressor, StackingRegressor, BaggingRegressor)

# Sklearn - Other Models
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Sklearn - Metrics
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              mean_absolute_percentage_error, explained_variance_score)

# Boosting Libraries
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available, skipping...")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, skipping...")

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Time Series
from prophet import Prophet

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create charts directory
import os
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CHARTS_DIR = REPO_ROOT / 'charts'
OUTPUTS_DIR = REPO_ROOT / 'outputs'

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep existing relative paths working after moving this file into scripts/
os.chdir(REPO_ROOT)

print("All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"XGBoost version: {xgb.__version__}")

print("=" * 80)
print("AUTOMATIC DATASET DOWNLOAD FROM KAGGLE")
print("=" * 80)
print("\nDataset: mrsimple07/energy-consumption-prediction")
print("Source: https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction")
print("\nDownloading...")

# Download latest version from Kaggle
dataset_path = kagglehub.dataset_download("mrsimple07/energy-consumption-prediction")

print(f"\n✓ Dataset downloaded successfully!")
print(f"✓ Path: {dataset_path}")

# Find the CSV file in the downloaded path
import glob
csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

if csv_files:
    DATA_FILE = csv_files[0]
    print(f"✓ Found CSV file: {os.path.basename(DATA_FILE)}")
    print(f"✓ Full path: {DATA_FILE}")
else:
    raise FileNotFoundError("No CSV file found in the downloaded dataset!")

print("\n" + "=" * 80)
print("DOWNLOAD COMPLETE - Ready to load data!")
print("=" * 80)

# Load the dataset
df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded from: {os.path.basename(DATA_FILE)}")
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst few rows:")
df.head(10)

# Statistical Summary
print("Statistical Summary:")
df.describe()

# Check for missing values
print("Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Data types
print(f"\nData types:")
print(df.dtypes)

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year
df['DayOfYear'] = df['Timestamp'].dt.dayofyear
df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week

print("Time-based features extracted!")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
df.head()

# Energy Consumption over time
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df['Timestamp'], df['EnergyConsumption'], linewidth=1, alpha=0.7, color='#2E86AB')
ax.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy Consumption (kWh)', fontsize=12, fontweight='bold')
ax.set_title('Energy Consumption Over Time', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/01_energy_consumption_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/01_energy_consumption_timeseries.png")

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
numerical_cols = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'EnergyConsumption']
colors = ['#E76F51', '#F4A261', '#E9C46A', '#2A9D8F', '#264653', '#E63946']

for idx, (col, color) in enumerate(zip(numerical_cols, colors)):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[col], bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel(col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/02_feature_distributions.png")

# Correlation heatmap
plt.figure(figsize=(14, 10))
correlation_cols = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
                    'RenewableEnergy', 'Hour', 'Month', 'EnergyConsumption']
correlation_matrix = df[correlation_cols].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1,
            annot_kws={'size': 9, 'weight': 'bold'})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('charts/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/03_correlation_heatmap.png")

# Energy consumption patterns
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Hourly pattern
hourly_avg = df.groupby('Hour')['EnergyConsumption'].mean()
axes[0, 0].bar(hourly_avg.index, hourly_avg.values, color='#06A77D', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Avg Energy Consumption', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Energy Consumption by Hour', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Monthly pattern
monthly_avg = df.groupby('Month')['EnergyConsumption'].mean()
axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='#F18F01', alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Month', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Avg Energy Consumption', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Energy Consumption by Month', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Day of week pattern
day_avg = df.groupby('DayOfWeek')['EnergyConsumption'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
axes[1, 0].bar(range(len(day_avg)), day_avg.values, color='#A23B72', alpha=0.8, edgecolor='black')
axes[1, 0].set_xticks(range(len(day_avg)))
axes[1, 0].set_xticklabels(day_avg.index, rotation=45)
axes[1, 0].set_xlabel('Day of Week', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Avg Energy Consumption', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Energy Consumption by Day of Week', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Holiday vs Non-Holiday
holiday_avg = df.groupby('Holiday')['EnergyConsumption'].mean()
axes[1, 1].bar(holiday_avg.index, holiday_avg.values, color=['#E63946', '#457B9D'], 
               alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Holiday', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Avg Energy Consumption', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Energy Consumption: Holiday vs Non-Holiday', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('charts/04_consumption_patterns.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/04_consumption_patterns.png")

# Scatter plots - relationships
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

scatter_pairs = [
    ('Temperature', 'EnergyConsumption'),
    ('Humidity', 'EnergyConsumption'),
    ('SquareFootage', 'EnergyConsumption'),
    ('Occupancy', 'EnergyConsumption'),
    ('RenewableEnergy', 'EnergyConsumption'),
    ('Hour', 'EnergyConsumption')
]

colors_scatter = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight']

for idx, ((x_col, y_col), cmap) in enumerate(zip(scatter_pairs, colors_scatter)):
    ax = axes[idx // 3, idx % 3]
    scatter = ax.scatter(df[x_col], df[y_col], c=df['Hour'], 
                         cmap=cmap, alpha=0.5, s=10)
    ax.set_xlabel(x_col, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=10, fontweight='bold')
    ax.set_title(f'{x_col} vs {y_col}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Hour')

plt.tight_layout()
plt.savefig('charts/05_scatter_relationships.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/05_scatter_relationships.png")

# Box plots for categorical features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# HVAC Usage
df.boxplot(column='EnergyConsumption', by='HVACUsage', ax=axes[0])
axes[0].set_xlabel('HVAC Usage', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Energy Consumption', fontsize=11, fontweight='bold')
axes[0].set_title('Energy Consumption by HVAC Usage', fontsize=12, fontweight='bold')
axes[0].get_figure().suptitle('')

# Lighting Usage
df.boxplot(column='EnergyConsumption', by='LightingUsage', ax=axes[1])
axes[1].set_xlabel('Lighting Usage', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Energy Consumption', fontsize=11, fontweight='bold')
axes[1].set_title('Energy Consumption by Lighting Usage', fontsize=12, fontweight='bold')

# Holiday
df.boxplot(column='EnergyConsumption', by='Holiday', ax=axes[2])
axes[2].set_xlabel('Holiday', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Energy Consumption', fontsize=11, fontweight='bold')
axes[2].set_title('Energy Consumption by Holiday', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/06_categorical_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/06_categorical_boxplots.png")

# Encode categorical variables
le_hvac = LabelEncoder()
le_lighting = LabelEncoder()
le_day = LabelEncoder()
le_holiday = LabelEncoder()

df['HVACUsage_encoded'] = le_hvac.fit_transform(df['HVACUsage'])
df['LightingUsage_encoded'] = le_lighting.fit_transform(df['LightingUsage'])
df['DayOfWeek_encoded'] = le_day.fit_transform(df['DayOfWeek'])
df['Holiday_encoded'] = le_holiday.fit_transform(df['Holiday'])

# Cyclical encoding for time features
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek_encoded'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek_encoded'] / 7)

# Interaction features
df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
df['Temp_Squared'] = df['Temperature'] ** 2
df['HVAC_Lighting'] = df['HVACUsage_encoded'] * df['LightingUsage_encoded']
df['SquareFootage_Occupancy'] = df['SquareFootage'] * df['Occupancy']

# Lag features
df['EnergyConsumption_lag1'] = df['EnergyConsumption'].shift(1)
df['EnergyConsumption_lag24'] = df['EnergyConsumption'].shift(24)
df['EnergyConsumption_lag168'] = df['EnergyConsumption'].shift(168)  # 1 week

# Rolling statistics
df['EnergyConsumption_rolling_mean_24h'] = df['EnergyConsumption'].rolling(window=24, min_periods=1).mean()
df['EnergyConsumption_rolling_std_24h'] = df['EnergyConsumption'].rolling(window=24, min_periods=1).std()
df['EnergyConsumption_rolling_max_24h'] = df['EnergyConsumption'].rolling(window=24, min_periods=1).max()
df['EnergyConsumption_rolling_min_24h'] = df['EnergyConsumption'].rolling(window=24, min_periods=1).min()

# Fill NaN values
df.fillna(method='bfill', inplace=True)

print("Feature engineering completed!")
print(f"Total features: {df.shape[1]}")
print("\nNew features created:")
new_features = [col for col in df.columns if col not in ['Timestamp', 'HVACUsage', 'LightingUsage', 
                                                           'DayOfWeek', 'Holiday', 'EnergyConsumption']]
print(new_features)

# Define feature sets
feature_cols = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy',
    'HVACUsage_encoded', 'LightingUsage_encoded', 'Holiday_encoded',
    'Hour', 'Month', 'DayOfYear', 'WeekOfYear',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'Temp_Humidity', 'Temp_Squared', 'HVAC_Lighting', 'SquareFootage_Occupancy',
    'EnergyConsumption_lag1', 'EnergyConsumption_lag24', 'EnergyConsumption_lag168',
    'EnergyConsumption_rolling_mean_24h', 'EnergyConsumption_rolling_std_24h',
    'EnergyConsumption_rolling_max_24h', 'EnergyConsumption_rolling_min_24h'
]

X = df[feature_cols].values
y = df['EnergyConsumption'].values

# Time-based split (80-20)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
df_train, df_test = df[:split_idx], df[split_idx:]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {len(feature_cols)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete!")

# Helper function for evaluation
def evaluate_model(name, y_true, y_pred, verbose=True):
    """Calculate comprehensive metrics for model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    evs = explained_variance_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'EVS': evs
    }
    
    if verbose:
        print(f"\n{name} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return metrics

# Dictionary to store all results
models_results = {}

print("Helper functions ready!")

# Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
models_results['Linear Regression'] = evaluate_model('Linear Regression', y_test, lr_pred)
models_results['Linear Regression']['predictions'] = lr_pred

# Ridge Regression
print("Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)
models_results['Ridge'] = evaluate_model('Ridge', y_test, ridge_pred)
models_results['Ridge']['predictions'] = ridge_pred

# Lasso Regression
print("Training Lasso Regression...")
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)
models_results['Lasso'] = evaluate_model('Lasso', y_test, lasso_pred)
models_results['Lasso']['predictions'] = lasso_pred

# ElasticNet
print("Training ElasticNet...")
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model.fit(X_train_scaled, y_train)
elastic_pred = elastic_model.predict(X_test_scaled)
models_results['ElasticNet'] = evaluate_model('ElasticNet', y_test, elastic_pred)
models_results['ElasticNet']['predictions'] = elastic_pred

# Bayesian Ridge
print("Training Bayesian Ridge...")
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train_scaled, y_train)
bayesian_pred = bayesian_model.predict(X_test_scaled)
models_results['Bayesian Ridge'] = evaluate_model('Bayesian Ridge', y_test, bayesian_pred)
models_results['Bayesian Ridge']['predictions'] = bayesian_pred

# Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
models_results['Decision Tree'] = evaluate_model('Decision Tree', y_test, dt_pred)
models_results['Decision Tree']['predictions'] = dt_pred

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, 
                                  min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
models_results['Random Forest'] = evaluate_model('Random Forest', y_test, rf_pred)
models_results['Random Forest']['predictions'] = rf_pred

# Extra Trees
print("Training Extra Trees...")
et_model = ExtraTreesRegressor(n_estimators=100, max_depth=20, 
                                min_samples_split=5, random_state=42, n_jobs=-1)
et_model.fit(X_train_scaled, y_train)
et_pred = et_model.predict(X_test_scaled)
models_results['Extra Trees'] = evaluate_model('Extra Trees', y_test, et_pred)
models_results['Extra Trees']['predictions'] = et_pred

# Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                      learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
models_results['Gradient Boosting'] = evaluate_model('Gradient Boosting', y_test, gb_pred)
models_results['Gradient Boosting']['predictions'] = gb_pred

# AdaBoost
print("Training AdaBoost...")
ada_model = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
ada_model.fit(X_train_scaled, y_train)
ada_pred = ada_model.predict(X_test_scaled)
models_results['AdaBoost'] = evaluate_model('AdaBoost', y_test, ada_pred)
models_results['AdaBoost']['predictions'] = ada_pred

# XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
models_results['XGBoost'] = evaluate_model('XGBoost', y_test, xgb_pred)
models_results['XGBoost']['predictions'] = xgb_pred

# LightGBM
if LIGHTGBM_AVAILABLE:
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.1,
                                   random_state=42, verbose=-1)
    lgb_model.fit(X_train_scaled, y_train)
    lgb_pred = lgb_model.predict(X_test_scaled)
    models_results['LightGBM'] = evaluate_model('LightGBM', y_test, lgb_pred)
    models_results['LightGBM']['predictions'] = lgb_pred
else:
    print("LightGBM not available, skipping...")

# CatBoost
if CATBOOST_AVAILABLE:
    print("Training CatBoost...")
    cat_model = cb.CatBoostRegressor(iterations=100, depth=8, learning_rate=0.1,
                                      random_state=42, verbose=0)
    cat_model.fit(X_train_scaled, y_train)
    cat_pred = cat_model.predict(X_test_scaled)
    models_results['CatBoost'] = evaluate_model('CatBoost', y_test, cat_pred)
    models_results['CatBoost']['predictions'] = cat_pred
else:
    print("CatBoost not available, skipping...")

# SVR - RBF Kernel
print("Training SVR (RBF)...")
svr_rbf_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr_rbf_model.fit(X_train_scaled, y_train)
svr_rbf_pred = svr_rbf_model.predict(X_test_scaled)
models_results['SVR (RBF)'] = evaluate_model('SVR (RBF)', y_test, svr_rbf_pred)
models_results['SVR (RBF)']['predictions'] = svr_rbf_pred

# SVR - Linear Kernel
print("Training SVR (Linear)...")
svr_linear_model = SVR(kernel='linear', C=100, epsilon=0.1)
svr_linear_model.fit(X_train_scaled, y_train)
svr_linear_pred = svr_linear_model.predict(X_test_scaled)
models_results['SVR (Linear)'] = evaluate_model('SVR (Linear)', y_test, svr_linear_pred)
models_results['SVR (Linear)']['predictions'] = svr_linear_pred

# SVR - Polynomial Kernel
print("Training SVR (Poly)...")
svr_poly_model = SVR(kernel='poly', degree=3, C=100, epsilon=0.1)
svr_poly_model.fit(X_train_scaled, y_train)
svr_poly_pred = svr_poly_model.predict(X_test_scaled)
models_results['SVR (Poly)'] = evaluate_model('SVR (Poly)', y_test, svr_poly_pred)
models_results['SVR (Poly)']['predictions'] = svr_poly_pred

# KNN
print("Training KNN...")
knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
models_results['KNN'] = evaluate_model('KNN', y_test, knn_pred)
models_results['KNN']['predictions'] = knn_pred

# Multi-Layer Perceptron (MLP)
print("Building MLP model...")
mlp_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

mlp_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(mlp_model.summary())

# Train MLP
print("Training MLP...")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

mlp_history = mlp_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

mlp_pred = mlp_model.predict(X_test_scaled, verbose=0).flatten()
models_results['MLP'] = evaluate_model('MLP', y_test, mlp_pred)
models_results['MLP']['predictions'] = mlp_pred
models_results['MLP']['history'] = mlp_history

# Prepare sequence data for LSTM/GRU
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24
X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, time_steps)
X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, time_steps)

print(f"LSTM/GRU Training shape: {X_train_lstm.shape}")
print(f"LSTM/GRU Test shape: {X_test_lstm.shape}")

# LSTM Model
print("Building LSTM model...")
lstm_model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(time_steps, X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(lstm_model.summary())

# Train LSTM
print("Training LSTM...")
lstm_history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
models_results['LSTM'] = evaluate_model('LSTM', y_test_lstm, lstm_pred)
models_results['LSTM']['predictions'] = lstm_pred
models_results['LSTM']['history'] = lstm_history

# GRU Model
print("Building GRU model...")
gru_model = Sequential([
    GRU(128, activation='relu', return_sequences=True, input_shape=(time_steps, X_train_lstm.shape[2])),
    Dropout(0.2),
    GRU(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    GRU(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(gru_model.summary())

# Train GRU
print("Training GRU...")
gru_history = gru_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

gru_pred = gru_model.predict(X_test_lstm, verbose=0).flatten()
models_results['GRU'] = evaluate_model('GRU', y_test_lstm, gru_pred)
models_results['GRU']['predictions'] = gru_pred
models_results['GRU']['history'] = gru_history

# Plot training histories for deep learning models
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

dl_models = [('MLP', mlp_history), ('LSTM', lstm_history), ('GRU', gru_history)]

for idx, (name, history) in enumerate(dl_models):
    # Loss
    ax = axes[0, idx]
    ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    ax.set_title(f'{name} - Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE
    ax = axes[1, idx]
    ax.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax.set_title(f'{name} - MAE', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/07_deep_learning_training_history.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/07_deep_learning_training_history.png")

# Prepare data for Prophet
prophet_train = pd.DataFrame({
    'ds': df_train['Timestamp'],
    'y': df_train['EnergyConsumption'],
    'temperature': df_train['Temperature'],
    'humidity': df_train['Humidity'],
    'occupancy': df_train['Occupancy']
})

prophet_test = pd.DataFrame({
    'ds': df_test['Timestamp'],
    'temperature': df_test['Temperature'],
    'humidity': df_test['Humidity'],
    'occupancy': df_test['Occupancy']
})

print(f"Prophet train shape: {prophet_train.shape}")
print(f"Prophet test shape: {prophet_test.shape}")

# Train Prophet
print("Training Prophet...")
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='additive'
)

prophet_model.add_regressor('temperature')
prophet_model.add_regressor('humidity')
prophet_model.add_regressor('occupancy')

prophet_model.fit(prophet_train)

prophet_forecast = prophet_model.predict(prophet_test)
prophet_pred = prophet_forecast['yhat'].values

models_results['Prophet'] = evaluate_model('Prophet', y_test, prophet_pred)
models_results['Prophet']['predictions'] = prophet_pred

# Plot Prophet components
fig = prophet_model.plot_components(prophet_forecast)
plt.tight_layout()
plt.savefig('charts/08_prophet_components.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/08_prophet_components.png")

# Voting Regressor (ensemble of best models)
print("Training Voting Regressor...")
voting_model = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
])

voting_model.fit(X_train_scaled, y_train)
voting_pred = voting_model.predict(X_test_scaled)
models_results['Voting Regressor'] = evaluate_model('Voting Regressor', y_test, voting_pred)
models_results['Voting Regressor']['predictions'] = voting_pred

# Stacking Regressor
print("Training Stacking Regressor...")
stacking_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42))
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacking_model.fit(X_train_scaled, y_train)
stacking_pred = stacking_model.predict(X_test_scaled)
models_results['Stacking Regressor'] = evaluate_model('Stacking Regressor', y_test, stacking_pred)
models_results['Stacking Regressor']['predictions'] = stacking_pred

# Feature importance from Random Forest
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(14, 10))
plt.barh(feature_importance_df['feature'][:20], feature_importance_df['importance'][:20], 
         color='#1F77B4', alpha=0.8, edgecolor='black')
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 20 Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('charts/09_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/09_feature_importance.png")

# Create comparison dataframe
comparison_data = []
for model_name, results in models_results.items():
    comparison_data.append({
        'Model': model_name,
        'RMSE': results['RMSE'],
        'MAE': results['MAE'],
        'R2': results['R2'],
        'MAPE': results['MAPE'],
        'EVS': results['EVS']
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('RMSE')

print("\n" + "="*100)
print("MODEL COMPARISON (Sorted by RMSE)")
print("="*100)
print(comparison_df.to_string(index=False))
print("\n" + "="*100)

# Save comparison results to CSV
comparison_csv_path = OUTPUTS_DIR / 'model_comparison_results.csv'
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"Model comparison saved to: {comparison_csv_path}")

# RMSE Comparison Chart
fig, ax = plt.subplots(figsize=(16, 10))
bars = ax.barh(comparison_df['Model'], comparison_df['RMSE'], 
               color='#E76F51', alpha=0.8, edgecolor='black')
ax.set_xlabel('RMSE (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Model Comparison - Root Mean Squared Error', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f' {width:.3f}',
            ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('charts/10_model_comparison_rmse.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/10_model_comparison_rmse.png")

# R² Comparison Chart
comparison_df_r2 = comparison_df.sort_values('R2', ascending=False)
fig, ax = plt.subplots(figsize=(16, 10))
bars = ax.barh(comparison_df_r2['Model'], comparison_df_r2['R2'], 
               color='#2A9D8F', alpha=0.8, edgecolor='black')
ax.set_xlabel('R² Score (Higher is Better)', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Model Comparison - R² Score', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}',
            ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('charts/11_model_comparison_r2.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/11_model_comparison_r2.png")

# Multi-metric comparison
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE']
colors = ['#E76F51', '#F4A261', '#2A9D8F', '#264653']
titles = ['Lower is Better', 'Lower is Better', 'Higher is Better', 'Lower is Better']

for idx, (metric, color, title) in enumerate(zip(metrics_to_plot, colors, titles)):
    ax = axes[idx // 2, idx % 2]
    
    if metric == 'R2':
        sorted_df = comparison_df.sort_values(metric, ascending=False)
    else:
        sorted_df = comparison_df.sort_values(metric)
    
    bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=color, alpha=0.8, edgecolor='black')
    ax.set_xlabel(f'{metric} ({title})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison - {metric}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar in bars:
        width = bar.get_width()
        format_str = '.4f' if metric == 'R2' else '.3f'
        ax.text(width, bar.get_y() + bar.get_height()/2, f' {width:{format_str}}',
                ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('charts/12_model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/12_model_comparison_all_metrics.png")

# Top 5 Models - Predictions vs Actual
top_5_models = comparison_df.head(5)['Model'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()

for idx, model_name in enumerate(top_5_models):
    ax = axes[idx]
    
    if model_name in ['LSTM', 'GRU']:
        actual = y_test_lstm
        timestamps = df_test['Timestamp'].values[time_steps:]
    else:
        actual = y_test
        timestamps = df_test['Timestamp'].values
    
    pred = models_results[model_name]['predictions']
    
    plot_points = min(500, len(actual))
    ax.plot(timestamps[:plot_points], actual[:plot_points], 
            label='Actual', linewidth=2, alpha=0.7, color='#1f77b4')
    ax.plot(timestamps[:plot_points], pred[:plot_points], 
            label='Predicted', linewidth=2, alpha=0.7, color='#ff7f0e')
    
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Energy Consumption', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nRMSE: {models_results[model_name]["RMSE"]:.3f} | R²: {models_results[model_name]["R2"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('charts/13_top5_models_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/13_top5_models_predictions.png")

# Scatter plots - Predicted vs Actual (Top 6 models)
top_6_models = comparison_df.head(6)['Model'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, model_name in enumerate(top_6_models):
    ax = axes[idx]
    
    if model_name in ['LSTM', 'GRU']:
        actual = y_test_lstm
    else:
        actual = y_test
    
    pred = models_results[model_name]['predictions']
    
    ax.scatter(actual, pred, alpha=0.5, s=15)
    
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nR²: {models_results[model_name]["R2"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/14_scatter_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/14_scatter_predictions_vs_actual.png")

# Residual plots for top 6 models
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, model_name in enumerate(top_6_models):
    ax = axes[idx]
    
    if model_name in ['LSTM', 'GRU']:
        actual = y_test_lstm
    else:
        actual = y_test
    
    pred = models_results[model_name]['predictions']
    residuals = actual - pred
    
    ax.scatter(pred, residuals, alpha=0.5, s=15)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name} - Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/15_residual_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/15_residual_plots.png")

# Error distribution for top 6 models
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, model_name in enumerate(top_6_models):
    ax = axes[idx]
    
    if model_name in ['LSTM', 'GRU']:
        actual = y_test_lstm
    else:
        actual = y_test
    
    pred = models_results[model_name]['predictions']
    errors = actual - pred
    
    ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.axvline(x=errors.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    ax.set_xlabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name} - Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/16_error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved: charts/16_error_distribution.png")

# Comprehensive Summary
print("\n" + "="*100)
print("ENERGY CONSUMPTION PREDICTION - COMPREHENSIVE ANALYSIS SUMMARY")
print("="*100)

print(f"\nDataset Information:")
print(f"  Total Samples: {len(df):,}")
print(f"  Training Samples: {len(X_train):,}")
print(f"  Test Samples: {len(X_test):,}")
print(f"  Features Used: {len(feature_cols)}")
print(f"  Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

print(f"\nModels Trained: {len(models_results)}")
print(f"  Linear Models: 5 (Linear, Ridge, Lasso, ElasticNet, Bayesian Ridge)")
print(f"  Tree Models: 6 (Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging)")
print(f"  Boosting Models: 1-3 (XGBoost + LightGBM/CatBoost if available)")
print(f"  SVM Models: 3 (SVR with RBF, Linear, Poly kernels)")
print(f"  Neighbor Models: 1 (KNN)")
print(f"  Deep Learning: 3 (MLP, LSTM, GRU)")
print(f"  Time Series: 1 (Prophet)")
print(f"  Ensemble: 2 (Voting, Stacking)")

best_model = comparison_df.iloc[0]['Model']
best_rmse = comparison_df.iloc[0]['RMSE']
best_r2 = comparison_df.iloc[0]['R2']

print(f"\n" + "="*100)
print(f"BEST PERFORMING MODEL: {best_model}")
print("="*100)
print(f"  RMSE: {best_rmse:.4f}")
print(f"  MAE:  {comparison_df.iloc[0]['MAE']:.4f}")
print(f"  R²:   {best_r2:.4f}")
print(f"  MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%")

print(f"\n" + "="*100)
print("TOP 5 MODELS")
print("="*100)
for idx, row in comparison_df.head(5).iterrows():
    print(f"\n{idx+1}. {row['Model']}")
    print(f"   RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | R²: {row['R2']:.4f} | MAPE: {row['MAPE']:.2f}%")

print(f"\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)
print(f"\n1. Most Important Features (Top 5):")
for i, row in feature_importance_df.head(5).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

print(f"\n2. Model Performance Categories:")
excellent = comparison_df[comparison_df['R2'] >= 0.9]
good = comparison_df[(comparison_df['R2'] >= 0.8) & (comparison_df['R2'] < 0.9)]
moderate = comparison_df[comparison_df['R2'] < 0.8]
print(f"   - Excellent (R² >= 0.9): {len(excellent)} models")
print(f"   - Good (0.8 <= R² < 0.9): {len(good)} models")
print(f"   - Moderate (R² < 0.8): {len(moderate)} models")

print(f"\n3. Recommendations:")
print(f"   ✓ Deploy {best_model} for production use (best overall performance)")
print(f"   ✓ Consider ensemble of top 3 models for improved robustness")
print(f"   ✓ Monitor model drift with rolling validation")
print(f"   ✓ Retrain models periodically with new data")
print(f"   ✓ Focus on top features for feature engineering")
print(f"   ✓ Implement A/B testing for model deployment")

print(f"\n4. Visualizations:")
print(f"   ✓ All charts saved to 'charts/' folder (16 visualizations)")
print(f"   ✓ Model comparison saved to 'outputs/model_comparison_results.csv'")

print(f"\n" + "="*100)
print("ANALYSIS COMPLETE!")
print("="*100)
print("\nCheck the 'charts/' folder for all visualizations.")
print("Review 'outputs/model_comparison_results.csv' for detailed metrics.")
