import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays # For standard Indian holidays
import datetime

# --- 1. Data Simulation (Replace with your actual data) ---
# For demonstration purposes, let's create a synthetic dataset
# Your actual data should have at least 'ds' (date) and 'y' (withdrawal amount) columns.
print("Simulating historical ATM withdrawal data...")
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame({'ds': dates})

# Simulate a general trend
data['y'] = 5000 + (data.index * 5)

# Simulate weekly seasonality (more withdrawals on weekends)
data['y'] += data['ds'].dt.dayofweek.apply(lambda x: 1000 if x >= 5 else 0)

# Simulate monthly seasonality (e.g., higher at the beginning/end of month)
data['y'] += data['ds'].dt.day.apply(lambda x: 800 if x <= 5 or x >= 25 else 0)

# Add some random noise
data['y'] += np.random.normal(0, 500, len(data))

# Ensure withdrawals are non-negative
data['y'] = data['y'].apply(lambda x: max(0, x))

# Simulate a few major Indian festival spikes and salary day spikes
# These are illustrative; you'll need actual dates for your specific festivals/salary days.
def add_event_spike(df, date_str, spike_amount, days_before=0):
    date = pd.to_datetime(date_str)
    for i in range(days_before + 1):
        target_date = date - pd.Timedelta(days=i)
        df.loc[df['ds'] == target_date, 'y'] += spike_amount * (1 - i / (days_before + 1)) # Decreasing spike before event

# Example Indian festivals and salary days (adjust as per your needs)
add_event_spike(data, '2021-11-04', 3000, days_before=3) # Diwali 2021
add_event_spike(data, '2022-10-24', 3500, days_before=3) # Diwali 2022
add_event_spike(data, '2023-11-12', 4000, days_before=3) # Diwali 2023

add_event_spike(data, '2021-03-29', 1500, days_before=2) # Holi 2021
add_event_spike(data, '2022-03-18', 1800, days_before=2) # Holi 2022
add_event_spike(data, '2023-03-08', 2000, days_before=2) # Holi 2023

# Salary days (e.g., 1st of every month, and 15th for some companies)
for year in range(2020, 2024):
    for month in range(1, 13):
        add_event_spike(data, f'{year}-{month}-01', 2500, days_before=1) # 1st of month
        if month % 2 == 0: # Example: some companies pay on 15th every other month
            add_event_spike(data, f'{year}-{month}-15', 1500, days_before=1)

print("Data simulation complete. First 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())

# --- 2. Define Indian Holidays and Custom Events ---

def generate_indian_holidays_and_events(years):
    """
    Generates a DataFrame of Indian government holidays, major festivals,
    and salary days for use with Prophet.
    """
    all_holidays = pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

    # Standard Indian Government Holidays (using 'holidays' library)
    # Note: The 'holidays' library covers many national/state holidays.
    # You might need to refine this based on specific state holidays relevant to your ATMs.
    in_holidays = holidays.India(years=years, state='MH') # Example: Maharashtra state holidays
    for date, name in sorted(in_holidays.items()):
        all_holidays = pd.concat([all_holidays, pd.DataFrame([{
            'holiday': name,
            'ds': pd.to_datetime(date),
            'lower_window': 0, # Start effect on the day of the holiday
            'upper_window': 0  # End effect on the day of the holiday
        }])], ignore_index=True)

    # Custom Festivals (Manual addition - highly recommend maintaining a separate list)
    # You can add more specific festivals here.
    custom_festivals = [
        {'holiday': 'Diwali', 'ds': '2020-11-14', 'lower_window': -3, 'upper_window': 1},
        {'holiday': 'Diwali', 'ds': '2021-11-04', 'lower_window': -3, 'upper_window': 1},
        {'holiday': 'Diwali', 'ds': '2022-10-24', 'lower_window': -3, 'upper_window': 1},
        {'holiday': 'Diwali', 'ds': '2023-11-12', 'lower_window': -3, 'upper_window': 1},
        {'holiday': 'Holi', 'ds': '2020-03-09', 'lower_window': -2, 'upper_window': 0},
        {'holiday': 'Holi', 'ds': '2021-03-29', 'lower_window': -2, 'upper_window': 0},
        {'holiday': 'Holi', 'ds': '2022-03-18', 'lower_window': -2, 'upper_window': 0},
        {'holiday': 'Holi', 'ds': '2023-03-08', 'lower_window': -2, 'upper_window': 0},
        {'holiday': 'Eid al-Fitr', 'ds': '2020-05-24', 'lower_window': -2, 'upper_window': 1},
        {'holiday': 'Eid al-Fitr', 'ds': '2021-05-13', 'lower_window': -2, 'upper_window': 1},
        {'holiday': 'Eid al-Fitr', 'ds': '2022-05-02', 'lower_window': -2, 'upper_window': 1},
        {'holiday': 'Eid al-Fitr', 'ds': '2023-04-22', 'lower_window': -2, 'upper_window': 1},
        # Add more specific festivals like Ganesh Chaturthi, Durga Puja, Christmas, etc.
    ]
    for h in custom_festivals:
        all_holidays = pd.concat([all_holidays, pd.DataFrame([{
            'holiday': h['holiday'],
            'ds': pd.to_datetime(h['ds']),
            'lower_window': h['lower_window'],
            'upper_window': h['upper_window']
        }])], ignore_index=True)

    # Salary Days (e.g., 1st and 15th of every month)
    salary_days = []
    for year in years:
        for month in range(1, 13):
            # 1st of the month
            salary_days.append({
                'holiday': 'Salary_Day_1st',
                'ds': pd.to_datetime(f'{year}-{month}-01'),
                'lower_window': -1, 'upper_window': 0 # Effect on day before and day of
            })
            # 15th of the month (example for a second payday)
            if month % 2 == 0: # Just an example, adjust logic for your specific paydays
                 salary_days.append({
                    'holiday': 'Salary_Day_15th',
                    'ds': pd.to_datetime(f'{year}-{month}-15'),
                    'lower_window': -1, 'upper_window': 0
                })

    for h in salary_days:
        all_holidays = pd.concat([all_holidays, pd.DataFrame([{
            'holiday': h['holiday'],
            'ds': pd.to_datetime(h['ds']),
            'lower_window': h['lower_window'],
            'upper_window': h['upper_window']
        }])], ignore_index=True)

    # Remove duplicates if any (e.g., if a holiday falls on a salary day)
    all_holidays.drop_duplicates(subset=['ds', 'holiday'], inplace=True)
    all_holidays.sort_values(by='ds', inplace=True)
    return all_holidays

# Generate holidays for the data range
holiday_years = range(data['ds'].min().year, data['ds'].max().year + 2) # +2 for future forecasting
indian_holidays_df = generate_indian_holidays_and_events(holiday_years)
print("\nGenerated Indian Holidays and Custom Events (first 5 rows):")
print(indian_holidays_df.head())

# --- 3. Feature Engineering for XGBoost ---

def create_xgboost_features(df, holidays_df):
    """
    Creates features for the XGBoost model, including temporal, lagged,
    and detailed holiday/event indicators.
    """
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['ds'].dt.quarter
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)

    # Lagged features (e.g., previous day's withdrawal, previous week's same day)
    # Ensure 'y' column exists for lagged features (it will for training data)
    if 'y' in df.columns:
        # Changed inplace=True to direct assignment
        df['y_lag_1'] = df['y'].shift(1).fillna(df['y'].mean()) # Use df['y'].mean() for more robust filling
        df['y_lag_7'] = df['y'].shift(7).fillna(df['y'].mean()) # Use df['y'].mean() for more robust filling
    else:
        # For future predictions where 'y' is not known, these will be NaN or need imputation
        # For simplicity in this example, we'll just create empty columns for future_df
        df['y_lag_1'] = np.nan
        df['y_lag_7'] = np.nan


    # Detailed Holiday/Event Features for XGBoost
    # Create binary flags for each specific holiday and its surrounding days
    unique_holidays = holidays_df['holiday'].unique()
    for h_name in unique_holidays:
        holiday_dates = holidays_df[holidays_df['holiday'] == h_name]
        for _, row in holiday_dates.iterrows():
            ds = row['ds']
            lw = row['lower_window']
            uw = row['upper_window']

            # Create a range of dates affected by this holiday
            affected_dates = pd.date_range(start=ds + pd.Timedelta(days=lw),
                                           end=ds + pd.Timedelta(days=uw),
                                           freq='D')

            # Create a binary column for this specific holiday
            col_name = f'is_{h_name.replace(" ", "_").replace("-", "_").lower()}'
            df[col_name] = df['ds'].isin(affected_dates).astype(int)

    return df

# --- 4. Split Data into Training and Testing Sets ---
# Use a time-based split for forecasting
train_size = int(len(data) * 0.8)
train_df = data.iloc[:train_size].copy()
test_df = data.iloc[train_size:].copy()

print(f"\nTraining data size: {len(train_df)} rows")
print(f"Testing data size: {len(test_df)} rows")

# --- 5. Prophet Model Training ---
print("\nTraining Prophet model...")
m = Prophet(
    seasonality_mode='additive', # Or 'multiplicative' if seasonality scales with trend
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    holidays=indian_holidays_df # Incorporate defined holidays
)

# Fit the Prophet model
m.fit(train_df)

# Create future dataframe for Prophet predictions (including test period)
future = m.make_future_dataframe(periods=len(test_df), include_history=True)
# Ensure holidays are present in the future dataframe for Prophet to use them
# Prophet automatically merges holidays if provided during initialization.

# Generate Prophet predictions for the entire dataset (train + test)
forecast = m.predict(future)
print("Prophet prediction complete. First 5 rows of forecast:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Merge Prophet's predictions back to the original dataframes
train_df = pd.merge(train_df, forecast[['ds', 'yhat']], on='ds', how='left')
test_df = pd.merge(test_df, forecast[['ds', 'yhat']], on='ds', how='left')

# Calculate Prophet residuals for training XGBoost
train_df['prophet_residual'] = train_df['y'] - train_df['yhat']
# FIX: Calculate prophet_residual for test_df as well
test_df['prophet_residual'] = test_df['y'] - test_df['yhat']

print("\nProphet residuals calculated for training data. First 5 rows:")
print(train_df[['ds', 'y', 'yhat', 'prophet_residual']].head())
print("\nProphet residuals calculated for test data. First 5 rows:")
print(test_df[['ds', 'y', 'yhat', 'prophet_residual']].head())


# --- 6. XGBoost Model Training on Residuals ---
print("\nPreparing features for XGBoost...")

# Create features for both training and testing sets
train_df_xgb = create_xgboost_features(train_df.copy(), indian_holidays_df)
test_df_xgb = create_xgboost_features(test_df.copy(), indian_holidays_df)

# Define features (X) and target (y_residual) for XGBoost
# Exclude 'ds', 'y', 'yhat', 'prophet_residual' from features
features = [col for col in train_df_xgb.columns if col not in ['ds', 'y', 'yhat', 'prophet_residual']]

# Align columns between train and test sets for XGBoost
# This is crucial if some holiday features might only appear in one set
# Get all unique feature names from both train and test to ensure alignment
all_feature_cols = list(set(train_df_xgb.columns) | set(test_df_xgb.columns))
# Filter to only include columns that are actual features (not target or internal)
features = [f for f in all_feature_cols if f not in ['ds', 'y', 'yhat', 'prophet_residual']]


X_train = train_df_xgb[features]
y_train_residual = train_df_xgb['prophet_residual']
# Ensure X_test has all the same columns as X_train, filling missing with 0
X_test = test_df_xgb.reindex(columns=features, fill_value=0)


# Handle potential NaN values in features (e.g., from lagged features at the very beginning)
# Changed inplace=True to direct assignment
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean()) # Use train mean for test set

print(f"XGBoost features: {features[:5]}... ({len(features)} total features)")
print(f"X_train shape: {X_train.shape}, y_train_residual shape: {y_train_residual.shape}")
print(f"X_test shape: {X_test.shape}")

# Initialize and train XGBoost Regressor
print("\nTraining XGBoost model on Prophet residuals...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', # For regression tasks
    n_estimators=1000,           # Number of boosting rounds
    learning_rate=0.05,          # Step size shrinkage
    max_depth=5,                 # Maximum depth of a tree
    subsample=0.8,               # Subsample ratio of the training instance
    colsample_bytree=0.8,        # Subsample ratio of columns when constructing each tree
    random_state=42,
    n_jobs=-1,                   # Use all available cores
    tree_method='hist'           # Faster tree method for larger datasets
)

xgb_model.fit(X_train, y_train_residual)
print("XGBoost training complete.")

# Predict residuals using the trained XGBoost model
xgb_residual_predictions = xgb_model.predict(X_test)
test_df['xgb_residual_pred'] = xgb_residual_predictions
print("\nXGBoost residual predictions generated. First 5 rows of test_df with residual pred:")
print(test_df[['ds', 'y', 'yhat', 'prophet_residual', 'xgb_residual_pred']].head())


# --- 7. Hybrid Prediction ---
# The final hybrid forecast is Prophet's prediction plus XGBoost's residual prediction
test_df['hybrid_forecast'] = test_df['yhat'] + test_df['xgb_residual_pred']
print("\nHybrid forecast generated. First 5 rows of test_df with hybrid forecast:")
print(test_df[['ds', 'y', 'yhat', 'hybrid_forecast']].head())

# --- 8. Evaluation ---
print("\n--- Model Evaluation ---")

# Evaluate Prophet's standalone performance on the test set
prophet_mae = mean_absolute_error(test_df['y'], test_df['yhat'])
prophet_rmse = np.sqrt(mean_squared_error(test_df['y'], test_df['yhat']))
print(f"Prophet (Standalone) MAE on Test Set: {prophet_mae:.2f}")
print(f"Prophet (Standalone) RMSE on Test Set: {prophet_rmse:.2f}")

# Evaluate Hybrid model performance on the test set
hybrid_mae = mean_absolute_error(test_df['y'], test_df['hybrid_forecast'])
hybrid_rmse = np.sqrt(mean_squared_error(test_df['y'], test_df['hybrid_forecast']))
print(f"Hybrid (Prophet + XGBoost) MAE on Test Set: {hybrid_mae:.2f}")
print(f"Hybrid (Prophet + XGBoost) RMSE on Test Set: {hybrid_rmse:.2f}")

# --- 9. Visualize Results (Optional - requires matplotlib) ---
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 7))
    plt.plot(train_df['ds'], train_df['y'], label='Actual (Train)', color='blue', alpha=0.7)
    plt.plot(test_df['ds'], test_df['y'], label='Actual (Test)', color='blue')
    plt.plot(test_df['ds'], test_df['yhat'], label='Prophet Forecast (Test)', color='orange', linestyle='--')
    plt.plot(test_df['ds'], test_df['hybrid_forecast'], label='Hybrid Forecast (Test)', color='green')

    # Plot actual values for the period covered by the future dataframe
    # This ensures the plot includes the entire forecast period
    full_data_for_plot = pd.concat([train_df, test_df])
    plt.plot(full_data_for_plot['ds'], full_data_for_plot['yhat'], label='Prophet Forecast (Full Data)', color='red', linestyle=':', alpha=0.6)

    plt.title('ATM Cash Withdrawal Forecasting: Actual vs. Prophet vs. Hybrid Forecast')
    plt.xlabel('Date')
    plt.ylabel('Withdrawal Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot residuals to see what XGBoost is learning
    plt.figure(figsize=(15, 5))
    plt.plot(train_df['ds'], train_df['prophet_residual'], label='Prophet Residuals (Train)', color='purple', alpha=0.7)
    plt.plot(test_df['ds'], test_df['prophet_residual'], label='Prophet Residuals (Test)', color='red', alpha=0.7)
    plt.plot(test_df['ds'], test_df['xgb_residual_pred'], label='XGBoost Residual Prediction (Test)', color='cyan', linestyle='--')
    plt.title('Prophet Residuals and XGBoost Residual Predictions')
    plt.xlabel('Date')
    plt.ylabel('Residual Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

except ImportError:
    print("\nMatplotlib not installed. Skipping visualization.")
