import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import optuna
import warnings
import joblib
warnings.filterwarnings('ignore')

# Load your data
def load_and_prepare_data(file_path='combined_energies_sorted.csv'):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace('$$', '').str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData statistics:")
    print(df.describe())
    return df

# Prepare features and target
def prepare_features_target(df, target_col='interaction_energy'):
    y = df[target_col].values
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    feature_names = feature_cols
    print(f"\nFeatures: {feature_names}")
    print(f"Target: {target_col}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y, feature_names

# Split data into train, validation, and test sets (8:1:1)
def split_data_8_1_1(X, y, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    print(f"\nData split (8:1:1):")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Optuna objective function
def objective(trial, X_train, X_val, y_train, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse'
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# Hyperparameter optimization
def optimize_hyperparameters(X_train, X_val, y_train, y_val, n_trials=100):
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*60)
    print(f"Running {n_trials} trials...")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"\nBest RMSE: {best_rmse:.6f}")
    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params, study

# Train final model
def train_final_model(X_train, y_train, X_val, y_val, best_params):
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*60)
    
    params = best_params.copy()
    params.update({
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse'
    })
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    return model

# Evaluate the model (metrics only, no plots)
def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    def calculate_metrics(y_true, y_pred, set_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    train_metrics = calculate_metrics(y_train, y_train_pred, 'Train')
    val_metrics = calculate_metrics(y_val, y_val_pred, 'Validation')
    test_metrics = calculate_metrics(y_test, y_test_pred, 'Test')
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"{'Set':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*50)
    print(f"{'Train':<12} {train_metrics['RMSE']:<12.6f} {train_metrics['MAE']:<12.6f} {train_metrics['R²']:<12.6f}")
    print(f"{'Validation':<12} {val_metrics['RMSE']:<12.6f} {val_metrics['MAE']:<12.6f} {val_metrics['R²']:<12.6f}")
    print(f"{'Test':<12} {test_metrics['RMSE']:<12.6f} {test_metrics['MAE']:<12.6f} {test_metrics['R²']:<12.6f}")
    print("="*60)
    
    # Feature importance (text output only)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFEATURE IMPORTANCE:")
    print(feature_importance.to_string(index=False))
    
    return test_metrics

# Main execution
def main():
    print("="*60)
    print("XGBOOST MODEL WITH OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("8:1:1 Train/Validation/Test Split (NO SCALING NEEDED)")
    print("="*60)
    
    # 1. Load data
    df = load_and_prepare_data('combined_energies_sorted.csv')
    
    # 2. Prepare features and target
    X, y, feature_names = prepare_features_target(df)
    
    # 3. Split data into 8:1:1
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_8_1_1(X, y)
    
    # 4. NO SCALING - XGBoost doesn't need it!
    
    # 5. Hyperparameter optimization with Optuna
    best_params, study = optimize_hyperparameters(
        X_train, X_val, y_train, y_val, n_trials=100
    )
    
    # 6. Train final model
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    
    # 7. Evaluate model
    test_metrics = evaluate_model(
        model, X_train, X_val, X_test,
        y_train, y_val, y_test, feature_names
    )
    
    # 8. Save model (only the model, not the study)
    joblib.dump(model, 'xgboost_optuna_model.pkl')
    print("\n✓ Model saved as 'xgboost_optuna_model.pkl'")
    
    return model, study

if __name__ == "__main__":
    model, study = main()