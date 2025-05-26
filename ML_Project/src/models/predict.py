import pandas as pd
import yaml
import json
import os
from joblib import load
from sklearn.metrics import r2_score

def load_model(model_path):
    """Загрузка модели"""
    if model_path.endswith('.cbm'):
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    elif model_path.endswith('.model'):
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(model_path)
        return model
    else:
        return load(model_path)

def evaluate_model(model, X_test, y_true):
    """Оценка модели с универсальным расчетом RMSE"""
    y_pred = model.predict(X_test)
    
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = mse ** 0.5 
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2_score(y_true, y_pred))
    }

def main():
    os.makedirs("reports", exist_ok=True)
    
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    X_test = pd.read_csv("data/features/X_test_processed.csv")
    y_test = pd.read_csv("data/features/y_test_processed.csv")
    
    target_col = params["feature_params"]["target"]
    y_true = y_test[target_col] if target_col in y_test.columns else y_test.iloc[:, 0]
    
    metrics = {}
    model_files = {
        'linear_regression': 'models/linear_regression.joblib',
        'decision_tree': 'models/decision_tree.joblib',
        'catboost': 'models/catboost.cbm',
        'xgboost': 'models/xgboost.model',
        'mlp': 'models/mlp.joblib'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                model = load_model(path)
                metrics[name] = evaluate_model(model, X_test, y_true)
                print(f"Модель {name} успешно оценена")
            except Exception as e:
                print(f"Ошибка оценки модели {name}: {str(e)}")
                metrics[name] = {"error": str(e)}
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Оценка завершена. Метрики сохранены в reports/metrics.json")

if __name__ == "__main__":
    main()