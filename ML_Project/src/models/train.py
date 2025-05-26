import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(model, model_name, params):
    """Обучение и сохранение модели"""
    project_dir = Path(__file__).resolve().parents[2]
    features_path = project_dir / params["data_params"]["features_dir"] / "train_processed.csv"
    output_path = project_dir / "models" / f"{model_name}.pkl"
    
    data = pd.read_csv(features_path)
    X = data.drop(columns=[params["feature_params"]["target"]])
    y = data[params["feature_params"]["target"]]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=params["data_params"]["test_size"],
        random_state=params["data_params"]["random_state"]
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"{model_name} performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")
    
    return model

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    train_model(model, "linear_regression", params)