from xgboost import XGBRegressor
import pandas as pd
import yaml
from joblib import dump

with open("params.yaml") as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv("data/features/X_train_processed.csv")
y_train = pd.read_csv("data/features/y_train_processed.csv")

target_col = params["feature_params"]["target"]
y = y_train[target_col] if target_col in y_train.columns else y_train.iloc[:, 0]

model = XGBRegressor(
    max_depth=params["xgboost"]["max_depth"],
    learning_rate=params["xgboost"]["learning_rate"],
    random_state=params["xgboost"].get("random_state", 42)
)
model.fit(X_train, y)

model.save_model("models/xgboost.model")