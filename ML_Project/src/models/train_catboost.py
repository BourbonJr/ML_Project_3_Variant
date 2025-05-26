from catboost import CatBoostRegressor
import pandas as pd
import yaml
from joblib import dump

with open("params.yaml") as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv("data/features/X_train_processed.csv")
y_train = pd.read_csv("data/features/y_train_processed.csv")

target_col = params["feature_params"]["target"]
y = y_train[target_col] if target_col in y_train.columns else y_train.iloc[:, 0]

cat_features = [
    i for i, col in enumerate(X_train.columns) 
    if col in params["feature_params"]["categorical_features"]
]

model = CatBoostRegressor(
    iterations=params["catboost"]["iterations"],
    learning_rate=params["catboost"]["learning_rate"],
    cat_features=cat_features,
    random_seed=params["catboost"].get("random_seed", 42),
    verbose=100 
)
model.fit(X_train, y)


model.save_model("models/catboost.cbm")
