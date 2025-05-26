import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from joblib import dump

with open("params.yaml") as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv("data/features/X_train_processed.csv")
y_train = pd.read_csv("data/features/y_train_processed.csv")

target_col = params["feature_params"]["target"]
y = y_train[target_col] if target_col in y_train.columns else y_train.iloc[:, 0]

model = LinearRegression(
    fit_intercept=params["linear_regression"]["fit_intercept"]
)
model.fit(X_train, y)

dump(model, "models/linear_regression.joblib")