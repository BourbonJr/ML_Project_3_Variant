import pandas as pd
import yaml
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

with open("params.yaml") as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv("data/features/X_train_processed.csv")
y_train = pd.read_csv("data/features/y_train_processed.csv")

print("Столбцы в X_train:", X_train.columns.tolist())
print("Столбцы в y_train:", y_train.columns.tolist())

target_col = params["feature_params"]["target"]
if target_col in y_train.columns:
    y = y_train[target_col]
else:
    y = y_train.iloc[:, 0]

model = DecisionTreeRegressor(
    max_depth=params["decision_tree"]["max_depth"],
    min_samples_split=params["decision_tree"]["min_samples_split"],
    random_state=params["decision_tree"].get("random_state", 42)
)
model.fit(X_train, y)

dump(model, "models/decision_tree.joblib")