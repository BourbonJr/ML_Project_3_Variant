import pandas as pd
import numpy as np
import yaml
import json  # Добавлен отсутствующий импорт
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from sklearn.model_selection import train_test_split
import os

# 1. Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# 2. Создание директорий для отчетов
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/plots", exist_ok=True)

# 3. Загрузка данных
X_train = pd.read_csv("data/features/X_train_processed.csv")
y_train = pd.read_csv("data/features/y_train_processed.csv")

# 4. Валидация данных
print("Форма X_train:", X_train.shape)
print("Форма y_train:", y_train.shape)

# 5. Подготовка целевой переменной
target_col = params["feature_params"]["target"]
if target_col in y_train.columns:
    y = y_train[target_col]
else:
    y = y_train.iloc[:, 0]

# 6. Разделение на train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y, 
    test_size=0.2, 
    random_state=42
)

# 7. Создание и обучение модели
model = MLPRegressor(
    hidden_layer_sizes=params["mlp"]["hidden_layer_sizes"],
    activation=params["mlp"]["activation"],
    random_state=params["mlp"].get("random_state", 42),
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=10,
    verbose=True
)

history = model.fit(X_train, y_train)

# 8. Оценка модели
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

train_metrics = evaluate_model(model, X_train, y_train)
val_metrics = evaluate_model(model, X_val, y_val)

# 9. Визуализация кривой обучения
plt.figure(figsize=(10, 5))
plt.plot(history.loss_curve_)
plt.title('Кривая обучения MLP')
plt.ylabel('Loss')
plt.xlabel('Эпохи')
plt.grid(True)
plt.savefig('reports/plots/mlp_learning_curve.png')
plt.close()

# 10. Сохранение результатов
dump(model, "models/mlp.joblib")

metrics = {
    'train': train_metrics,
    'validation': val_metrics,
    'learning_curve': 'reports/plots/mlp_learning_curve.png'
}

with open("reports/mlp_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nМетрики на тренировочных данных:")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nМетрики на валидационных данных:")
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nМодель и метрики успешно сохранены")