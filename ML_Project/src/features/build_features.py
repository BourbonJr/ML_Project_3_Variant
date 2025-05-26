import pandas as pd
import os
from pathlib import Path

def main():
    os.makedirs("data/features", exist_ok=True)
    
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")


    
    X_train.to_csv("data/features/X_train_processed.csv", index=False)
    X_test.to_csv("data/features/X_test_processed.csv", index=False)
    y_train.to_csv("data/features/y_train_processed.csv", index=False)
    y_test.to_csv("data/features/y_test_processed.csv", index=False)

if __name__ == "__main__":
    main()