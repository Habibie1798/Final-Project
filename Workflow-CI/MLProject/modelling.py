import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Untuk support parameter dari MLflow Project/CI: --data_path
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='namadataset_preprocessing/telco_preprocessed.csv')
args = parser.parse_args()

# Load data hasil preprocessing
df = pd.read_csv(args.data_path)
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Akurasi: {acc}")
