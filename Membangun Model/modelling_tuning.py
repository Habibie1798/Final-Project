import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# =======================
# 1. KONEK DAGSHUB
dagshub.init(
    repo_owner='adambagushabibiear',     # Ganti sesuai username kamu
    repo_name='Final_Project',           # Ganti sesuai nama repo kamu
    mlflow=True
)

# =======================
# 2. LOAD DATA
df = pd.read_csv('namadataset_preprocessing/telco_preprocessed.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================
# 3. TUNING
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)

start_time = time.time()
grid.fit(X_train, y_train)
training_time = time.time() - start_time

best_model = grid.best_estimator_

# =======================
# 4. METRIKS & PREDIKSI
preds = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]   # Untuk ROC AUC

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
roc_auc = roc_auc_score(y_test, proba)   # METRIK TAMBAHAN

# =======================
# 5. LOGGING MANUAL KE DAGSHUB/MLflow
with mlflow.start_run() as run:
    # Log parameter tuning
    mlflow.log_param('n_estimators', best_model.n_estimators)
    mlflow.log_param('max_depth', best_model.max_depth)
    mlflow.log_param('cv_folds', 3)
    # Log metrik utama
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    # METRIK TAMBAHAN
    mlflow.log_metric('roc_auc', roc_auc)
    mlflow.log_metric('training_time', training_time)
    mlflow.log_metric('num_features', X.shape[1])
    # Log model untuk serve MLflow
    mlflow.sklearn.log_model(best_model, "model")   # << PENTING!
    # Log artefak model juga (opsional untuk backup)
    joblib.dump(best_model, 'rf_best_model.pkl')
    mlflow.log_artifact('rf_best_model.pkl')
    # Log confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('conf_matrix.png')
    mlflow.log_artifact('conf_matrix.png')
    plt.close()
    # LOG ARTEFAK TAMBAHAN: ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    mlflow.log_artifact('roc_curve.png')
    plt.close()
    # Info run id untuk serve
    print(f"Run ID untuk serve: {run.info.run_id}")

print(f"Best Params: {grid.best_params_}")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")
print("Model, metriks, dan artifacts sudah di-log ke DagsHub MLflow!")
