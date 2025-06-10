import requests
import pandas as pd
import json

CSV_PATH = r"C:\Users\ABE\Proyek Akhir Dicoding\Final Project\namadataset_preprocessing\telco_preprocessed.csv"
ENDPOINT = "http://127.0.0.1:5002/invocations"

df = pd.read_csv(CSV_PATH)
X = df.drop('Churn', axis=1)

# Pilih 1 baris
data_test = X.iloc[[0]]
payload = {
    "dataframe_split": {
        "columns": list(X.columns),
        "data": data_test.values.tolist()
    }
}

headers = {"Content-Type": "application/json"}
print("Mengirim 1 request prediksi...")
response = requests.post(ENDPOINT, headers=headers, data=json.dumps(payload))
print("Status code:", response.status_code)
try:
    print("Response:", response.json())
except Exception as e:
    print("Response:", response.text)
