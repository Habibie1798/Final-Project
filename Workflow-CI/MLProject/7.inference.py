import requests
import pandas as pd

# Ganti ini sesuai kolom fitur yang kamu pakai di modelmu
data = pd.DataFrame({
    "feature1": [5.1],
    "feature2": [3.5],
    "feature3": [1.4],
    "feature4": [0.2]
})

response = requests.post(
    url="http://127.0.0.1:5002/invocations",
    headers={"Content-Type": "application/json"},
    json={"columns": data.columns.tolist(), "data": data.values.tolist()}
)

print("Prediksi:", response.json())
