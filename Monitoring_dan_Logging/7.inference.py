import requests

# Data inference (1 baris) SESUAI schema yang dibutuhkan oleh model
# Pastikan semua kolom yang required diisi
data = {
    "inputs": [
        {
            "gender": 1,
            "SeniorCitizen": 0,
            "Partner": 1,
            "Dependents": 0,
            "tenure": 12,
            "PhoneService": 1,
            "PaperlessBilling": 1,
            "MonthlyCharges": 70.35,
            "TotalCharges": 845.5,

            "MultipleLines_No": False,
            "MultipleLines_No phone service": False,
            "MultipleLines_Yes": True,

            "InternetService_DSL": True,
            "InternetService_Fiber optic": False,
            "InternetService_No": False,

            "OnlineSecurity_No": False,
            "OnlineSecurity_No internet service": True,
            "OnlineSecurity_Yes": False,

            "OnlineBackup_No": False,
            "OnlineBackup_No internet service": True,
            "OnlineBackup_Yes": False,

            "DeviceProtection_No": True,
            "DeviceProtection_No internet service": False,
            "DeviceProtection_Yes": False,

            "TechSupport_No": True,
            "TechSupport_No internet service": False,
            "TechSupport_Yes": False,

            "StreamingTV_No": True,
            "StreamingTV_No internet service": False,
            "StreamingTV_Yes": False,

            "StreamingMovies_No": False,
            "StreamingMovies_No internet service": False,
            "StreamingMovies_Yes": True,

            "Contract_Month-to-month": True,
            "Contract_One year": False,
            "Contract_Two year": False,

            "PaymentMethod_Bank transfer (automatic)": False,
            "PaymentMethod_Credit card (automatic)": True,
            "PaymentMethod_Electronic check": False,
            "PaymentMethod_Mailed check": False
        }
    ]
}

# Kirim request POST ke server MLflow
response = requests.post(
    url="http://127.0.0.1:5002/invocations",
    headers={"Content-Type": "application/json"},
    json=data
)

# Tampilkan hasil prediksi
print("Prediksi:", response.json())
