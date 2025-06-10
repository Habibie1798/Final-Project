from prometheus_client import start_http_server, Counter, Histogram, Gauge
import requests
import time

MLFLOW_SERVE_URL = "http://127.0.0.1:5002/invocations"

INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total inference requests')
INFERENCE_SUCCESS = Counter('inference_success_total', 'Total successful inference')
INFERENCE_FAIL = Counter('inference_fail_total', 'Total failed inference')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency in seconds')
LAST_PREDICTION = Gauge('last_prediction', 'Last model prediction (0/1)')

SAMPLE_PAYLOAD = {
  "dataframe_split": {
    "columns": [
      "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","PaperlessBilling","MonthlyCharges","TotalCharges",
      "MultipleLines_No","MultipleLines_No phone service","MultipleLines_Yes",
      "InternetService_DSL","InternetService_Fiber optic","InternetService_No",
      "OnlineSecurity_No","OnlineSecurity_No internet service","OnlineSecurity_Yes",
      "OnlineBackup_No","OnlineBackup_No internet service","OnlineBackup_Yes",
      "DeviceProtection_No","DeviceProtection_No internet service","DeviceProtection_Yes",
      "TechSupport_No","TechSupport_No internet service","TechSupport_Yes",
      "StreamingTV_No","StreamingTV_No internet service","StreamingTV_Yes",
      "StreamingMovies_No","StreamingMovies_No internet service","StreamingMovies_Yes",
      "Contract_Month-to-month","Contract_One year","Contract_Two year",
      "PaymentMethod_Bank transfer (automatic)","PaymentMethod_Credit card (automatic)",
      "PaymentMethod_Electronic check","PaymentMethod_Mailed check"
    ],
    "data": [[1,0,1,0,12,1,1,65.3,1230.5,
        True,False,False,True,False,False,True,False,False,True,False,False,
        True,False,False,True,False,False,True,False,False,True,False,False,
        True,False,False,True,False,False,True,False]]
  }
}

def request_loop():
    while True:
        INFERENCE_REQUESTS.inc()
        start_time = time.time()
        try:
            r = requests.post(MLFLOW_SERVE_URL, json=SAMPLE_PAYLOAD, timeout=2)
            latency = time.time() - start_time
            INFERENCE_LATENCY.observe(latency)
            if r.status_code == 200:
                INFERENCE_SUCCESS.inc()
                result = r.json()["predictions"][0]
                LAST_PREDICTION.set(result)
            else:
                INFERENCE_FAIL.inc()
        except Exception:
            INFERENCE_FAIL.inc()
        time.sleep(10)

if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus exporter running on port 8000...")
    request_loop()
