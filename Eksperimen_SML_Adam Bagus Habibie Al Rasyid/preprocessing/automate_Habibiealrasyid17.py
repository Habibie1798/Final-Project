import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0, 'Female':1, 'Male':0})

    multi_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cat)

    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print('Preprocessing selesai. Hasil disimpan di:', output_path)

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, '..', 'namadataset_raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_path = os.path.join(BASE_DIR, 'namadataset_preprocessing', 'telco_preprocessed.csv')
    preprocess(input_path, output_path)
