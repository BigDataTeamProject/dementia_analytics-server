from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
import csv
import os

print(os.path.dirname(os.path.realpath(__file__)))
os.chdir("/Users/ijeonhui/Github/BigDataTeamProject/dementia_analytics-server")

app = Flask(__name__)

# 모델 로드
with open('./xgb_model.pkl', 'rb') as f:
    loaded_scaler = joblib.load('xgb_scaler.pkl')
    loaded_model = joblib.load('xgb_model.pkl')


dataset_mean = pd.read_csv('파일위치/dataset_mean.csv', encoding = 'utf-8')
@app.route('/mean', methods=['GET'])
def get_dataset_mean():
    return dataset_mean.to_json()
#     return dataset_mean.to_json(orient='records')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # POST 요청에서 딕셔너리 데이터 가져오기
    features = preprocess_data(data)  # 데이터 전처리 함수 호출
    prediction = loaded_model.predict(loaded_scaler.transform(features))  # 예측
    print(prediction)
    result = {'prediction': prediction.tolist}
    return jsonify(result)


def preprocess_data(data):
    features = np.array([[
        data['sleep_breath_average'],
        data['sleep_hr_average'],
        data['sleep_hr_lowest'],
        data['sleep_deep'],
        data['sleep_rem'],
        data['activity_cal_total'],
        data['sleep_awake'],
        data['activity_steps'],
        data['activity_total'],
        data['sleep_duration'],
        data['activity_daily_movement']
    ]])
    return features

# dataset_mean = pd.read_csv('파일위치/dataset_mean.csv', encoding = 'utf-8')
# def load_dataset_mean(dataset_mean.csv):
#     dataset_mean = {}
#     with open(dataset_mean.csv, 'r') as csv_file:
#         reader = csv.DictReader(csv_file)
#         for row in reader:
#             feature_name = row['Feature']
#             mean_value = float(row['Mean'])
#             dataset_mean[feature_name] = mean_value
#     return dataset_mean

if __name__ == '__main__':
    app.run()
