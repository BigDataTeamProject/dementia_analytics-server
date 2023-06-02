from flask import Flask, jsonify, request
import numpy as np
import pickle

app = Flask(__name__)

# 모델 로드
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # POST 요청에서 딕셔너리 데이터 가져오기
    features = preprocess_data(data)  # 데이터 전처리 함수 호출
    prediction = model.predict(features)  # 예측
    result = {'prediction': prediction.tolist()}
    return jsonify(result)

def preprocess_data(data):
    features = np.array(list(data.values()))
    return features

if __name__ == '__main__':
    app.run()
