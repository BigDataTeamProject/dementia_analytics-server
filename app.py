from flask import Flask, request

app = Flask(__name__)

@app.route('/da', methods=['GET', 'POST'])
def dementia_alnaytics():
    if request.method == 'POST':
        # 머신러닝 모델 데이터 전송
        return request.data

if __name__ == '__main__':
    app.run()
