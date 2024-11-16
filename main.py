from flask import Flask, Response, jsonify, request
import cv2
import threading
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import requests
import time

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="KASbyY8hQkoVk1tKqmCc"
)

results = None


def send_notification(device_id):
    url = "http://localhost:3333/notifications/{device_id}"
    data = {"deviceId": device_id}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Notificação enviada para o dispositivo {device_id}")
        else:
            print(
                "Erro ao enviar notificação: {response.status_code} - {response.text}")
    except Exception as e:
        print("Erro ao enviar notificação: {e}")


def process_frame(frame):
    global results
    results = CLIENT.infer(frame, model_id="closeddoors/1")
    # results = model(frame)
    # print(results)


@app.route('/camera_feed')
def camera_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Captura da câmera
        if not cap.isOpened():
            print("Erro ao acessar a câmera")
            return

        currentClass = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (320, 240))

            threading.Thread(target=process_frame, args=(frame,)).start()

            if results:
                predictions = results.get('predictions', [])
                for prediction in predictions:

                    if (prediction['class'] == 'Open' or prediction['class'] == 'Semi') and prediction['confidence'] >= 0.7:
                        print("Porta aberta!")
                        currentClass = 'Open'
                        break

                    if currentClass == 'Closed':

                        print("NOTIFICAÇÃO A CAMINHO!\n")
                        # Mandar para id do dispositivo
                        # '''
                        # /notifications/{deviceId}
                        # '''
                        # send_notification(receive_id)
                        currentClass = None
                        break

                    if prediction['class'] == 'Closed' and prediction['confidence'] >= 0.8:

                        currentClass = 'Closed'
                        print("Porta fechada!\n")

                        # Tempo passado pelo usuário
                        time.sleep(5)
                        break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/receive_id', methods=['POST'])
def receive_id():
    global received_id
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'ID não fornecido'}), 400

    received_id = data['id']
    print(f"ID recebido: {received_id}")
    return jsonify({'message': 'ID recebido com sucesso', 'id': received_id}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)
