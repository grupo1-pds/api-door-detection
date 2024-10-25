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

# model = YOLO('best.pt')

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
results = None


def send_notification(device_id):
    url = f"http://localhost:3333/notifications/{device_id}"
    data = {"deviceId": device_id}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Notificação enviada para o dispositivo {device_id}")
        else:
            print(f"Erro ao enviar notificação: {
                  response.status_code} - {response.text}")
    except Exception as e:
        print(f"Erro ao enviar notificação: {e}")


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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = cv2.resize(frame, (320, 240))

            threading.Thread(target=process_frame, args=(frame,)).start()

            if results:
                predictions = results.get('predictions', [])
                currentClass = None
                for prediction in predictions:

                    if (prediction['class'] == 'Open' or prediction['class'] == 'Semi') and prediction['confidence'] >= 0.7:
                        print("Porta aberta!")
                        currentClass = 'Open'
                        continue

                    if currentClass == 'Closed':

                        print("NOTIFICAÇÃO A CAMINHO!\n")
                        # Mandar para id do dispositivo
                        '''
                        /notifications/{deviceId}

                        '''
                        send_notification(receive_id)
                        break

                        currentClass = 'Closed'
                        print(currentClass)
                        print("Porta fechada!\n")

                        # Tempo passado pelo usuário
                        time.sleep(5)
                        continue

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
