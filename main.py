import threading
import time
from flask import Flask, Response, jsonify, request
import cv2
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import requests

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="4J9Ptdi6V5dvW7xlOeoJ"
)

results = None


def send_notification(device_id):
    url = f"http://safeelder.life:8080/notifications/{device_id}?notificationType=bathroom"
    print(url)
    data = {"type": "door"}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Notificação enviada para o dispositivo {device_id}")
        else:
            print(
                f"Erro ao enviar notificação: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Erro ao enviar notificação: {e}")


def process_frame(frame):
    global results
    results = CLIENT.infer(frame, model_id="closeddoors/1")
    # results = model(frame)
    # print(results)


@app.route('/camera_feed_door', methods=['POST'])
def camera_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Captura da câmera
        if not cap.isOpened():
            print(f"Erro ao acessar a câmera")
            return

        closed_start_time = None  # Momento em que a porta foi detectada como fechada
        notification_sent = False  # Controle de envio de notificações
        current_class = None  # Estado atual

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (320, 240))

            threading.Thread(target=process_frame, args=(frame,)).start()

            if results:
                predictions = results.get('predictions', [])

                for prediction in predictions:
                    if prediction['class'] in ['Open', 'Semi']:
                        current_class = 'Open'
                        closed_start_time = None  # Reseta o controle de tempo de porta fechada
                        notification_sent = False  # Permite o envio de notificação no futuro

                    elif prediction['class'] == 'Closed':
                        if closed_start_time is None:
                            closed_start_time = time.time()  # Inicia o contador de tempo de porta fechada

                        # Porta fechada por X segundos (tempo estipulado pelo usuário)
                        if time.time() - closed_start_time >= (int(received_time)*60):
                            if not notification_sent:
                                print("NOTIFICAÇÃO A CAMINHO!\n")
                                send_notification(received_id)
                                notification_sent = True
                                current_class = 'Notified'
                                break
                        # else:
                           # print("||| Porta fechada! |||\n||| Aguardando... |||")

                    else:
                        # Para outros estados, resetar variáveis de controle
                        closed_start_time = None
                        notification_sent = False

            if current_class == 'Notified':
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/receive_id_door', methods=['POST'])
def receive_id():
    global received_id
    global received_time
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'ID não fornecido'}), 400

    received_id = data['id']
    received_time = data['time']
    print(f"ID recebido: {received_id}")
    return jsonify({'message': 'ID recebido com sucesso', 'id': received_id}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)
