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
    api_key="P23WgD6oL94DxdKRk7lJ"
)

results = None


def send_notification(device_id):
    url = "http://safeelder.life:8080/notifications/{device_id}"
    data = {"type": "door"}
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
                prediction = results.get('predictions', [0])

                if prediction['class'] in ['Open', 'Semi'] and prediction['confidence'] > 0.7:
                    print("=== Porta aberta! ===")
                    current_class = 'Open'
                    closed_start_time = None  # Reseta o controle de tempo de porta fechada
                    notification_sent = False  # Permite o envio de notificação no futuro

                elif prediction['class'] == 'Closed' and prediction['confidence'] > 0.7:
                    if closed_start_time is None:
                        closed_start_time = time.time()  # Inicia o contador de tempo de porta fechada

                    # Porta fechada por 10 segundos (tempo estipulado pelo usuário)
                    if time.time() - closed_start_time >= int(received_time):
                        if not notification_sent:
                            print("NOTIFICAÇÃO A CAMINHO!\n")
                            # Enviar notificação aqui
                            notification_sent = True
                            current_class = 'Notified'
                            break
                    else:
                        print(
                            "||| Porta fechada! |||\n||| Aguardando 10 segundos... |||")

                else:
                    # Para outros estados, resetar variáveis de controle
                    closed_start_time = None
                    notification_sent = False

            if current_class == 'Notified':
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/receive_id', methods=['POST'])
def receive_id():
    global received_id
    global received_time
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'ID não fornecido'}), 400

    received_id = data['id']
    received_time = data['time']
    received_time = 10
    print(f"ID recebido: {received_id}")
    return jsonify({'message': 'ID recebido com sucesso', 'id': received_id}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)
