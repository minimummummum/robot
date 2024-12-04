from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import socket
import threading
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from robot_action import Robot_Action
from simulation import Simulation
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

host = '0.0.0.0'
image_socket_port = 8080
receive_socket_port = 8081
send_socket_port = 8082
flask_port = 5000
global_send = None
servo_state = [-90,-90,45,45,0,0,0,0,0] # 오른어깨 왼어깨 오른손 왼손 오른다리 왼다리 오른발 왼발
servo_default = [-90,-90,45,45,0,0,0,0,0]
send_lock = threading.Lock()
servo_lock = threading.Lock()
receive_sensor = [0,0,0,0,0,0]
balance_sw = False
img = []
ranker = {}
with open("data/ranker.json", "r", encoding="utf-8") as f:
    ranker = json.load(f)

encoded_data = 0
# 클라이언트에서 이미지를 받는 함수
def image_socket_server():
    global img
    global encoded_data
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, image_socket_port))
    server_socket.listen(1)
    marker_start = b'\xff\xd8'
    marker_end = b'\xff\xd9'
    value_command = b''
    sw = 0
    while True:
        client_socket, addr = server_socket.accept()
        print(f"{addr}에서 이미지 소켓 연결됨")
        while True:
            data = client_socket.recv(4096)
            if not data:
                print("이미지 데이터 없음")
                break
            value_command += data
            if sw == 0:
                index_start = value_command.find(marker_start)
                if index_start != -1:
                    sw = 1
            if sw == 1:
                index_end = value_command.find(marker_end)
                if index_end != -1:
                    encoded_data = base64.b64encode(value_command[index_start:index_end] + marker_end).decode('utf-8')
                    decoded_data = base64.b64decode(encoded_data)
                    nparr = np.frombuffer(decoded_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    flip_img = cv2.flip(img, 1)
                    _, buffer = cv2.imencode('.jpg', flip_img)
                    flip_img_base64 = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('image_data', {'data': flip_img_base64}, namespace='/video_feed')

                    

                    cv2.waitKey(1)
                    value_command = value_command[index_end+len(marker_end):]
                    sw = 0
        client_socket.close()
# 텍스트를 받아서 전역 변수에 저장하는 함수
def set_text_to_send(text):
    global global_send
    with send_lock:
        global_send = text
# 텍스트를 클라이언트에 보내는 함수
def send_socket_server():
    global global_send
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, send_socket_port))
    server_socket.listen(1)
    while True:
        client_socket, addr = server_socket.accept()
        print(f"{addr}에서 송신 소켓 연결됨")
        set_text_to_send(str(servo_state))
        while True:
            with send_lock:
                try:
                    if global_send:
                        client_socket.sendall(global_send.encode('utf-8'))
                        global_send = None
                except Exception as e:
                    print(f"전송 실패: {e}")
                    break
        client_socket.close()
# 학습 된 DQN을 이용하여 액션 선택
rb_a = Robot_Action()
def action_select():
    global img
    data = [0, 0, 0, 0]
    data[0] = servo_state[0] + 90
    data[1] = servo_state[1] + 90
    data[2] = servo_state[2] - 45
    data[3] = servo_state[3] - 45
    action = rb_a.action(img, data)
    return action
# main
def main():
    global servo_state
    action_sw = False
    action_message = ""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, receive_socket_port))
    server_socket.listen(1)
    received_data = b''
    client_socket, addr = server_socket.accept()
    print(f"{addr}에서 수신 소켓 연결됨")
    while True:
        action = action_select()
        if action == 1:
            action_sw = True
            action_message = "[right]"
        elif action == 2:
            action_sw = True
            action_message = "[left]"
        elif action == "wait":
            with servo_lock:
                set_text_to_send(str(servo_default))
            continue
        elif action:
            with servo_lock:
                servo_state[0] = action[0] - 90
                servo_state[1] = action[1] - 90
                servo_state[2] = action[2] + 45
                servo_state[3] = action[3] + 45
                servo_state[0] = np.clip(servo_state[0], -90, 90)
                servo_state[1] = np.clip(servo_state[1], -90, 90)
                servo_state[2] = np.clip(servo_state[2], 30, 95)
                servo_state[3] = np.clip(servo_state[3], 30, 95)
                set_text_to_send(str(servo_state))
        else:
            print(action)
        if action_sw:
            set_text_to_send(action_message)
        while action_sw:
            data = client_socket.recv(128)
            received_data = data
            if b'[' in received_data and b']' in received_data:
                start_index = received_data.find(b'[')
                end_index = received_data.find(b']')
                data_chunk = received_data[start_index+1:end_index].decode('utf-8')
                if data_chunk == "complete":
                    action_sw = False
# 랭킹 모드
simulation = Simulation()
def game(name):
    global servo_state
    average = 0
    set_time = 15
    data = [random.choice(range(0, 181, 10)), random.choice(range(0, 181, 10)),
            random.choice(range(-10, 51, 10)), random.choice(range(-10, 51, 10))]
    if data[0] > 90:
        data[0] -= abs(data[2])
    else:
        data[0] += abs(data[2])
    if data[1] > 90:
        data[1] -= abs(data[3])
    else:
        data[1] += abs(data[3])
    simulation.simulation_run(data)
    start_time = time.time()
    last_count = None
    socketio.emit('speak', {'data': "시작"}, namespace='/video_feed')
    socketio.emit('play_audio', {'data': '/static/audio/example.mp3'}, namespace='/video_feed')
    while True:
        stop_time = int(time.time() - start_time)
        if stop_time >= set_time:
            break
        count = str(set_time - stop_time)
        if count != last_count:
            last_count = count
            socketio.emit('text', {'data': last_count}, namespace='/video_feed')
        data2 = [servo_state[0]+90, servo_state[1]+90, servo_state[2]-45, servo_state[3]-45]
        distance = [math.sqrt((a - b) ** 2) for a, b in zip(data, data2)]
        match_rate = [(1 - distance[0] / 180) * 100,
               (1 - distance[1] / 180) * 100,
               (1 - distance[2] / 60) * 100,
               (1 - distance[3] / 60) * 100]
        average = int(sum(match_rate)/len(match_rate))
        #print("일치율: ", average,"%", data, data2)
        socketio.emit('match_rate', {'data': int(average)}, namespace='/video_feed')
        time.sleep(0.1)
    img = encoded_data
    simulation.simulation_run([0, 0, 0, 0])
    socketio.emit('match_rate', {'data': 0}, namespace='/video_feed')
    ranking(average, name, img)

def ranking(score, name, img):
    global ranker
    value = [score, name, img]
    text = "랭킹 등극 실패!"
    print(len(ranker))
    if len(ranker) < 9:
        rankkey = ""
        i = 1
        while True:
            rankkey = "rank"+str(i)
            if rankkey not in ranker:
                ranker[rankkey] = value
                break
            i+=1
        ranker = dict(sorted(ranker.items(), key=lambda item: item[1][0], reverse=True))
        ranker_items = list(ranker.items())
        key_index = next(index for index, (key, value) in enumerate(ranker_items) if key == rankkey)
        text = f"랭킹 {key_index + 1}위 등극!"
    else:
        min_key = min(ranker, key=lambda k: ranker[k][0])
        min_score = ranker[min_key][0]
        if score > min_score:
            ranker[min_key] = value
            ranker = dict(sorted(ranker.items(), key=lambda item: item[1][0], reverse=True))
            ranker_items = list(ranker.items())
            min_index = next(index for index, (key, value) in enumerate(ranker_items) if key == min_key)
            text = f"랭킹 {min_index + 1}위 등극!"
    socketio.emit('text', {'data': text}, namespace='/video_feed')
    socketio.emit('ranking', {'data': ranker}, namespace='/video_feed')
    socketio.emit('speak', {'data': "종료, "+text}, namespace='/video_feed')
    socketio.emit('stop_audio', {}, namespace='/video_feed')
    with open("data/ranker.json", "w", encoding="utf-8") as f:
        json.dump(ranker, f, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    audio_url = '/static/audio/example.mp3'
    return render_template('index.html', audio_url=audio_url)

@socketio.on('connect', namespace='/video_feed')
def handle_connect():
    socketio.emit('ranking', {'data': ranker}, namespace='/video_feed')
    
# html에서 message 받기
@socketio.on('message', namespace='/video_feed')
def handle_message(message):
    event_type = message.get('event_type')
    if event_type == 'image_data':
        handle_image_data(message)
    elif event_type == 'custom_data':
        handle_custom_data(message)

def handle_image_data(message):
    print(f"Received image data: {message}")
def handle_custom_data(message):
    if message.get('data') == 'Start':
        print(message.get('data2'))
        game(message.get('data2'))
if __name__ == '__main__':
    image_thread = threading.Thread(target=image_socket_server)
    send_thread = threading.Thread(target=send_socket_server)
    main = threading.Thread(target=main)
    image_thread.start()
    send_thread.start()
    main.start()
    socketio.run(app, host=host, port=flask_port, debug=False)

