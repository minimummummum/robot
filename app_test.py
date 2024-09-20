from flask import Flask, render_template, Response
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
from tracking_action import Tracking_Action

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

host = '0.0.0.0'
image_socket_port = 8080
receive_socket_port = 8081
send_socket_port = 8082
flask_port = 5000
global_send = None
servo_state = [0,0,0,0,0,0,0,0] # 오른어깨 왼어깨 오른손 왼손 오른다리 왼다리 오른발 왼발
servo_default = [-90,-90,45,45,-5,0,0,0,0]
send_lock = threading.Lock()
receive_lock = threading.Lock()
servo_lock = threading.Lock()
receive_sensor = [0,0,0,0,0,0]
balance_sw = False
img = []
# 이미지 데이터를 처리하는 소켓 서버 함수
def image_socket_server():
    global img
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, image_socket_port))
    server_socket.listen(1)
    marker_start = b'\xff\xd8'
    marker_end = b'\xff\xd9'
    value_command = b''
    captureimg = None
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
                    socketio.emit('image_data', {'data': encoded_data}, namespace='/video_feed')

                    decoded_data = base64.b64decode(encoded_data)
                    nparr = np.frombuffer(decoded_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    #cv2.imshow('img', img)

                    cv2.waitKey(1)
                    value_command = value_command[index_end+len(marker_end):]
                    sw = 0
        client_socket.close()

# 텍스트 데이터를 받는 소켓 서버 함수
def receive_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, receive_socket_port))
    server_socket.listen(1)
    global receive_sensor
    received_data = b''
    while True:
        client_socket, addr = server_socket.accept()
        print(f"{addr}에서 수신 소켓 연결됨")
        while True:
            data = client_socket.recv(128)
            if not data:
                print("텍스트 데이터 없음")
                break
            received_data += data
            start_index = received_data.find(b'[')
            end_index = received_data.find(b']')
            while start_index != -1 and end_index != -1:
                data_chunk = received_data[start_index+1:end_index].decode('utf-8')
                received_data = received_data[end_index + 1:]
                count = 0
                for i in data_chunk.split(', '):
                    try:
                        int(i)
                        with receive_lock:
                            receive_sensor[count] = int(i)
                    except: pass
                    count+=1
                start_index = received_data.find(b'[')
                end_index = received_data.find(b']')
        client_socket.close()

# 텍스트를 받아서 전역 변수에 저장하는 함수
def set_text_to_send(text):
    global global_send
    with send_lock:
        global_send = text
# 텍스트를 보내는 소켓 서버 함수
def send_socket_server():
    global global_send
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, send_socket_port))
    server_socket.listen(1)
    while True:
        client_socket, addr = server_socket.accept()
        print(f"{addr}에서 송신 소켓 연결됨")
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
def incline():
    receive_sensor_average1 = 0
    receive_sensor_average2 = 0
    for i in range(10):
        receive_sensor_average1 += receive_sensor[3]
        time.sleep(0.05)
        if i == 9: receive_sensor_average1 /= i+1
    for i in range(10):
        receive_sensor_average2 += receive_sensor[3]
        time.sleep(0.05)
        if i == 9: receive_sensor_average2 /= i+1
    return receive_sensor_average2 - receive_sensor_average1
def balance():
    global servo_state
    while balance_sw:
        receive_sensor_average = 0
        add6 = 3
        add7= 3
        state6_max = 15
        state7_max = 15
        average_cut = 5
        for i in range(2):
            receive_sensor_average += receive_sensor[3]
            time.sleep(0.1)
            if i == 1: receive_sensor_average /= i+1
        #print("기울기 평균",receive_sensor_average)
        with servo_lock:
            if -average_cut<receive_sensor_average<average_cut:
                pass 
            elif -average_cut>=receive_sensor_average:
                if servo_state[6] < 0:
                    add6*=2
                    state6_max*=2
                servo_state[6] = min(servo_state[6]+add6, state6_max)
                if servo_state[7] >= 0:
                    add7*=2
                    state7_max*=2
                servo_state[7] = min(servo_state[7]+add7, state7_max)
                set_text_to_send(str(servo_state))
                print(servo_state)
            elif receive_sensor_average>=average_cut:
                if servo_state[6] <= 0:
                    add6*=2
                    state6_max*=2
                servo_state[6] = max(servo_state[6]-add6, -state6_max)
                if servo_state[7] > 0:
                    add7*=2
                    state7_max*=2
                servo_state[7] = max(servo_state[7]-add7, -state7_max)
                set_text_to_send(str(servo_state))
                print(servo_state)
            else:
                pass

            
servo_state = [0,0,0,0,0,0,0,0] # 오른어깨 왼어깨 오른손 왼손 오른다리 왼다리 오른발 왼발
def set_angle(servo, angle):
    global servo_state
    with servo_lock:
        if servo == "right_arm":
            servo_state[0] = angle
        elif servo == "left_arm":
            servo_state[1] = angle
        elif servo == "right_hand":
            servo_state[2] = angle
        elif servo == "left_hand":
            servo_state[3] = angle
        elif servo == "right_leg":
            servo_state[4] = angle
        elif servo == "left_leg":
            servo_state[5] = angle
        elif servo == "right_foot":
            servo_state[6] = angle
        elif servo == "left_foot":
            servo_state[7] = angle
        else:
            pass
        set_text_to_send(str(servo_state))
foot = 10
leg = 20
hadn = 10
arm = 45
angle_time = 0.25
def move_left():
#    왼발 들기
    set_angle("left_foot", foot*2)
    set_angle("right_foot", foot*1.5)
    time.sleep(angle_time)
    
#    왼발 돌리기
    set_angle("right_leg", leg)
    time.sleep(angle_time)
    
#     왼발 내리기
    set_angle("right_foot", foot)
    set_angle("left_foot", foot)
    time.sleep(angle_time)
    set_angle("right_foot", 0)
    set_angle("left_foot", 0)
    time.sleep(angle_time)

#    오른발 들기
    set_angle("right_foot", -foot*2)
    set_angle("left_foot", -foot*1.5)
    time.sleep(angle_time)
    
#     오른발 돌리기
    set_angle("right_leg", 0)
    time.sleep(angle_time)

#    오른발 내리기
    set_angle("left_foot", -foot)
    set_angle("right_foot", -foot)
    time.sleep(angle_time)
    set_angle("left_foot", 0)
    set_angle("right_foot", 0)
    time.sleep(angle_time)

def move_right():
    #    오른발 들기
    set_angle("right_foot", -foot*2)
    set_angle("left_foot", -foot*1.5)
    time.sleep(angle_time)
    
#     오른발 돌리기
    set_angle("right_leg", leg)
    time.sleep(angle_time)

#    오른발 내리기
    set_angle("left_foot", -foot)
    set_angle("right_foot", -foot)
    time.sleep(angle_time)
    set_angle("left_foot", 0)
    set_angle("right_foot", 0)
    time.sleep(angle_time)
    
#    왼발 들기
    set_angle("left_foot", foot*2)
    set_angle("right_foot", foot*1.5)
    time.sleep(angle_time)
    
#    왼발 돌리기
    set_angle("right_leg", 0)
    time.sleep(angle_time)
    
#     왼발 내리기
    set_angle("right_foot", foot)
    set_angle("left_foot", foot)
    time.sleep(angle_time)
    set_angle("right_foot", 0)
    set_angle("left_foot", 0)
    time.sleep(angle_time)


tr_a = Tracking_Action()
def tracking():
    global img
    move_action(tr_a.tracking(img)) # action 리턴

def move_action(action):
    if action == 1:
        move_right
    elif action == 2:
        move_left
    else:
        pass

def main():
    while True:  
        #tracking()
        time.sleep(10)

# Flask 앱 라우트
@app.route('/')
def index():
    return render_template('index.html')

# Flask 앱의 비디오 피드 라우트
@app.route('/video_feed')
def video_feed():
    return Response()
# html에서 message 받기
@socketio.on('message', namespace='/video_feed')
def handle_message(message):
    event_type = message.get('event_type')
    if event_type == 'image_data':
        handle_image_data(message)
    elif event_type == 'custom_data':
        handle_custom_data(message)
# message 해석 후 서보모터 제어
def handle_image_data(message):
    print(f"Received image data: {message}")
def handle_custom_data(message):
    global balance_sw
    if message.get('data') == '앞으로 나란히':
        print("앞으로 나란히!")
        with servo_lock:
            servo_state[0] = 0
            servo_state[1] = 0
            set_text_to_send(str(servo_state))
    elif message.get('data') == 'balance':
        if balance_sw: balance_sw = False
        else:
            balance_sw = True
            balance()
        
    print(f"Received custom data: {message}")
if __name__ == '__main__':
    # 각각의 소켓 서버를 다른 스레드에서 실행
    image_thread = threading.Thread(target=image_socket_server)
    receive_thread = threading.Thread(target=receive_socket_server)
    send_thread = threading.Thread(target=send_socket_server)
    main = threading.Thread(target=main)
    image_thread.start()
    receive_thread.start()
    send_thread.start()
    main.start()
    # Flask 앱을 SocketIO 서버로 실행
    socketio.run(app, host=host, port=flask_port, debug=False)
