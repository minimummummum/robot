import socket

# 서버 주소와 포트 설정
HOST = '127.0.0.1'  # 서버의 IP 주소
PORT = 65433       # 포트 번호

# 소켓 생성
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))  # 서버에 연결
    message = "안녕하세요, 서버!"  # 서버에 보낼 메시지
    s.sendall(message.encode())  # 메시지 전송
    data = s.recv(1024)          # 서버로부터 데이터 수신

print(f"서버로부터 받은 데이터: {data.decode()}")
