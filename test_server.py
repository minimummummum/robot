import socket

# 서버 주소와 포트 설정
HOST = '127.0.0.1'  # 로컬 호스트
PORT = 65422      # 포트 번호

# 소켓 생성
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))  # 주소 바인딩
    s.listen()             # 클라이언트 연결 대기
    print(f"서버가 {HOST}:{PORT}에서 대기 중입니다...")
    
    conn, addr = s.accept()  # 클라이언트 연결 수락
    with conn:
        print(f"연결된 클라이언트: {addr}")
        while True:
            data = conn.recv(1024)  # 클라이언트로부터 데이터 수신
            if not data:
                break  # 데이터가 없으면 종료
            print(f"받은 데이터: {data.decode()}")
            conn.sendall(data)  # 받은 데이터를 다시 클라이언트에 전송
