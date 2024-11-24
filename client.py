from network import WLAN, STA_IF
import time as utime
import usocket as socket
import json
import uasyncio as asyncio
from Arducam import *
from machine import Pin, PWM, I2C
from imu import MPU6050
import pickle
# 네트워크 초기화
SSID = 'mini' # 와이파이
PASSWORD = '16381638' # 비밀번호
wlan = WLAN(STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
max_wait = 10
while max_wait > 0:
    if wlan.status() < 0 or wlan.status() >= 3:
        break
    max_wait -= 1
    print('연결 대기중')
    utime.sleep(1)
if wlan.status() != 3:
    raise RuntimeError('네트워크 연결 실패')
else:
    print('네트워크 연결 성공')
    status = wlan.ifconfig()
    print('ip = ' + status[0])
ip = '192.168.0.43'
s0 = socket.socket()
s0.connect((ip, 8080)) # camera data
s1 = socket.socket()
s1.connect((ip, 8081)) # send data
s2 = socket.socket()
s2.connect((ip, 8082)) # recive data
# 카메라 초기화
mode = 0
start_capture = 0
stop_flag = 0
value_command = 0
flag_command = 0
mycam = ArducamClass(OV5642)
mycam.Camera_Detection()
mycam.Spi_Test()
mycam.Camera_Init()
mycam.Spi_write(ARDUCHIP_TIM, VSYNC_LEVEL_MASK)
utime.sleep(1)
mycam.clear_fifo_flag()
mycam.Spi_write(ARDUCHIP_FRAMES, 0x00) #0x00 1장
mycam.OV5642_set_JPEG_size(OV5642_320x240)
#mycam.OV5642_set_JPEG_size(OV5642_640x480)
camera_sw = True
count = 0
once_number = 4096
length = 0
buffer=bytearray(once_number)
# 가속도 센서 초기화
# SCL 19 SDA 18
i2c_sensor = I2C(1, sda=Pin(18), scl=Pin(19), freq=400000)
imu = MPU6050(i2c_sensor)
sensor_state = [0,0,0,0,0,0]
gyro_state = [0,0,0]
accel = imu.accel # 단위 g
gyro = imu.gyro # 단위 도/초
# 서보모터 초기화
time = 0.25
right_arm = PWM(Pin(6), 50)#오른팔
left_arm = PWM(Pin(14), 50)#왼팔
arm = 45 # -90 default
right_hand = PWM(Pin(7), 50)#오른손
left_hand = PWM(Pin(15), 50)#왼손
hand = 10 # 45 default
right_leg = PWM(Pin(10), 50)#오른다리
left_leg = PWM(Pin(17), 50)#왼다리
leg = 5 # 0 default
right_foot = PWM(Pin(20), 50)#오른발
left_foot = PWM(Pin(12), 50)#왼발
foot = 10 # 0 default
servo_data = [right_arm,left_arm,right_hand,left_hand,right_leg,left_leg,right_foot,left_foot]
async def set_angle(servo, angle):
    if servo == left_arm: angle *= -1
    elif servo == left_hand: angle *= -1
    a = int(((((angle + 90) * 2) / 180) + 0.5) / 20 * 65535)
    servo.duty_u16(a)
# 데이터 얻기
async def get_sensor_data():
    global camera_sw
    global count
    global once_number
    global length
    global buffer
    if camera_sw:
        mycam.flush_fifo();
        mycam.clear_fifo_flag();
        mycam.start_capture();
        while not mycam.get_bit(ARDUCHIP_TRIG,CAP_DONE_MASK):
            await asyncio.sleep_ms(1)
        count = 0
        once_number = 4096
        length = mycam.read_fifo_length()
        if length <= once_number:
            once_number = length
        buffer=bytearray(once_number)
        #print(length, "length")
        mycam.SPI_CS_LOW()
        mycam.set_fifo_burst()
        camera_sw = False
    count += once_number
    if count>=length:
        count=once_number-(count-length)
        buffer2=bytearray(count)
        mycam.spi.readinto(buffer2)
        mycam.SPI_CS_HIGH()
        mycam.clear_fifo_flag()
        camera_sw = True
        return buffer2
    mycam.spi.readinto(buffer)
    return buffer
async def main0():
    global s0
    global s1
    while True:
        sensor_data = await get_sensor_data()
        s0.sendall(sensor_data)
        await asyncio.sleep_ms(1)
async def move():
    await set_angle(left_foot, 0)
    await set_angle(right_foot, 0)
    await set_angle(left_leg, 0)
    await set_angle(right_leg, 0)
    await asyncio.sleep(time*10)
    while True:
        await move_right()
        await move_left()
        await move_left()
        await move_right()
async def move_left():
#    왼발 들기
    await set_angle(left_foot, foot*2)
    await set_angle(right_foot, foot*1.5)
    await asyncio.sleep(time)
#    왼발 돌리기
    await set_angle(right_leg, leg)
    await asyncio.sleep(time)
#     왼발 내리기
    await set_angle(right_foot, foot)
    await set_angle(left_foot, foot)
    await asyncio.sleep(time)
    await set_angle(right_foot, 0)
    await set_angle(left_foot, 0)
    await asyncio.sleep(time)
#    오른발 들기
    await set_angle(right_foot, -foot*2)
    await set_angle(left_foot, -foot*1.5)
    await asyncio.sleep(time)
#     오른발 돌리기
    await set_angle(right_leg, 0)
    await asyncio.sleep(time)
#    오른발 내리기
    await set_angle(left_foot, -foot)
    await set_angle(right_foot, -foot)
    await asyncio.sleep(time)
    await set_angle(left_foot, 0)
    await set_angle(right_foot, 0)
    await asyncio.sleep(time)
async def move_right():
#    오른발 들기
    await set_angle(right_foot, -foot*2)
    await set_angle(left_foot, -foot*1.5)
    await asyncio.sleep(time)
#     오른발 돌리기
    await set_angle(right_leg, leg)
    await asyncio.sleep(time)
#    오른발 내리기
    await set_angle(left_foot, -foot)
    await set_angle(right_foot, -foot)
    await asyncio.sleep(time)
    await set_angle(left_foot, 0)
    await set_angle(right_foot, 0)
    await asyncio.sleep(time)
#    왼발 들기
    await set_angle(left_foot, foot*2)
    await set_angle(right_foot, foot*1.5)
    await asyncio.sleep(time)
#    왼발 돌리기
    await set_angle(right_leg, 0)
    await asyncio.sleep(time)
#     왼발 내리기
    await set_angle(right_foot, foot)
    await set_angle(left_foot, foot)
    await asyncio.sleep(time)
    await set_angle(right_foot, 0)
    await set_angle(left_foot, 0)
    await asyncio.sleep(time)
async def main2(): #@@@2024-09-21
    # 시나리오.
    # 1. 로봇 시야에서 사람이 가쪽에 있을 경우, robotsensor 전송 중지, 특수 플래그(action 왼쪽 or 오른쪽) 전송 후 대기
    # 2. 여기서 특수한 플래그 받았을 경우, 그에 따라 main1() 실행 후 종료되면 s1.sendall로 종료 플래그 전송
    # 3. 서버에서 종료 플래그 받았을 때, 왼쪽 혹은 오른쪽에 있으면 다시 특수 플래그 전송 후 대기
    # 4. 2와 동일
    # 5. 서버에서 종료 플래그 받았을 때, 가운데에 있으면 다시 robotsensor 전송
    # 즉, 클라이언트에서는 특수 플래그를 받았을 경우 main1() 실행 후, 종료 플래그 전송만 하면 됨. 따로 대기 할 필요 없음 어짜피 데이터는 안 올 거니깐.
    global s2
    global servo_data
    received_data = b''
    
    while True:
        s2.setblocking(False)
        try:
            data = s2.recv(128)
            received_data += data
            start_index = received_data.find(b'[')
            end_index = received_data.find(b']')
            while start_index != -1 and end_index != -1:
                data_chunk = received_data[start_index+1:end_index].decode('utf-8')
                received_data = received_data[end_index + 1:]
                start_index = received_data.find(b'[')
                end_index = received_data.find(b']')
                print("받은 데이터:", data_chunk)
                if data_chunk == "right":
                    await move_right()
                    s1.sendall("[complete]".encode())
                elif data_chunk == "left":
                    await move_left()
                    s1.sendall("[complete]".encode())
                else:
                    servo_count = 0
                    for i in data_chunk.split(', '):
                        try:
                            int(i)
                            await set_angle(servo_data[servo_count], int(i))
                        except: pass
                        servo_count += 1
        except OSError as e:
            err = e.errno
            if err == 11 or err == 35:
                data = 0
            else:
                print(f"에러 발생: {e}")
        finally:
            s2.setblocking(True)
            await asyncio.sleep_ms(100)

loop = asyncio.get_event_loop()
loop.create_task(main0())
loop.create_task(main2())
loop.run_forever()




