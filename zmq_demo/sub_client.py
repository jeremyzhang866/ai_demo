import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:6000")  # 连接发布端口

# 订阅一个或多个 topic
socket.setsockopt_string(zmq.SUBSCRIBE, "weather")  # 你也可以换成 "sports" 或空字符串订阅所有

print("[Subscriber] Waiting for messages...")

while True:
    message = socket.recv_string()
    print(f"[Subscriber] Got: {message}")
