import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)  # 响应端（reply）
socket.bind("tcp://*:5555")  # 监听 5555 端口

print("Server started on port 5555...")

while True:
    message = socket.recv_string()
    print(f"Received request: {message}")
    socket.send_string(f"Hello, {message}")
