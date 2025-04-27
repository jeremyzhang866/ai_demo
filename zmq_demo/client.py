import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)  # 请求端（request）
socket.connect("tcp://localhost:5555")  # 连接到服务端

socket.send_string("World")
reply = socket.recv_string()
print(f"Received reply: {reply}")
