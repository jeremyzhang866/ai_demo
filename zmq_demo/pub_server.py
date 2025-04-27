import zmq
import time
import random

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:6000")  # 发布端口

topics = ["sports", "weather", "news"]

print("[Publisher] Start publishing...")

while True:
    topic = random.choice(topics)
    msg = f"{topic} {time.time()}"
    print(f"[Publisher] {msg}")
    socket.send_string(msg)
    time.sleep(1)
