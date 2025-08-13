# tracker/unity.py
import socket, json

class UdpSender:
    def __init__(self, ip, port):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict):
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
