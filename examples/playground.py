# client.py
import socket
import time

def client():
    host = '127.0.0.1'  # IP of the server
    port = 8888

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    time.sleep(5)  # Wait for the server to be ready
    s.sendall(b"Hello from client!")
    data = s.recv(1024)

    print("Received from server:", data.decode())
    s.close()

if __name__ == "__main__":
    client()
