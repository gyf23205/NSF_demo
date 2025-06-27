# server.py
import socket
import time
def server():
    host = '127.0.0.1'  # Use '0.0.0.0' to accept connections from other machines
    port = 8888

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()

    clients = []  # Track all client addresses
    print("Server waiting for connection...")
    while len(clients) < 1:  # Wait for at least one client to connect
        conn, addr = s.accept()
        print("Connected by", addr)
        clients.append((conn, addr))  # Store the address
    print("All client connected")

    # # Send a boolean variable
    # my_bool = True
    # conn.sendall(str(my_bool).encode())
    first_conn, first_addr = clients[0]
    first_conn.setblocking(False)  # Make socket non-blocking
    slogon = "Welcome to the server!"
    for i in range(20):
        time.sleep(1)
        try:
            data = first_conn.recv(1024).decode()
            if data:
                slogon = data
        except BlockingIOError:
            # No data received, continue with current slogon
            pass
        print(slogon)
    
    # Send a tuple
    my_tuple = (1, 'hello', True)
    conn.sendall(str(my_tuple).encode())

    conn.close()

if __name__ == "__main__":
    server()
