
import socket

class ConnectRPI:
    def __init__(self, server_ip="10.42.0.1", server_port=8085) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None

    def connect(self):
        if self.client_socket is not None:
            self.close()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((self.server_ip, self.server_port))
            print(f"Connected to {self.server_ip}:{self.server_port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            self.close()

    def send_data(self, data):
        # if not self.client_socket:
        #     print("No connection established. Attempting to connect...")
        self.connect()
        if self.client_socket:
            try:
                self.client_socket.send(data.encode("utf-8"))
            except Exception as e:
                print(f"Failed to send data: {e}")
                self.close()

    def receive_data(self):
        try:
            return self.client_socket.recv(2048).decode("utf-8")
        except Exception as e:
            print(f"Failed to receive data: {e}")
            return None

    def close(self):
        if self.client_socket:
            try:
                self.client_socket.close()
                print("Connection closed.")
            finally:
                self.client_socket = None