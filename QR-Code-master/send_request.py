# import socket

# # server_ip = "192.168.0.103"
# class ConnectRPI:
    
#     # client_socket: None | object
    
#     def __init__(self,
#                  server_ip = "10.42.0.1",
#                  server_port = 8085) -> None:
        
#         self.server_ip = server_ip
#         self.server_port = server_port

        

#     # @classmethod
#     def creat_client_object(self):
#         return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
#     def make_connection(self):
#         # create a socket object
#         # connect to the server
#         self.client_socket.connect((self.server_ip, self.server_port))
#         print(f"Connected to {self.server_ip}:{self.server_port}")
#         # return self.client_socket

#     def send_data(self, result):
#         self.client_socket = self.creat_client_object()
#         self.make_connection()  
#         self.client_socket.send(result.encode("utf-8"))
        
#         # if result == "close":
#         # self.close_connection()
#         # self.client_socket.close()

#     def recieve_data(self):
#         # receive the processed result from the server
#         processed_result = self.client_socket.recv(2048).decode("utf-8")
#         print(f"Processed result: {processed_result}")
#         return processed_result
    
#     # close the connection
#     def close_connection(self):
#         self.client_socket = self.creat_client_object()
#         self.make_connection()
#         self.send_data("close")
#         self.client_socket.close()

# # connect = ConnectRPI()
# # for i in range(144):
# #     connect.send_data(f"False&True: {i}")
# #     connect.recieve_data()
#     # connect.client_socket.close()

# # connect.make_connection()
# # connect.close_connection()

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
                # self.client_socket.settimeout(1/10)
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