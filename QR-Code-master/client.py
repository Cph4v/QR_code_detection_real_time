import requests

def send_data(url, data):
    response = requests.post(url, json={"data": data})
    print("Response from server:", response.json())

def get_data(url):
    response = requests.get(url)
    print("Data received from server:", response.json())

# if __name__ == "__main__":
# server_url = "http://192.168.0.104:8001"
# server_url = "http://192.168.43.103:8000"
server_url = "http://192.168.2.22:8000"
# server_url = "http://192.168.2.22:8000"
# server_url = "http://rapidqr.ddns.me:8000"
test_data = {"123456789134": "123456789131"}
# test_data = {"key": "value", "int": 1, "bool": True}

# Sending data to the server
# send_data(f"{server_url}/send_green/", test_data)

# Getting data from the server
get_data(f"{server_url}/get_database/")
