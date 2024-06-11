from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import uvicorn

app = FastAPI()

class Item(BaseModel):
    data: dict

@app.post("/send_green/")
async def receive_data(item: Item):
    try:
        # Function to handle numpy.int64 serialization problem in JSON
        def convert(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError("Object of type 'np.int64' is not JSON serializable")

        # Function to modify the incoming JSON data
        def convert_laptop_to_Green(json_data):
            data_dict = json.loads(json_data)
            modified_data = {shelf: "Green-" + laptop for shelf, laptop in data_dict.items()}
            return json.dumps(modified_data)

        # Function to update the local database with new data
        def update_local_database(json_data):
            try:
                with open("allocated_laptops_QRs.txt", "r") as file:
                    local_database = json.load(file)
            except FileNotFoundError:
                local_database = {}

            new_data = json.loads(json_data)
            local_database.update(new_data)

            with open("allocated_laptops_QRs.txt", "w") as file:
                json.dump(local_database, file, default=convert)

            return local_database

        json_data = json.dumps(item.data, default=convert)
        json_data = convert_laptop_to_Green(json_data)

        with open("received_data.txt", "w") as file:
            file.write(json_data)

        updated_database = update_local_database(json_data)
        print(updated_database)  # Printing to console (or use logging in production)

        return {"status": "data received and saved successfully", "updated_database": updated_database}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_database/")
async def send_data():
    total_data = {}
    try:
        with open("allocated_laptops_QRs.txt", "r") as file:
            json_data = json.load(file)
        data = [
            {"index": index, "shelf_id": shelf, "product_id": laptop}
            for index, (shelf, laptop) in enumerate(json_data.items())
        ]
        total_data['data'] = data
        total_data['detail'] = "Data sent successfully"
        total_data['status'] = 200
    except FileNotFoundError:
        total_data['data'] = None
        total_data['detail'] = "Data not found"
        total_data['status'] = 404
    except json.JSONDecodeError:
        total_data['data'] = None
        total_data['detail'] = "Error decoding JSON data"
        total_data['status'] = 500
    except Exception as e:
        total_data['data'] = None
        total_data['detail'] = f"Unexpected error: {str(e)}"
        total_data['status'] = 500
    return total_data


# import requests
# def send_data(url, data):
#     response = requests.post(url, json={"data": data})
#     print("Response from server:", response.json())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # send_data("192.168.0.104:8001/send-data/", test_data)
