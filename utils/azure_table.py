import os
import uuid
from datetime import datetime, timezone
from azure.data.tables import TableServiceClient
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Load connection string and table names from environment variables
TABLE_CONN_STR = os.getenv("AZURE_TABLE_CONNECTION_STRING")
PEDESTRIAN_TABLE_NAME = os.getenv("AZURE_PEDESTRIAN_TABLE", "AuthorizedPedestrians")
VEHICLE_TABLE_NAME = os.getenv("AZURE_VEHICLE_TABLE", "AuthorizedVehicles")
PEDESTRIAN_LOGS_TABLE_NAME = os.getenv("AZURE_PEDESTRIAN_LOGS_TABLE", "PedestrianLogs")
VEHICLE_LOGS_TABLE_NAME = os.getenv("AZURE_VEHICLE_LOGS_TABLE", "VehicleLogs")

# Initialize table service client
table_service = TableServiceClient.from_connection_string(TABLE_CONN_STR)

# Create individual table clients
pedestrian_table_client = table_service.get_table_client(PEDESTRIAN_TABLE_NAME)
vehicle_table_client = table_service.get_table_client(VEHICLE_TABLE_NAME)
pedestrian_logs_table_client = table_service.get_table_client(PEDESTRIAN_LOGS_TABLE_NAME)
vehicle_logs_table_client = table_service.get_table_client(VEHICLE_LOGS_TABLE_NAME)

# Attempt to create tables if they don't already exist
for client in [pedestrian_table_client, vehicle_table_client, pedestrian_logs_table_client, vehicle_logs_table_client]:
    try:
        client.create_table()  # Create table
    except Exception:
        pass  # Ignore if table already exists

# Fetch all authorized pedestrian embeddings
def get_all_authorized_pedestrians():
    entities = pedestrian_table_client.query_entities("PartitionKey eq 'Pedestrian'")
    return [{"face_embedding": eval(entity["Embedding"])} for entity in entities]

# Fetch all authorized vehicle data and embeddings
def get_all_authorized_vehicles():
    entities = vehicle_table_client.query_entities("PartitionKey eq 'Vehicle'")
    return [
        {
            "face_embedding": eval(entity["Embedding"]),
            "license_plate": entity["RowKey"],
            "car_make": entity["CarMake"],
            "car_model": entity["CarModel"]
        }
        for entity in entities
    ]

# Log pedestrian verification attempt to Azure Table
def log_pedestrian_verification_attempt(log_data: dict):
    log_data["PartitionKey"] = "PedestrianLog"
    log_data["RowKey"] = str(uuid.uuid4())  # Generate unique ID
    log_data["LoggedAtUTC"] = datetime.now(timezone.utc)  # Add current UTC timestamp
    pedestrian_logs_table_client.upsert_entity(log_data)  # Insert or update entity

# Log vehicle verification attempt to Azure Table
def log_vehicle_verification_attempt(log_data: dict):
    log_data["PartitionKey"] = "VehicleLog"
    log_data["RowKey"] = str(uuid.uuid4())  # Generate unique ID
    log_data["LoggedAtUTC"] = datetime.now(timezone.utc)  # Add current UTC timestamp
    vehicle_logs_table_client.upsert_entity(log_data)  # Insert or update entity

# Retrieve all pedestrian log entries
def fetch_pedestrian_logs():
    return list(pedestrian_logs_table_client.list_entities())

# Retrieve all vehicle log entries
def fetch_vehicle_logs():
    return list(vehicle_logs_table_client.list_entities())
