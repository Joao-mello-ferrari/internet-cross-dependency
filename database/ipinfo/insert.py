import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
from tqdm import tqdm

# Database connection details
DB_NAME = "geo_db"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
CSV_FILE = "sorted_csv_ipv4.csv"  # Change to your actual file path
CHUNK_SIZE = 100000  # Number of rows per batch

dtype_dict = {
    "start_ip": "string",
    "end_ip": "string",
    "join_key": "string",
    "city": "string",
    "region": "string",
    "country": "string",
    "latitude": "float64",
    "longitude": "float64",
    "postal_code": "string",  # Ensure postal codes are read as strings
    "timezone": "string",
    "start_ip_int": "int64",
    "end_ip_int": "int64",
}

def create_table():
    """Creates the table if it does not exist."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ip_data (
            id SERIAL PRIMARY KEY,  -- Auto-incremented integer ID
            start_ip TEXT NOT NULL,
            end_ip TEXT NOT NULL,
            join_key TEXT,
            city TEXT,
            region TEXT,
            country TEXT,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            postal_code TEXT,
            timezone TEXT,
            start_ip_int BIGINT NOT NULL,
            end_ip_int BIGINT NOT NULL
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()

def insert_data():
    """Reads CSV in chunks and inserts into PostgreSQL."""
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    reader = pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE, dtype=dtype_dict)

    c = 0
    for chunk in tqdm(reader, desc="Inserting chunks"):
        c += 1
        #if c > 10: return
        chunk.to_sql("ip_data", engine, if_exists="append", index=False)
        print(f"Inserted {len(chunk)} rows")

    
if __name__ == "__main__":
    create_table()
    insert_data()

