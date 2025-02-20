import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

db_port = os.getenv("DB_PORT")
if not db_port or not db_port.isdigit():
    db_port = 3306
else:
    db_port = int(db_port)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": db_port,
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "database": os.getenv("DB_NAME")
}

def get_db_connection():
    """Returns a new database connection."""
    return mysql.connector.connect(**DB_CONFIG)

def close_connection(conn):
    """Closes the database connection."""
    conn.close()