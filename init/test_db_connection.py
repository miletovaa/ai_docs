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

def test_db_connection():
    try:
        print(f"⏳ Подключаемся к MySQL серверу: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        print("✅ Успешное подключение к MySQL!")
        print("📋 Список таблиц:")
        for table in tables:
            print(f" - {table[0]}")

        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"❌ Ошибка подключения к MySQL: {err}")

if __name__ == "__main__":
    test_db_connection()