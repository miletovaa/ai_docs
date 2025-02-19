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

project_prexif = os.getenv("PROJECT_NAME")

CREATE_FILES_TABLE = f"""
CREATE TABLE `{project_prexif}_files` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    path TEXT NOT NULL,
    content LONGTEXT NOT NULL,
    file_category ENUM('frontend', 'backend', 'mssql', 'unknown') NOT NULL DEFAULT 'unknown'
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

CREATE_FILE_BLAME_TABLE = f"""
CREATE TABLE `{project_prexif}_files_blame` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    path TEXT NOT NULL,
    line_number INT NOT NULL,
    author VARCHAR(255) NOT NULL,
    commit_hash VARCHAR(40) NOT NULL,
    commit_date DATETIME NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES `{project_prexif}_files`(id) ON DELETE CASCADE
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

CREATE_HISTORY_TABLE = """
CREATE TABLE github_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    commit_hash VARCHAR(40) NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

def table_exists(cursor, table_name):
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    return cursor.fetchone() is not None

def create_tables():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        if not table_exists(cursor, f"{project_prexif}_files"):
            print(f"🛠️ Creating table: {project_prexif}_files")
            cursor.execute(CREATE_FILES_TABLE)
        else:
            print(f"✅ Table {project_prexif}_files already exists.")

        if not table_exists(cursor, f"{project_prexif}_files_blame"):
            print(f"🛠️ Creating table: {project_prexif}_files_blame")
            cursor.execute(CREATE_FILE_BLAME_TABLE)
        else:
            print(f"✅ Table {project_prexif}_files_blame already exists.")

        if not table_exists(cursor, "github_history"):
            print("🛠️ Creating table: github_history")
            cursor.execute(CREATE_HISTORY_TABLE)
        else:
            print("✅ Table github_history already exists.")

        conn.commit()
        cursor.close()
        conn.close()

        print("🚀 Database setup completed successfully!")

    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")

if __name__ == "__main__":
    create_tables()