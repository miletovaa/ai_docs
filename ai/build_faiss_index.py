import faiss
import numpy as np
import openai
import mysql.connector
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

from db.connector import get_db_connection
from common.setup_logs import setup_logger

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
project_prefix = os.getenv("PROJECT_NAME")
vector_dimension = int(os.getenv("VECTOR_DIMENSION"))
embedding_model = os.getenv("EMBEDDING_MODEL")

logger = setup_logger()

conn = get_db_connection()

index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')

def text_to_vector(text: str):
    if not text.strip():
        return None
    try:
        client = OpenAI()
        embedding = client.embeddings.create(
            model=embedding_model,
            input=text
        ).data[0].embedding

        input_tokens = len(text.split())
        cost_per_token = 0.100 / 1000000
        cost = input_tokens * cost_per_token
        logger.info(f"FAISS ada-002. Number of tokens: {input_tokens}, cost: {cost:.6f}$")

        return {"embedding": embedding, "cost": cost}
    except Exception as e:
        logger.error(f"[OpenAI] Error in embedding: {e}")
        return None

def build_faiss_index():
    logger.info("Start Building Faiss index.")

    try:
        cursor = conn.cursor()
        logger.info(f"Successful DB connection")
    except mysql.connector.Error as err:
        logger.error(f"DB connection error: {err}")
        exit(1)

    try:
        cursor.execute(f"""SELECT * FROM {project_prefix}_files""")
        project_files = cursor.fetchall()
        logger.info(f"Got {len(project_files)} records from {project_prefix}_files to index.")
    except mysql.connector.Error as err:
        logger.error(f"SQL Error: {err}")
        cursor.close()
        conn.close()
        exit(1)

    if os.path.exists(index_path):
        logger.info("Найден существующий Faiss индекс, пытаемся загрузить.")
        existing_index = faiss.read_index(index_path)
        if not isinstance(existing_index, faiss.IndexIDMap2):
            logger.info("Индекс не IDMap2, оборачиваем в IndexIDMap2.")
            index = faiss.IndexIDMap2(existing_index)
        else:
            index = existing_index
        logger.info("Существующий индекс загружен.")
    else:
        logger.info("No FAISS index file. Creating a new one.")
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(vector_dimension))

    for record in project_files:
        record_id = record[0]
        filename = record[1]
        file_content = record[2]
        file_category = record[3]

        text = f"FILE NAME: {filename}. FILE CATEGORY: {file_category}. FILE CONTENT: {file_content}."

        embedding = text_to_vector(text)["embedding"]

        if embedding:
            faiss_vector = np.array(embedding, dtype='float32').reshape(1, -1)
            faiss_id = np.array([record_id], dtype='int64')
            index.add_with_ids(faiss_vector, faiss_id)
            logger.info(f"Добавлен вектор с ID={record_id} в Faiss индекс.")
            print(f"Added vector with ID={record_id} to Faiss index.")
        else:
            print(f"Embedding not found for record ID={record_id}, skipping.")
            logger.warning(f"Эмбеддинг не получен для записи ID={record_id}, пропускаем.")

    faiss.write_index(index, index_path)
    logger.info(f"Faiss индекс успешно сохранён по пути: {index_path}")

    cursor.close()
    conn.close()
    logger.info("Завершено построение Faiss индекса.")

if __name__ == "__main__":
    try:
        build_faiss_index()
    except Exception as e:
        logger.error(f"Unexpectable error: {e}", exc_info=True)
