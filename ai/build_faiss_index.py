import faiss
import numpy as np
import openai
import mysql.connector
import os
from dotenv import load_dotenv
from openai import OpenAI

from db.connector import get_db_connection
from common.setup_logs import setup_logger

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
project_prefix = os.getenv("PROJECT_NAME")
vector_dimension = int(os.getenv("VECTOR_DIMENSION"))
embedding_model = os.getenv("EMBEDDING_MODEL")

logger = setup_logger(__name__)
conn = get_db_connection()

index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')

def text_to_vector(text: str):
    if not text.strip():
        logger.warning("Received empty text for vectorization.")
        return None
    try:
        logger.info(f"Generating embedding for text: {text[:30]}...")
        client = OpenAI()
        embedding = client.embeddings.create(
            model=embedding_model,
            input=text
        ).data[0].embedding

        input_tokens = len(text.split())
        cost_per_token = 0.100 / 1000000
        cost = input_tokens * cost_per_token
        logger.info(f"Embedding generated. Tokens: {input_tokens}, Cost: {cost:.6f}$")

        return {"embedding": embedding, "cost": cost}
    except Exception as e:
        logger.error(f"[OpenAI] Error in embedding: {e}", exc_info=True)
        return None

def build_faiss_index():
    logger.info("Starting FAISS index build process.")

    try:
        cursor = conn.cursor()
        logger.info("Connected to the database successfully.")
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}", exc_info=True)
        exit(1)

    try:
        cursor.execute(f"SELECT * FROM {project_prefix}_files")
        project_files = cursor.fetchall()
        logger.info(f"Retrieved {len(project_files)} records from {project_prefix}_files for indexing.")
    except mysql.connector.Error as err:
        logger.error(f"SQL query error: {err}", exc_info=True)
        cursor.close()
        conn.close()
        exit(1)

    if os.path.exists(index_path):
        logger.info("Existing FAISS index found, attempting to load.")
        existing_index = faiss.read_index(index_path)
        if not isinstance(existing_index, faiss.IndexIDMap2):
            logger.info("Index is not IDMap2, wrapping in IndexIDMap2.")
            index = faiss.IndexIDMap2(existing_index)
        else:
            index = existing_index
        logger.info("Existing index loaded successfully.")
    else:
        logger.info("No FAISS index file found. Creating a new one.")
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(vector_dimension))

    for record in project_files:
        record_id = record[0]
        filename = record[1]
        file_content = record[2]
        file_category = record[3]

        text = f"FILE NAME: {filename}. FILE CATEGORY: {file_category}. FILE CONTENT: {file_content}."

        embedding = text_to_vector(text)
        if embedding:
            faiss_vector = np.array(embedding["embedding"], dtype='float32').reshape(1, -1)
            faiss_id = np.array([record_id], dtype='int64')
            index.add_with_ids(faiss_vector, faiss_id)
            logger.info(f"Added vector with ID={record_id} to FAISS index.")
        else:
            logger.warning(f"Embedding not generated for record ID={record_id}, skipping.")

    faiss.write_index(index, index_path)
    logger.info(f"FAISS index successfully saved at: {index_path}")

    cursor.close()
    conn.close()
    logger.info("FAISS index build process completed.")

if __name__ == "__main__":
    try:
        build_faiss_index()
    except Exception as e:
        logger.error(f"Unexpected error during FAISS index build: {e}", exc_info=True)