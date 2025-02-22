import os
import faiss
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any
from pydantic import BaseModel
from db.connector import get_db_connection
from common.setup_logs import setup_logger
from ai.openai_methods import text_to_vector, send_openai_request, calculate_cost_of_request

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
project_prefix = os.getenv("PROJECT_NAME")
vector_dimension = int(os.getenv("VECTOR_DIMENSION"))
default_option = os.getenv("DEFAULT_APP_OPTION")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.json")

logger = setup_logger()
conn = get_db_connection()
faiss_index = None  # FAISS инициализируется позже

def load_faiss_index():
    """Загружает FAISS индекс в память"""
    global faiss_index
    index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Файл FAISS индекса не найден: {index_path}")

    faiss_index = faiss.read_index(index_path)
    logger.info(f"FAISS индекс загружен из {index_path}")

def get_selected_option(telegram_id):
    cursor = conn.cursor(dictionary=True)

    query = f"SELECT option_id FROM users WHERE telegram_id = '{telegram_id}'"
    cursor.execute(query)
    record = cursor.fetchone()
    conn.commit()
    cursor.close()

    return record["option_id"]

def get_gpt_model(option=default_option):
    cursor = conn.cursor(dictionary=True)

    query = f"SELECT model_api FROM {project_prefix}_app_options WHERE id = '{option}'"
    cursor.execute(query)
    record = cursor.fetchone()
    conn.commit()
    cursor.close()

    return record["model_api"]

def get_prompt(keys, option=default_option):
    cursor = conn.cursor(dictionary=True)

    prompt_type = keys[0]
    role = keys[1]

    query = f"SELECT prompt_{prompt_type}_role_{role} FROM {project_prefix}_app_options WHERE id = '{option}'"

    cursor.execute(query)
    record = cursor.fetchone()
    conn.commit()
    cursor.close()

    if record[f"prompt_{prompt_type}_role_{role}"] is None:
        return ""
    else:
        return record[f"prompt_{prompt_type}_role_{role}"]

async def process_user_prompt(prompt, telegram_id):
    """Запрос в OpenAI для обработки промпта"""
    option = get_selected_option(telegram_id)

    system_context = get_prompt(["processing", "system"], option)
    assistant_context = get_prompt(["processing", "assistant"], option)

    response = await send_openai_request(prompt, system_context, assistant_context)

    if not response:
        logger.error("OpenAI API returned an empty response.")
        return {"prompt": None, "cost": None}

    cost = calculate_cost_of_request(response)
    logger.info("Cost of the request: %.6f$", cost)
    return {"prompt": response, "cost": cost}

async def get_file_ids_with_faiss(prompt):
    """Находит ID файлов через FAISS"""
    top_k = 5 # Number of the project files to consider while creating the response

    global faiss_index

    dialogue_history_string = ""

    # TODO:
    # if isinstance(dialogue_history, list) and dialogue_history:
    #     dialogue_history_string = ' '.join(dialogue_history)
        
    if len(dialogue_history_string) > 0:
        request_to_embed = (
            f'Question is: "{prompt}"\n, {dialogue_history_string}'
        )
    else:
        request_to_embed = f'Question is: "{prompt}"'
    embedding = text_to_vector(request_to_embed)["embedding"]

    if embedding is None:
        raise ValueError("Ошибка получения эмбеддинга.")

    embedding = np.array(embedding, dtype='float32').reshape(1, -1)
    distances, ids = faiss_index.search(embedding, top_k)

    return {"distances": distances.tolist(), "ids": ids.tolist()}

async def get_file_content_by_ids(ids):
    """Получает содержимое файлов по их ID"""
    if not ids:
        return []

    cursor = conn.cursor(dictionary=True)

    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    in_ids = ', '.join(map(str, ids))
    query = f"SELECT * FROM {project_prefix}_files WHERE id IN ({in_ids})"

    cursor.execute(query)
    records = cursor.fetchall()
    conn.commit()
    cursor.close()
    return records

async def get_final_user_response(prompt, telegram_id, files):
    """Генерирует окончательный ответ на основе найденных файлов"""
    if not files:
        return "Извините, не удалось найти релевантные файлы."

    files_content_string = " ".join(record["path"] + ": " + record["content"] + "; " for record in files)

    option = get_selected_option(telegram_id)

    gpt_model = get_gpt_model(option)

    system_context = get_prompt(["general", "system"], option)
    assistant_context = get_prompt(["general", "assistant"], option).format(files_content_string)

    response = await send_openai_request(prompt, system_context, assistant_context, gpt_model)

    cost = calculate_cost_of_request(response)
    logger.info("Cost of the request: %.6f$", cost)

    if not response:
        return "Ошибка обработки запроса AI."

    return response

async def run_pipeline(user_input, telegram_id):
    """Полный процесс обработки запроса"""
    global faiss_index

    if faiss_index is None:
        load_faiss_index()

    print("Processing user input...")

    processed = await process_user_prompt(user_input, telegram_id)
    prompt = processed["prompt"]

    print("Getting file IDs with FAISS...")

    # prompt = user_input
    
    ids_and_distances = await get_file_ids_with_faiss(prompt)
    ids = ids_and_distances.get("ids", [])

    print("Getting file content by IDs...")

    files = await get_file_content_by_ids(ids)
    final_response = await get_final_user_response(prompt, telegram_id, files)

    print("Final response:", final_response)

    return final_response