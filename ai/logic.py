import asyncio
import datetime
from datetime import datetime
import os
import faiss
import numpy as np

from typing import Dict, Any
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import sys
import json

from db.connector import get_db_connection
from common.setup_logs import setup_logger
from ai.openai_methods import text_to_vector, send_openai_request, calculate_cost_of_request


load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
project_prefix = os.getenv("PROJECT_NAME")
vector_dimension = int(os.getenv("VECTOR_DIMENSION"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.json")

logger = setup_logger()

executor = ThreadPoolExecutor()

sessions: Dict[str, Any] = {}

class InitData(BaseModel):
    action: str


global faiss_index
faiss_index = None

# Processed user prompt that is used in FAISS and OpenAI API
global prompt


print("Python executable:", sys.executable)
logger.info(f"sys.executable: {sys.executable}")
logger.info(f"Python path: {sys.path}")
logger.info(f"Environment PATH: {os.environ.get('PATH')}")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def get_prompt(keys):
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        data = prompts
        for key in keys:
            data = data[key]

        return data
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        return f"⚠️ Warning: Prompt for {' -> '.join(keys)} not found."


async def process_user_prompt(prompt):
    system_context_settings = get_prompt(["process_user_prompt", "system_context_settings"])
    assistent_context_settings = get_prompt(["process_user_prompt", "assistent_context_settings"])

    response = await send_openai_request(prompt, system_context_settings, assistent_context_settings)

    if not response:
        logger.error("OpenAI API returned an empty response.")
        return None, None

    cost = calculate_cost_of_request(response)

    return {"prompt": response, "cost": cost}


async def get_file_ids_with_faiss():
    top_k = 5 # Number of the project files to consider while creating the response

    global faiss_index

    if faiss_index is None:
        raise ValueError("FAISS индекс не загружен в память.")

    # Получаем данные из сессии
    # global dialogue_history
    global prompt
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
        raise ValueError("Error in getting the embedding from OpenAI API.")
    
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding, dtype='float32')
        
    elif embedding.dtype != 'float32':
        embedding = embedding.astype('float32')
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    elif embedding.ndim == 2 and embedding.shape[0] == 1:
        pass
    else:
        raise ValueError("Incorrect shape of the embedding array.")


    logger.info(f"Embedding sent to FAISS: {embedding}")

    distances, ids = faiss_index.search(embedding, top_k)

    logger.info(f"Ids: {ids}; distances: {distances} (get_file_ids_with_faiss())")

    return {"distances": distances.tolist(), "ids": ids.tolist()}

async def get_file_content_by_ids(ids):
    global conn

    if not ids:
        return []

    cursor = conn.cursor(dictionary=True)

    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    in_ids = ', '.join(map(str, ids))
    query = f"SELECT * FROM {project_prefix}_files WHERE id IN ({in_ids})"
    
    logger.info(f"Executing SQL Query: {query}")

    cursor.execute(query)
    records = cursor.fetchall()

    cursor.close()
    return records


async def get_final_user_response(files):
    if not files:
        logger.warning("No files found for the user prompt.")
        return "I'm sorry, I couldn't find any files related to your question."
    
    files_content_string = " ".join(record["content"] for record in files)
    
    system_context_settings = get_prompt(["get_final_user_response", "system_context_settings"])
    assistent_context_settings = get_prompt(["get_final_user_response", "assistent_context_settings"]).format(files_content_string)

    global prompt

    logger.info(f"Sending the final request to OpenAI API.")
    logger.info(prompt)

    response = await send_openai_request(prompt, system_context_settings, assistent_context_settings)

    if not response:
        logger.error("OpenAI API returned an empty response.")
        return None, None

    cost = calculate_cost_of_request(response)

    return {"answer": response, "cost": cost}


# TODO: not implemented yet, but is important. To be cleaned later.
def ocenka_distances_level(call_id, distances): # оценивает насколько вопрос относится к теме
    # session['distances'] = distances # насколько ответ схож с вопросом.
    logger.info(f"Посмотрим внутри функции ocenka_distances_level на distances - {distances}")
    
    
    #Допустим, у вас в индексе 3 вектора, и вы ищете 2 ближайших (top_k=2):
    distances, ids = faiss_index.search(embedding, 2)
    print(distances)
    # [[0.1, 0.5]]  # Расстояния до 2 ближайших векторов
    print(ids)
    # [[12, 45]]    # ID этих двух векторов

    # Если FAISS не находит подходящего вектора (например, для слишком малых данных в индексе), он вернёт -1:
    distances, ids = faiss_index.search(embedding, 2)
    print(distances)
    # [[0.1, inf]]
    print(ids)
    # [[12, -1]]
    # Расстояния дают представление о том, насколько найденные векторы (результаты) похожи на запрос. Если расстояние слишком велико (например, больше определённого порога), это может означать, что результат не релевантен.
    # Порог для релевантности
    threshold = 0.5
    
    # Фильтруем результаты по порогу
    relevant_results = [
        {"id": id, "distance": distance}
        for id, distance in zip(ids[0], distances[0])
        if distance <= threshold
    ]

    if not relevant_results:
        print("Нет релевантных результатов")
    else:
        print("Релевантные результаты:", relevant_results)

        # Пороговая фильтрация для улучшения производительности

        # Обрезаем результаты с расстоянием > 1.0
        filtered_ids = [id for id, distance in zip(ids[0], distances[0]) if distance <= 1.0]
        print("Фильтрованные ID:", filtered_ids)


if __name__ == "__main__":
    import asyncio

    async def main():
        while True:
            global faiss_index
            
            # Initialize the database connection
            global conn 
            conn = get_db_connection()

            # Checking if we have FAISS index in the memory 
            index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')

            if os.path.exists(index_path):
                faiss_info = faiss.read_index(index_path)
                logger.info(f"{project_prefix}_info.bin загружен в оперативку")
            else:
                faiss_info = faiss.IndexFlatL2(vector_dimension)
                logger.info(f"какая-та хуйня с загрузкой {project_prefix}_info.bin")
            faiss_index = faiss_info
            
            user_input = input("Enter your prompt (or type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("👋 Bye")
                break

            response = await process_user_prompt(user_input)

            global prompt
            prompt = response["prompt"]

            if response:
                print(f"🤖 Bot: {response}")

                ids_and_distances = await get_file_ids_with_faiss()

                ids = ids_and_distances.get("ids", [])
                print(f"🔍 Search Results: {ids}")

                files = await get_file_content_by_ids(ids)

                response = await get_final_user_response(files)

                print(f"🤖 Bot: {response["answer"]}")

    asyncio.run(main())