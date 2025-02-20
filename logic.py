import asyncio
import aiohttp
import json
import datetime
from datetime import datetime
import os
import re
import sqlite3
import mysql.connector
import faiss
import numpy as np
import math

from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import openai
from openai import OpenAI
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import sys

from db.connector import get_db_connection


load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
project_prefix = os.getenv("PROJECT_NAME")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()


logging.basicConfig(
    filename='logs/logic.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


executor = ThreadPoolExecutor()


sessions: Dict[str, Any] = {}

class InitData(BaseModel):
    action: str


global faiss_index

faiss_index = None
embedding_model = "text-embedding-ada-002"
vector_dimension = 1536  # Example for the model 'ada-002'


print("Python executable:", sys.executable)
logging.info(f"sys.executable: {sys.executable}")
logging.info(f"Python path: {sys.path}")
logging.info(f"Environment PATH: {os.environ.get('PATH')}")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def startup_event():
    
    global faiss_index
    
    # Initialize the database connection
    global conn 
    conn = get_db_connection()

    # Checking if we have FAISS index in the memory 
    index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')

    if os.path.exists(index_path):
        faiss_info = faiss.read_index(index_path)
        logging.info(f"{project_prefix}_info.bin загружен в оперативку")
    else:
        faiss_info = faiss.IndexFlatL2(vector_dimension)
        logging.info(f"какая-та хуйня с загрузкой {project_prefix}_info.bin")
    faiss_index = faiss_info


@app.on_event("shutdown")
async def shutdown_event():
    # Close the database connection on app shutdown
    conn.close()


async def send_openai_request(prompt, system_context, assistant_context, model="gpt-4o-mini"):
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": system_context},
        {"role": "assistant", "content": assistant_context},
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": model,
        "messages": messages
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:
                logging.info(f"📥 OpenAI API Response Status: {response.status}")

                if response.status == 200:
                    response_data = await response.json()
                    return response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                else:
                    error_message = await response.text()
                    logging.error(f"OpenAI API Error: {response.status} - {error_message}")
                    return ""

        except Exception as e:
            logging.error(f"Exception in send_openai_request: {e}")
            return ""
        
async def calculate_cost_of_request(response):
    if 'usage' in response:
        input_tokens = response['usage'].get('prompt_tokens', 0)
        output_tokens = response['usage'].get('completion_tokens', 0)

        input_cost_per_token = 0.150 / 1000000  # $0.150 / 1M input tokens
        output_cost_per_token = 0.600 / 1000000  # $0.600 / 1M output tokens

        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost = input_cost + output_cost  # Итоговая стоимость
        return total_cost
    else:
        logging.warning("Информация о токенах отсутствует в ответе API.")
        return 0


async def process_user_prompt(prompt):
    # TODO: settings prompts
    system_context_settings = ""
    assistent_context_settings = ""

    response = await send_openai_request(prompt, system_context_settings, assistent_context_settings)

    if not response:
        logging.error("OpenAI API returned an empty response.")
        return None, None

    cost = calculate_cost_of_request(response)

    return response, cost


async def get_faiss_ids_and_distances_from_faiss():
    # ВОЗВРАЩАЕТ ids и distances
    # ВОЗВРАЩАЕТ embedding_ids - ЭТО НАЙДЕННАЯ ГРУППА СМЫСЛОВ ДЛЯ SQL и дистанцию до искомого distances_level

    #TIMING
    timeing_log.info(f"Апплодисменты! Мы в async def get_faiss_ids_and_distances_from_faiss(call_id, session). \n искать в get_faiss_ids_and_distances_from_faiss не на английском, а на языке БЗ, а она не всегда на английском.")
    # может изменю, но сейчас здесь отправлю на эмбединги только text_in_lang_knows_base - это текст вопроса в окошко юзеру 

    # global уже определена эта глобальная переменная выше
    top_k = 3 #скоько подходящих сторк из БД/Faiss будем брать 
    if faiss_index is None:
        raise ValueError("FAISS индекс не загружен в память.")


    # Получаем данные из сессии
    stenograf_dialog_dkll = session.get('stenograf_dialog_dkll', [])
    preview_question_dkll = session.get('preview_question_dkll', '')
    # Преобразуем stenograf_dialog_dkll в строку, если список не пустой
    if isinstance(stenograf_dialog_dkll, list) and stenograf_dialog_dkll:
        stenograf_dialog_dkll = ' '.join(stenograf_dialog_dkll)
        

    # Объединяем две строки
    #    question_ids_and_distances = preview_question_dkll + ' ' + stenograf_dialog_dkll if stenograf_dialog_dkll else preview_question_dkll

    #ОН ПЛОХО ПОДГОТАВЛИВАЕТ ЗАПРОС. Т К ДОЛЖЕН БЫТЬ ЯВНО ВЫДЕЛЕН preview_question_dkll В ГЛАВНОЕ, А stenograf_dialog_dkll КАК ВТОРОСТЕПЕННОЕ Т Е ДОПОЛНИТЕЛЬНОЕ. ...И ЕЩЁ И НА ЯЗЫКЕ БЗ. !О, КАКИЕ-ТО НУЖНЫЕ ФРАЗОЧКИ Я МОГУ ПРЕПОДГОТАВЛИВАТЬ В ХЕЛЛО ТЕКСТ, КОТОРЫЙ ОДИН РАЗ ПЕРЕВОДИТ НА ЯЗЫК БЗ. И значения itsquestion и usedstenograf БУДУТ ПОДСТАВЛЕНЫ СЮДА НА ЯЗЫКЕ БЗ


    # Формируем запрос с явным выделением preview_question_dkll как основного текста и stenograf_dialog_dkll как дополнительного
    #    itsquestion = f'Вопрос такой: '
    itsquestion = f' \n Klausimas toks: '
    #    usedstenograf = f', а при ответе на этот вопрос дополнительно используй информацию из текста уже произошедшего диалога: '
    usedstenograf = f', \n atsakant į šį klausimą, papildomai naudokite jau vykusio dialogo informaciją: \n '
    if stenograf_dialog_dkll:
        question_ids_and_distances = (
            f'{itsquestion}"{preview_question_dkll}"{usedstenograf}{stenograf_dialog_dkll}'
        )
    else:
        question_ids_and_distances = f'Вопрос такой: "{preview_question_dkll}"'


    text_in_lang_knows_base = question_ids_and_distances

    #    TF-IDF - Инструмент для генерирования ключевых слов сплошного текста 

    #TIMING
    embedding = text_in_lang_knows_base_to_vector(call_id, session, text_in_lang_knows_base)
    #        logging.info(f"Такой вот эмбединг: {embedding}")
    #TIMING
    logging.info("обращаемся к text_in_lang_knows_base_to_vector.")
    # Преобразовываем query в вектор (embedding) через OpenAI
        
    if embedding is None:
        raise ValueError("Эмбендинги не отдал text_in_lang_knows_base_to_vector... не удалось.")
    
    if not isinstance(embedding, np.ndarray):
    # Если это список или другой объект, преобразуем в numpy.ndarray
        embedding = np.array(embedding, dtype='float32')
        
    elif embedding.dtype != 'float32':
        # Если тип не float32, приводим к float32
        embedding = embedding.astype('float32')
    if embedding.ndim == 1:
        # Если одномерный массив, преобразуем в двумерный
        embedding = embedding.reshape(1, -1)
    elif embedding.ndim == 2 and embedding.shape[0] == 1:
        # Формат уже правильный
        pass
    else:
        # Если массив не соответствует ожидаемому формату, выбрасываем ошибку
        raise ValueError("Эмбеддинг должен быть двумерным массивом с одной строкой [1, vector_dimension]...., ЧЕМ ОН НЕ ЯВЛЯЕТСЯ")

    #    return embedding
    # Преобразование в numpy массив
    #    embedding_np = np.array(embedding, dtype='float32').reshape(1, -1)

    logging.info(f"Такой вот эмбединг от правляем в FAISS: {embedding}")

    distances, ids = faiss_index.search(embedding, top_k)

    logging.info(f"Такой вот ids: {ids} и вот такой  distances: {distances}  в get_faiss_ids_and_distances_from_faiss()")
    
    session['ids'] = ids
    session['distances'] = distances

    await get_text_from_mysql(call_id, session, ids)

    return {"distances": distances.tolist(), "ids": ids.tolist()}


async def get_text_from_mysql(call_id, session, ids): # ЗДЕСЬ ЗАПУСК general_chatgpt_api(call_id, session) ... ПЕРЕД ЭТИМ ВОЗВРАЩАЕТ ids_sql - выбРАННЫЕ из mysql тексты строк с заданными через эмбеддинги номерами 
    session = sessions.get(call_id)
    if not session:
        logging.error(f"Сессия с call_id {call_id} не найдена.")
        return
    logging.info(f"------- Посмотрим внутри функции get_text_from_mysql на ids - {ids}")
    try:
        # Преобразуем ids в строку
        ids_str = str(ids)
        # Ищем все числа в строке (регулярное выражение для поиска цифр)
        id_list = re.findall(r'\d+', ids_str)
        logging.info(f"Извлечённые числовые id: {id_list}")

        # Проверяем, что есть числа
        if not id_list:
            logging.warning(f"Не найдено чисел в ids для call_id {call_id}.")
            session['ids_sql'] = ""
            return

    
        # Подключаемся к базе данных
        # Загрузка параметров из JSON для подключения к SQL
        config_path = '/var/www/botfyl.fyi/mysql_conn.json'
        with open(config_path, 'r') as file:
            config = json.load(file)
        # Подключение к базе данных
        if config["type"] == "sqlite":
            conn = sqlite3.connect(config["database"])
            print("Успешное подключение к SQLite")
            logging.info(f"подключилось к MySQL")
        elif config["type"] == "mysql":
            conn = mysql.connector.connect(
                host=config["host"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config["database"]
            )
            print("Успешное подключение к MySQL")
            logging.info(f"подключилося оно к MySQL")
        else:
            raise ValueError("Неподдерживаемый тип базы данных")
        
        #Это надо переделать через ассинхронную функцию
        ## Асинхронное взаимодействие с базой данных
        #ids_sql = await fetch_texts_from_db(ids)  # Предположим, что fetch_texts_from_db - асинхронная функция
        #session['ids_sql'] = ids_sql.strip()
        # Переменная для хранения текстов
        ids_sql = ""
        query = f"SELECT * FROM faiss_info WHERE id IN ({','.join(['%s'] * len(id_list))})"
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(map(int, id_list)))
            for result in cursor.fetchall():
                record_text = "\n".join(f"{desc[0]}: {value}" for desc, value in zip(cursor.description, result))
                ids_sql += record_text + "\n\n"

        conn.close()
        #TIMING
        timeing_log.info(f"Завершение подключения к локальной БД в get_text_from_mysql.")
    
        session['ids_sql'] = ids_sql.strip()

        #TIMING
        timeing_log.info(f"Если вы видите эту запись, ...то! Вы внутри функции get_text_from_mysql и достигнут результат get_faiss_ids_and_distances_from_faiss, который заключался в том, чтобы добыть ids_sql - это найденные тексты из БЗ для использования их в general_chatgpt_api.")
        logging.info(f"Полученные данные для call_id {call_id}: ... ") #{ids_sql}

        #надо не тоьько в строку преобразовать , а почистить от лишнего, то есть не отсавлять field_05: None, field_06: None, link_to_download_document: , main_image_link: , image_list: если в них ничего нет и т д
        #        general_chatgpt_api(call_id, session)


        #ВОТ ЭТО НЕ ПОНЯТНО ЗАЧЕМ ЗДЕСЬ В АССИНХРОНЕ ЗАПУСКАТЬ ФУНКЦИЮ general_chatgpt_api, И ВООБЩЕ ПОЧЕМУ/ЗАЧЕМ ОНА АССИНХРОННА? ОНА ДОЛЖНА ВЫПОЛНЯТЬСЯ ЧЁТКО В ОПРЕДЕЛЁННЫЙ МОМЕНТ В ОБЩЕЙ ЦЕПОЧКЕ СОБЫТИЙ И НИКОМУ ОНА НЕ МОЖЕТ УСТУПИТЬ СВОЁ МЕСТО И ЕЁ НЕЛЬЗЯ ПРОПУСТИТЬ ВПЕРЁД, ЧТОБЫ ОНА ВЫПОЛНИЛА СВОЮ ФУНКЦИЮ ПЕРЕД КЕМ-ТО...


        ## ВЫЗЫВАЕМ ФУНКЦИЮ  
        timeing_log.info(f"Запуск general_chatgpt_api(call_id, session) внутри get_text_from_mysql.")
    
    

        # Вызов асинхронной функции general_chatgpt_api 
        timeing_log.info(f"Запуск general_chatgpt_api(call_id, session) внутри get_text_from_mysql.")
        asyncio.create_task (general_chatgpt_api(call_id, session))

        
        #TIMING
        timeing_log.info(f"Завершение/конец asyncio.run_coroutine_threadsafe(general_chatgpt_api(call_id, session) внутри get_text_from_mysql.")
        
    except Exception as e:
        logging.error(f"Ошибка при получении данных из MySQL: {e}", exc_info=True)
        session['ids_sql'] = ""


def text_in_lang_knows_base_to_vector(call_id, session, text_in_lang_knows_base): #ВОЗВРАЩАЕТ embedding ТОМУ КТО ЕМУ ДАЁТ text_in_lang_knows_base 
    #    Эмбедингинизатор ada v2  $0.100 / 1M tokens,  $0.050 / 1M tokens in 24 ours
    if not text_in_lang_knows_base.strip():
        return None
    try:
        client = OpenAI()
        
        
        embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input = text_in_lang_knows_base
        ).data[0].embedding
        

        
        logging.info(f"Такой вот эмбединг вопроса сделали из text_in_lang_knows_base в text_in_lang_knows_base_to_vector: ...") #{embedding}
        
    # КАССА ЧЕК
        # Подсчёт количества токенов
        input_tokens = len(text_in_lang_knows_base.split())  # Примерная оценка количества токенов
        cost_per_token = 0.100 / 1000000  # Стоимость за один токен
        cost = input_tokens * cost_per_token
        logging.info(f"За токены ada-002: {input_tokens}, стоимость: {cost:.6f}$")
    # сохраняем в сессию расходы
        session['balance_money_price_ada02_question'] = cost
        return embedding  # Вернуть вектор
        
    except Exception as e:
        logging.error(f"[OpenAI] Ошибка при получении эмбеддинга: {e}")
        return None
        
    #КОД БОЛЕЕ КАЧЕСТВЕННОЙ ЧИСТКИ ТЕКСТА
    #            import re
    #            def remove_html(text):
    #                return re.sub(r"<.*?>", "", text)  # Убираем все HTML-теги
    #
    #            text_in_lang_knows_base = " ".join(
    #                remove_html(str(getattr(record, field, "") or "").strip())  # Убираем HTML и пробелы
    #                for field in ["poiasnenie", "name_document", "glava", "statia", "primechanie", "dopolnenie"]
    #            ).strip()
    #ЕЩЁ ТАМ ЧТО-ТО ПРО ЗАГЛАВНЫЕ БУКВЫ БЫЛО...     
        
# TODO: not implemented yet, but is important. To be cleaned later.
def ocenka_distances_level(call_id, distances): # оценивает насколько вопрос относится к теме
    # session['distances'] = distances # насколько ответ схож с вопросом.
    logging.info(f"Посмотрим внутри функции ocenka_distances_level на distances - {distances}")
    
    
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


async def general_chatgpt_api(call_id, session):
    session['t_start_gotovit_prompt'] = get_current_time()

    # ввести понятие ВЕС ИНФОРМАЦИИ на основании которой генерируется запрос.
    MAX_RETRIES = 3  # Максимальное количество повторов при ошибке
    RETRY_DELAY = 0.5  # Задержка между повторами (в секундах)

    preview_question_dkll = session.get('preview_question_dkll')
    static_user_information = session.get('static_user_information', "")

    role_system_general_chatgpt = session.get('role_system_general_chatgpt')
    role_assistant_general_chatgpt = session.get('role_assistant_general_chatgpt')
    
    lang_native = session.get('lang_native')
    lang_knows_base = session.get('lang_knows_base')
    
    ids_sql = session.get('ids_sql', [])
    logging.info(f"Так выглядит ids_sql достаный из сессии в блоке/функции general_chatgpt_api:  {ids_sql}")
    
    stenograf_dialog_dkll = session.get('stenograf_dialog_dkll', "")
    
    context_dialog_dkll = session.get('context_dialog_dkll', "")

    answer_length = 80
    
    prompt = preview_question_dkll
    if not isinstance(prompt, str) or not prompt.strip():
        logging.error("Некорректный или пустой prompt для GPT API.")
        return ""

    #СОБИРАЕМ ПЕРЕМЕННУЮ messages, ЧТОБЫ ОТПРАВИТЬ ЕЁ В data
    messages = [
        {   
            "role": "system", # лучше писать на английском
            "content": (
                f"{role_system_general_chatgpt}"
            ).format(answer_length=answer_length, lang_native=lang_native, lang_knows_base=lang_knows_base)
        },
        
#        {   
#            "role": "system",
#            "content": role_system_general_chatgpt.format(answer_length=answer_length, lang_native=lang_native, lang_knows_base=lang_knows_base)
#        },
        
        {
            "role": "assistant", # лучше писать на языке, на котором сделан фаис
            "content": (
                f"{role_assistant_general_chatgpt}"
            ).format(ids_sql=ids_sql, stenograf_dialog_dkll=stenograf_dialog_dkll, context_dialog_dkll=context_dialog_dkll, static_user_information=static_user_information)
        },
        {
            "role": "user",
            "content": prompt  # Подставляем сгенерированный prompt
        }
    ]

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": messages
    }
                
    result = None
    
    async with aiohttp.ClientSession() as session_http:
        
        for attempt in range(MAX_RETRIES):
            try:
                session['t_g_prompt_is_ready_and_sent'] = get_current_time()
                async with session_http.post(
                    'https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:

                    logging.info(f"Статус ответа OpenAI API (попытка {attempt + 1}): {response.status}")


                    if response.status != 200:
                        logging.error(f"Ошибка API: {response.status}. Попытка {attempt}.")
                    else:
                        session['t_g_gpt_answer_is_ready'] = get_current_time()
                        result = await response.json()

                        content_answer_gpt_general = result.get('choices', [{}])[0].get('message', {}).get('content', '')

                        if not content_answer_gpt_general:
                            logging.error(f"GPT API ответ пустой. Попытка {attempt}.")
                        else:
                            en_match = re.search(r'answer_gpt_general_dkll:\s*(.*?)\s*<<<END>>>', content_answer_gpt_general, re.DOTALL)
                            native_match = re.search(r'answer_gpt_general_native:\s*(.*?)\s*<<<END>>>', content_answer_gpt_general, re.DOTALL)

                            if en_match and native_match:
                                answer_gpt_general_dkll = en_match.group(1).strip()
                                answer_gpt_general_native = native_match.group(1).strip()
                                break  # Всё в порядке, выходим из цикла
                            else:
                                logging.error(f"Ответ не содержит 'answer_gpt_general_dkll' или 'answer_gpt_general_native'. Попытка {attempt}.")
                    
            except Exception as e:
                logging.error(f"Ошибка при вызове GPT API (попытка {attempt}): {e}")

            await asyncio.sleep(RETRY_DELAY)  # Ждём перед повтором запроса

    if not result:
        logging.error(f"Не удалось получить ответ от GPT API после {MAX_RETRIES} попыток.")
        return None
    
    response_data = await response.json()  # Ожидается, что это JSON-ответ
    content_answer_gpt_general = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

# Надо проверять структуру ответа, и если она не содержит всего ожидаемого, то повторять попытку запроса

# Извлекаем контент ответа
    if not content_answer_gpt_general:
        logging.error("Ответ answer_gpt_general не содержит ожидаемого контента в response в виде 'content': ... .")
        return None, None

# Регулярные выражения для поиска ответов с маркером <<<END>>>
    en_match = re.search(r'answer_gpt_general_dkll:\s*(.*?)\s*<<<END>>>', content_answer_gpt_general, re.DOTALL)
    native_match = re.search(r'answer_gpt_general_native:\s*(.*?)\s*<<<END>>>', content_answer_gpt_general, re.DOTALL)

    if not en_match or not native_match:
        logging.error("Ответ GPT API не содержит необходимых полей 'answer_gpt_general_dkll' и 'answer_gpt_general_native'.")
        return None

    
# Извлечение значений из найденных совпадений
    answer_gpt_general_dkll = en_match.group(1).strip() if en_match else None
    answer_gpt_general_native = native_match.group(1).strip() if native_match else None

# Сохраняем результаты в сессии
    if answer_gpt_general_dkll:
        session['answer_gpt_general_dkll'] = answer_gpt_general_dkll
    else:
        logging.warning("Ключ 'answer_gpt_general_dkll' не найден")

    if answer_gpt_general_native:
        session['answer_gpt_general_native'] = answer_gpt_general_native
    else:
        logging.warning("Ключ 'answer_gpt_general_native' не найден")

# Сохраняем результаты в стенограмме ЗДЕСЬ и этот answer_gpt_general_dkll и тот, что в сессии сейчас preview_question_dkll
    if preview_question_dkll:
        # Добавляем фразу клиента
        user_text = f"user: {preview_question_dkll} \n "
        session['stenograf_dialog_dkll'].append(user_text)

    if answer_gpt_general_dkll:
        # Добавляем фразу бота
        bot_text = f"gintaras: {answer_gpt_general_dkll} \n "
        session['stenograf_dialog_dkll'].append(bot_text)

    # Логируем результат
    logging.info(f"Стенография диалога обновлена: \n {stenograf_dialog_dkll} \n ")
    
    
    
# КАССА ЧЕК
    # Проверка и подсчёт стоимости
    if 'usage' in result:
        prompt_tokens = result['usage'].get('prompt_tokens', 0)  # Входящие токены
        completion_tokens = result['usage'].get('completion_tokens', 0)  # Исходящие токены
        # Стоимость токенов
        input_cost_per_token = 0.150 / 1000000  # $0.150 / 1M input tokens
        output_cost_per_token = 0.600 / 1000000  # $0.600 / 1M output tokens
        # Расчёт стоимости
        prompt_cost = prompt_tokens * input_cost_per_token  # Входящие токены
        completion_cost = completion_tokens * output_cost_per_token  # Исходящие токены
        total_cost_general_chatgpt_api = prompt_cost + completion_cost  # Итоговая стоимость
        # Логирование стоимости
        logging.info(f"Входящие токены: {prompt_tokens}, стоимость: {prompt_cost:.8f}$")
        logging.info(f"Исходящие токены: {completion_tokens}, стоимость: {completion_cost:.8f}$")
        logging.info(f"Общая стоимость general_chatgpt_api: {total_cost_general_chatgpt_api:.8f}$")
        # сохраняем в сессию расходы
        session['balance_money_general_chatgpt_api_detail'] = (
            f"prompt {prompt_tokens} for {prompt_cost:.8f}$ + "
            f"answer {completion_tokens} for {completion_cost:.8f}$"
        )
        session['balance_money_general_chatgpt_api_amount'] = total_cost_general_chatgpt_api
    else:
        logging.warning("Информация о токенах general_chatgpt_api отсутствует в ответе API.")

    session['t_g_gpt_answer_was_processed'] = get_current_time()


async def complit_line_calls_now(call_id, session):
    #Заполняем всю строку в call_now в указанную !!!!!!!!!! строку таблицы
    id_calls_now=session.get("id_calls_now")

    # Подсчёт оплаты за STT
    price_azur_stt_per_1chas = 1.0 # в Ажуре цена за 1 час аудио
    stt_sek_stop_countdown = session.get("stt_sek_stop_countdown")
    stt_sek_start_countdown = session.get("stt_sek_start_countdown")
    tts_price_hello_text = session.get("tts_price_hello_text")
    
    time_format = "%Y-%m-%d %H:%M:%S.%f"  # шаблон для разбора даты + миллисекунд

    stop_dt = datetime.strptime(stt_sek_stop_countdown, time_format)
    start_dt = datetime.strptime(stt_sek_start_countdown, time_format)
    
    # Вычисляем разницу в секундах
    delta = stop_dt - start_dt
    total_seconds = delta.total_seconds()
    # Продолжительность распознавания речи в секундах
    stt_sek = math.ceil(total_seconds) # Округляем секю в большую сторону
    # Преобразуем в минуты и секунды
    minutes = stt_sek // 60
    seconds = stt_sek % 60
    #Считаем стоимость
    stt_price = stt_sek * (price_azur_stt_per_1chas / 3600 )
    
    
    # Подсчёт оплаты за TTS
    price_azur_tts_per_1m_simb = 16 # в Ажуре цена за 1 млн символов
    answer_gpt_general_native = session.get("answer_gpt_general_native")
    tts_simballs = len(answer_gpt_general_native)  # Подсчёт количества символов
    tts_price = tts_simballs * ( price_azur_tts_per_1m_simb / 1000000 )
    # TIMING
    timeing_log.info(f"\n !$!$!$!$!$!$!$!-->>> СТОИМОСТЬ за {tts_simballs} символов сиетезированных в звук с помощью Azur_STT СОСТАВЛЯЕТ: {tts_price}$ (tts_price)")

    #сперва решаем с Hello text. 
    hello_text_lang_knows_base = session.get("hello_text_lang_knows_base")
    hello_text_lang_native = session.get("hello_text_lang_native")
    id_calls_now = session.get("id_calls_now")

    if hello_text_lang_knows_base is not None and hello_text_lang_native is not None:
        await CallsNow.filter(id=id_calls_now, call_id=call_id).update(
            input_txt_nativ='First line',
            input_txt_dkll='Hello text',
            output_txt_dkll=hello_text_lang_knows_base,
            output_txt_native=hello_text_lang_native,
            tts_price = tts_price_hello_text,
        )
        session['hello_text_lang_knows_base'] = None
        session['hello_text_lang_native'] = None
        tts_price = tts_price_hello_text 
    else:
        await CallsNow.filter(id=id_calls_now, call_id=call_id).update(
            input_txt_dkll=session.get("preview_question_dkll"),
            input_txt_nativ=session.get("preview_question_native"),
            output_txt_native=session.get("answer_gpt_general_native"),
            output_txt_dkll=session.get("answer_gpt_general_dkll"),
            tts_price = tts_price,
        )

    
    #Формула расчёта цены за всю линию в БД
    bm_all_line_price = session.get("balance_money_process_user_prompt_amount") + session.get("balance_money_price_ada02_question") + session.get("balance_money_general_chatgpt_api_amount") + tts_price + stt_price 
    #  # стоимость  приплюсовывается в самом конце к общей сумме - хвост  за ожидание
    bm_in_start_calls_now = session.get("bm_in_start_calls_now")  #  - забираем остаток на балансе сохраняемый в строке calling_list в поле bm_in_start_calls_now в момент перезарядки новой строки calls_now 
    # TIMING
    bm_in_start_usd = bm_in_start_calls_now
    bm_in_end_usd = bm_in_start_usd - bm_all_line_price
    # TIMING
    t_long_azuraudio_chvost_pered_stop = session.get("t_long_azuraudio_chvost_pered_stop")

    #СОХРАНЕНИЯ В БАЗЫ ДАННЫХ

    await CallsNow.filter( id = id_calls_now, call_id = call_id).update(
        call_id=session.get("call_id"),
        t_in_waiting_speech=session.get("t_in_waiting_speech"),
        t_we_have_stt_results=session.get("t_we_have_stt_results"),
        t_start_gotovit_prompt=session.get("t_start_gotovit_prompt"),
        t_g_prompt_is_ready_and_sent=session.get("t_g_prompt_is_ready_and_sent"),
        t_g_gpt_answer_is_ready=session.get("t_g_gpt_answer_is_ready"),
        t_g_gpt_answer_was_processed=session.get("t_g_gpt_answer_was_processed"),
        t_tts_audio_ready_and_sent_to_klient = session.get("t_tts_audio_ready_and_sent_to_klient"),
        t_tts_audio_was_send_to_klient = session.get("t_tts_audio_was_send_to_klient"),
        
        t_action_audio_starting=session.get("t_action_audio_starting"),
        t_action_audio_will_finish=session.get("t_action_audio_will_finish"),
        
        voice_name=session.get("voice_name"),

        bm_vchat_detail = session.get("balance_money_process_user_prompt_detail"),
        bm_vchat_amount = session.get("balance_money_process_user_prompt_amount"),
        bm_ada02_question = session.get("balance_money_price_ada02_question"),
        bm_gchat_detail = session.get("balance_money_general_chatgpt_api_detail"),
        bm_gchat_amount = session.get("balance_money_general_chatgpt_api_amount"),
        bm_all_line_price = bm_all_line_price, 
        bm_in_end_usd = bm_in_end_usd, # Остаток на балансе в конце этой строки
        
        stt_sek = stt_sek,
        stt_price = stt_price,
        tts_simballs = tts_simballs,
    )
    
    
    # TIMING
    timeing_log.info(f"После записи в БД значение  bm_in_end_usd стало равно bm_in_start_calls_now и оно: {session['bm_in_start_calls_now']}.  Устанавливаю его в таблицу CallsList , в поле bm_in_start_calls_now")
    await CallsList.filter(call_id=call_id).update(
        bm_in_start_calls_now = bm_in_end_usd,
    )


    #И ЕЩЁ НАДО ПРИПЛЮСОВЫВАТЬ ЦЕНУ ЗА  t_long_azuraudio_chvost_pered_stop В КОНЦЕ РАЗГОВОРА ПЕРЕД КЛИКОМ ПО КНОПКЕ СТОП. И проверить почему если клик по СТОП ранее чем включилось распознавание, то время учётное значительно увеличивается
    

#НАДО ПРОВЕСТИ ПОЛНЫЙ ПУТЬ ОБЩЕГО БАЛАНСА НА ОТДЕЛЬНОМ АККАУНТЕ
#НАДО НАСТРОИТЬ КОЭФ НАЦЕНКИ

# TIMING   # TIMING   # TIMING   # TIMING   # TIMING   # TIMING   
# TIMING: С МОМЕНТА НАЧАЛА ОЖИДАНИЯ ЗВУКА ВОПРОСА "bytes" in message, С ДАТЧИКА session['stt_sek_start_countdown'] = get_current_time() ДО НОВОГО ТАКОГО МОМЕНТА, Т Е ПОЛНЫЙ МАКСИМАЛЬНЫЙ ЦИКЛ

# TIMING: С МОМЕНТА РАСПОЗНОВАНИЯ РЕЧИ ДО МОМЕНТА НАЧАЛА ГОЛОСОВОГО ОТВЕТА - ВРЕМЯ ОБДУМЫВАНИЯ ОТВЕТА БОТОМ.   
#session['t_action_audio_starting']
#session['stt_sek_start_countdown']
##=====================
#    stt_sek_stop_countdown != "2025-02-15 00:00:00.000"
#    t_we_have_stt_results= session.get("t_we_have_stt_results")  # Начал думать
#    t_action_audio_starting = session.get("t_action_audio_starting")  # Закончил думать
## Преобразуем строки в объекты datetime
#    fmt = "%Y-%m-%d %H:%M:%S.%f"
#    start_time = datetime.strptime(t_we_have_stt_results, fmt)
#    stop_time = datetime.strptime(t_action_audio_starting, fmt)
## Рассчитываем продолжительность звонка в миллисекундах
#    total_milliseconds = int((stop_time - start_time).total_seconds() * 1000)
## Форматируем в HH:MM:SS
#    total_seconds = total_milliseconds // 1000
#    milliseconds = total_milliseconds % 1000  # Оставшиеся миллисекунды
##    hours, remainder = divmod(total_seconds, 3600)
##    minutes, seconds = divmod(remainder, 60)
## Собираем строку "миллисекунды/часы:минуты:секунды"
##    t_pause_bot_for_thing = f"{total_milliseconds}/{hours:02}:{minutes:02}:{seconds:02}"
#    t_pause_bot_for_thing = f"{total_seconds:02}sek {milliseconds:03}ms" # Формируем строку "00sek 000ms"
##    session['t_pause_bot_for_thing'] = t_pause_bot_for_thing  # Сохраняем значение в сессии
##TIMING
#    timeing_log.info(f" \n Начал думать в    {t_we_have_stt_results}(t_we_have_stt_results)\n Закончил думать в {t_action_audio_starting}(t_action_audio_starting) \n С МОМЕНТА РАСПОЗНОВАНИЯ РЕЧИ ДО МОМЕНТА НАЧАЛА ГОЛОСОВОГО ОТВЕТА \n ВРЕМЯ ОБДУМЫВАНИЯ ОТВЕТА БОТОМ: t_pause_bot_for_thing: {t_pause_bot_for_thing} ")
#    stt_sek_stop_countdown = "2025-02-15 00:00:00.000"
#    stt_sek_stop_countdown != zerro_time
# ===============
#    zerro_time = "2025-02-15 00:00:00.000"
#    while True: # Получаем значения из сессии
#        stt_sek_stop_countdown = session.get("stt_sek_stop_countdown")  # Начал думать
#        t_action_audio_starting = session.get("t_action_audio_starting")  # Закончил думать
##        id_calls_now = session.get("id_calls_now")
##        id_calls = session.get("id_calls")
#        # Если время изменилось, выходим из цикла
#        if (
#            stt_sek_stop_countdown != zerro_time
#            and t_action_audio_starting != zerro_time
#        ):
#            break
#        await asyncio.sleep(0.1)  # 100 мс
## Преобразуем строки в объекты datetime
#    fmt = "%Y-%m-%d %H:%M:%S.%f"
#    start_time = datetime.strptime(stt_sek_stop_countdown, fmt)
#    stop_time = datetime.strptime(t_action_audio_starting, fmt)
## Рассчитываем продолжительность звонка в миллисекундах
#    total_milliseconds = int((stop_time - start_time).total_seconds() * 1000)
## Форматируем в HH:MM:SS
#    total_seconds = total_milliseconds // 1000
#    milliseconds = total_milliseconds % 1000  # Оставшиеся миллисекунды
##    hours, remainder = divmod(total_seconds, 3600)
##    minutes, seconds = divmod(remainder, 60)
## Собираем строку "миллисекунды/часы:минуты:секунды"
##    t_pause_bot_for_thing = f"{total_milliseconds}/{hours:02}:{minutes:02}:{seconds:02}"
#    t_pause_bot_for_thing = f"{total_seconds:02}sek {milliseconds:03}ms" # Формируем строку "00sek 000ms"
##    session['t_pause_bot_for_thing'] = t_pause_bot_for_thing  # Сохраняем значение в сессии
##TIMING
#    timeing_log.info(f" \n Начал думать в    {stt_sek_stop_countdown}(stt_sek_stop_countdown)\n Закончил думать в {t_action_audio_starting}(t_action_audio_starting) \n С МОМЕНТА РАСПОЗНОВАНИЯ РЕЧИ ДО МОМЕНТА НАЧАЛА ГОЛОСОВОГО ОТВЕТА \n ВРЕМЯ ОБДУМЫВАНИЯ ОТВЕТА БОТОМ: t_pause_bot_for_thing: {t_pause_bot_for_thing} ")


#    calling_list
#            balance_money_inthestart=session.get("balance_money_inthestart"), # СТАТИЧНОЕ значение баланса из строки calling_list на начало связи


#    session['t_action_audio_starting'] = '2025-02-15 00:00:00.000'
#    session['stt_sek_stop_countdown'] = '2025-02-15 00:00:00.000'

## END

#    # Логируем содержимое сессии
#    logging.info("\n \n \n  Ф И Н А Л Ь Н А Я     С Е С С И Я ===================================================== \n ")
#    for key, value in session.items():
#        logging.info("  %s: %s", key, value)
#    logging.info("======================================================================================== \n \n ")

if __name__ == "__main__":
    import asyncio

    async def main():
        while True:
            user_input = input("Enter your prompt (or type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("👋 Bye")
                break

            response = await process_user_prompt(user_input)

            if response:
                print(f"🤖 Bot: {response}")

                search_results = await get_faiss_ids_and_distances_from_faiss()
                print(f"🔍 Search Results: {search_results}")

    asyncio.run(main())