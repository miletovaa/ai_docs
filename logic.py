# server.py

#    TF-IDF - Инструмент для генерирования ключевых слов сплошного текста

# Импорт необходимых библиотек и модулей
from pydub import AudioSegment  # Работа с аудиофайлами
import io  # Работа с потоками ввода-вывода
import wave  # Работа с WAV файлами
import struct  # Работа с бинарными данными
import asyncio  # Асинхронное программирование
from asyncio import Queue
import aiohttp  # Асинхронные HTTP-запросы
import aiofiles  # Необходимо установить: pip install aiofiles
import json  # Работа с JSON
import datetime  # Работа с датой и временем
from datetime import datetime
import os  # Работа с операционной системой
import traceback  # Для обработки и вывода трассировки ошибок
import re
import pytz
import traceback
import threading
import subprocess
import requests
import sqlite3
import pymysql
import mysql.connector
import faiss
import numpy as np
import math

from functools import partial
from models_sqlalchemy import SessionLocal, CallsNowSQLAlchemy
from typing import Dict, Any  # Типизация данных
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form, File  # FastAPI компоненты
from fastapi.staticfiles import StaticFiles  # Работа со статическими файлами
from transformers import GPT2TokenizerFast  # Токенизатор для GPT-2
from fastapi.responses import HTMLResponse  # Ответы в формате HTML
from fastapi import HTTPException
from pydantic import BaseModel  # Модели данных
from zoneinfo import ZoneInfo  # Работа с часовыми поясами
import azure.cognitiveservices.speech as speechsdk  # Azure Speech SDK для распознавания и синтеза речи
import openai  # Взаимодействие с OpenAI API
from openai import OpenAI
from tortoise import Tortoise, fields, models  # ORM для работы с базой данных
from tortoise.exceptions import DoesNotExist, ParamsError, OperationalError  # Исключение для отсутствующих записей
from tortoise.transactions import in_transaction
from models import CallsList, CallsNow, FaissIndex, AppsOptions, Klients1000, OutsaitVoicesAzure   #Импорт моделей данных, т е БД
from pathlib import Path  # Работа с путями файлов
from dotenv import load_dotenv  # Загрузка переменных окружения из .env файла
import logging  # Логирование
import chardet  # Определение кодировки файлов
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
#import sqlite3
from sentence_transformers import SentenceTransformer
from starlette.websockets import WebSocketDisconnect
import sys
from azure.cognitiveservices.speech.audio import AudioInputStream, AudioConfig









# Инициализация FastAPI приложения
app = FastAPI()


# Настройка логгера
logging.basicConfig(
    filename='',  # Путь к файлу логов
    level=logging.INFO,  # Уровень логирования (INFO)
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',  # Формат сообщения лога
    datefmt='%Y-%m-%d %H:%M:%S'  # Формат даты и времени
)
logger = logging.getLogger(__name__)  # Получение экземпляра логгера



## Определяем часовой пояс для Литвы
#lithuania_tz = ZoneInfo("Europe/Vilnius")
lithuanian_timezone = ZoneInfo("Europe/Vilnius")

# Загрузка переменных окружения из .env файла
load_dotenv()

executor = ThreadPoolExecutor()

# Определение базового каталога приложения
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))  # Повторная загрузка переменных окружения из конкретного файла


openai.api_key = os.getenv("OPENAI_API_KEY")
# Прямое назначение ключа OpenAI для тестирования (рекомендуется 



sessions: Dict[str, Any] = {}

# Определение моделей данных с использованием Pydantic
class InitData(BaseModel):
    action: str  # Действие (например, "start_call")



# Глобальные объекты для Faiss и модели эмбеддингов
global faiss_index

faiss_index = None
embedding_model = None #"text-embedding-ada-002"
vector_dimension = 1536  # Пример для модели 'ада 002'


print("Python executable:", sys.executable)
print("SDK version:", speechsdk.__version__)


logging.info(f"Speech SDK version: {speechsdk.__version__}")
logging.info(f"Speech SDK file: {speechsdk.__file__}")
logging.info(f"sys.executable: {sys.executable}")
logging.info(f"Python path: {sys.path}")
logging.info(f"Environment PATH: {os.environ.get('PATH')}")


def get_current_time_lt():
    """Возвращает текущее время в литовском часовом поясе с миллисекундами."""
    return datetime.now(lithuanian_timezone).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]





@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def startup_event():
    
    global faiss_index  # Указываем, что будем использовать глобальную переменную
    
    # Инициализация FAISS
    faiss_index = faiss.IndexFlatL2(vector_dimension)
    # Инициализация модели эмбеддингов
    embedding_model = "text-embedding-ada-002"  # Загрузка вашей модели
    
    
    db_url = os.getenv("DATABASE_URLХХХХХХХХХ")  # Получение URL базы данных из переменных окружения
    if db_url:
        logger.info(f"Переменная DATABASE_URLХХХХХХХХХ согласована для подключения к БД. db_url: {db_url}")
    else:
        logger.error("Переменная окружения DATABASE_URLХХХХХХХХХ не настроена или пустая. Проверьте конфигурацию.")
    
    
    logger.info(f"Используемая база данных: {db_url} \n\n")  # Логируем реальный URL БД
    
    # (Загрузка ОЗУ БАЗ ДАННЫХ указанных в файле models.py )Инициализация Tortoise ORM с указанием URL и модулей моделей
    await Tortoise.init(
        db_urlХХХХХХХХХ=db_urlХХХХХХХХХ,  # Используем переменную окружения
        modules={'models': ['models']}     # Указываем модуль, где находятся модели
    )
    logger.info("БД инициализировано с Tortoise ORM в server.py")
    await Tortoise.generate_schemas()
    
    

# это касается загрузки Faiss 
    index_path = os.path.join(os.path.dirname(__file__), 'migris_info.bin')
    if os.path.exists(index_path):
        faiss_migris_info = faiss.read_index(index_path)
        logging.info(f"migris_info.bin загружен в оперативку")
#TIMING
        timeing_log.info(f"\n  \n> \n> ----> ПЕРЕЗАГРУЗИЛСЯ FAISS В ОП")
    else:
        # Если нет файла с индексом — делаем новый пустой (или логируем ошибку)
        faiss_migris_info = faiss.IndexFlatL2(1536)
        logging.info(f"какая-та хуйня с загрузкой migris_info.bin")

#TIMING
        timeing_log.info(f"Присваиваем значение faiss_migris_info глобальной переменной faiss_index независимо от того, загружен ли файл или создан ли новый индекс")
    faiss_index = faiss_migris_info    

    
    
    
# Завершение соединений с базой данных при остановке приложения
@app.on_event("shutdown")
async def shutdown_event():
    await Tortoise.close_connections()

















async def registr_line_calls_now(call_id, session): # Никуда не уходим из этой функции. Она вставная
    id_calling_list = session.get('id_calling_list')
 
    get_data = await CallsList.get(id=id_calling_list, call_id=call_id)
# Заполняем значения в сессии из БД
    bm_in_start_calls_now = get_data.balance_mony_inthestart
    session['bm_in_start_calls_now'] = bm_in_start_calls_now
#TIMING
    timeing_log.info(f"Забираем bm_in_start_calls_now из id_calling_list: {id_calling_list} и bm_in_start_calls_now здесь: {bm_in_start_calls_now} отправлен в сессию")
    
    id_calling_list =session.get('id_calling_list')
    lang_native =session.get('lang_native')
    voice_name =session.get('voice_name')
    stt_results =session.get('stt_results')
    
    bm_in_start_calls_now =session.get('bm_in_start_calls_now')
#    balance_mony_inthestart =session.get('balance_mony_inthestart')

#TIMING
    timeing_log.info(f"\n ###### ВВВ registr_line_call_id подключение к БД calls_now ###### \n чтобы СОЗДАТЬ там НОВУЮ СТРОКУ и вносим в неё: \n # call_id: {call_id} # \n # id_calling_list: {id_calling_list} # \n # lang_native: {lang_native} # \n # voice_name: {voice_name} # \n # bm_in_start_calls_now: {bm_in_start_calls_now} # balance_mony_inthestart: {...} (ЗДЕСЬ СО ВРЕМЁН ХЕЛЛО_ТЕКСТ #")
    reg_calls_now = await CallsNow.create(
        call_id = call_id,
        id_calling_list = id_calling_list,
        lang_native = lang_native,
        voice_name = voice_name,
        stt_results = stt_results,
    )
#TIMING           
    timeing_log.info(f" ###### Завершено подключение к БД calls_now ###### \n ###### ###### ###### ######")

#Сохраняем в сессию
    session['id_calls_now'] = reg_calls_now.id  # Сохраняем ID строки

#TIMING
    id_calls_now =session.get('id_calls_now')
    timeing_log.info(f"\n \n Создали новую строку в registr_line_calls_now id_calls_now: {id_calls_now}")       
    
    

        
        
        


    
    
        
async def openai_block(call_id, session, stt_results):
#TIMING
    timeing_log.info(f"\nВВВВВВВВВВВВВВВВВВВВВВВВВВВВВВВ Зашли в async def openai_block")
#    stt_results = session['stt_results']
#    if session.get('stt_results'):
            
#TIMING
    timeing_log.info(f"\n В session['stt_results'] есть stt_results: {stt_results}, поэтому запускаем viwe_chatgpt_api ")   
    await viwe_chatgpt_api(call_id, session, stt_results)
#    else:
#TIMING
#    timeing_log.info(f"\n Нету stt_results: {stt_results}, поэтому запускаем НЕ viwe_chatgpt_api ")
    

    
    
    
    
    
    
    
    
    
    
    

async def viwe_chatgpt_api(call_id, session, stt_results): #генерирует/обновляет в сессию 'viwe_qwestion_native', 'viwe_qwestion_dkll',  'context_dialog_dkll', стенографирует 'viwe_qwestion_dkll' в 'stenograf_dialog_dkll'... уходит в search_ids_and_distances(call_id, session)...КАССУ ПОДБИВАЕТ
#    сюда заходим из adding_line_to_DB_calls_now, и это можно будет объединить в одну функцию судя по всему, т к adding_line_to_DB_calls_now слишком маленька, простая, и не используется в других функциях
#TIMING
    timeing_log.info(f"Вошли в функцию async def viwe_chatgpt_api(call_id, session, stt_results)")
    
    stenograf_dialog_dkll = session.get('stenograf_dialog_dkll', '')
    context_dialog_dkll = session.get('context_dialog_dkll', '')  # Извлекаем контекст (может быть пустым)
    static_user_information = session.get('static_user_information', '')
    lang_native = session.get('lang_native')
    logging.info(f"Значение lang_native в viwe_chatgpt_api: {lang_native}")
    lang_knows_base = session.get('lang_knows_base')
    logging.info(f"Значение lang_knows_base в viwe_chatgpt_api: {lang_knows_base}")
    
    role_system_viwe_chatgpt = session.get('role_system_viwe_chatgpt')
    role_assistant_viwe_chatgpt = session.get('role_assistant_viwe_chatgpt')
    
    prompt = stt_results

    
#TIMING
    timeing_log.info(f"В функции viwe_chatgpt_api достали из сессии: \n# stenograf_dialog_dkll: {stenograf_dialog_dkll},\n# context_dialog_dkll: {context_dialog_dkll},\n# static_user_information: {static_user_information},\n# lang_native: {lang_native},\n# lang_knows_base: {lang_knows_base},\n# role_system_viwe_chatgpt: {role_system_viwe_chatgpt},\n# role_assistant_viwe_chatgpt: {role_assistant_viwe_chatgpt},\n# prompt: {prompt}")

    
#TIMING
    timeing_log.info(f"В функции viwe_chatgpt_api приступим к формированию запроса async with aiohttp.ClientSession() as session_http")
#    async with aiohttp.ClientSession() as session_http:
    headers = {
        "Authorization": f"Bearer {openai.api_key}",  # Заголовок авторизации с API-ключом
        "Content-Type": "application/json"  # Тип содержимого
    }
    # Формируем сообщения для отправки в API
    messages = [
        {
            "role": "system",
            "content": (
                f"{role_system_viwe_chatgpt}"
            ).format(lang_knows_base=lang_knows_base, lang_native=lang_native)
        },
        {
            "role": "assistant",
            "content": (
                f"{role_assistant_viwe_chatgpt}"
            ).format(lang_knows_base=lang_knows_base, stenograf_dialog_dkll=stenograf_dialog_dkll)
        },
        {
            "role": "user",
            "content": prompt  # Подставляем сгенерированный prompt
        }
    ]
    data = {
        "model": "gpt-4o-mini",  # Модель для генерации
        "messages": messages  # Передаём сообщения
    }


    for message in messages:
        logging.info(f"meeeeessage отправленный от viwe_chatgpt_api: {message}")


# Отправляем POST-запрос к OpenAI API
#TIMING
        timeing_log.info(f"В функции viwe_chatgpt_api момент начала отправки запроса в https://api.openai.com/v1/chat/completions")

    async with aiohttp.ClientSession() as gpt_session:
        try:
            async with gpt_session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:
                logging.info(f"Статус ответа OpenAI API НА viwe_qwestion_text (call_id: {call_id}): {response.status}")

                # TIMING
                timeing_log.info("В функции viwe_chatgpt_api момент получения ответа из OpenAI API")

                if response.status == 200:
                    result = await response.json()  # Асинхронно получаем JSON
#TIMING
                    timeing_log.info(f"В функции viwe_chatgpt_api момент получения ответа из https://api.openai.com/v1/chat/completions")
#TIMING
                    timeing_log.info(f"Смотрим response.status == 200 viwe_chatgpt_api: \n {result}")
                else:
                    error_message = response.text()
                    logging.error(f"Ошибка OpenAI API (call_id: {call_id}): статус {response.status}, тело: {await response.text()}")
                    logging.error(f"GPT API Error: {response.status_code } - {error_message}")
                    return ""
        except Exception as e:
            logging.error(f"Исключение в viwe_chatgpt_api (call_id: {call_id}): {e}")
            return ""


    # Ожидается, что это JSON-ответ
    response_data = await response.json()
    
    # Извлекаем контент ответа
    content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
    if not content:
        logging.error("Ответ не содержит ожидаемого контента в response.")
        return None, None

    # Регулярные выражения для поиска ответов с маркером <<<END>>>
    en_match = re.search(r'viwe_qwestion_dkll:\s*(.*?)\s*<<<END>>>', content, re.DOTALL)
    native_match = re.search(r'viwe_qwestion_native:\s*(.*?)\s*<<<END>>>', content, re.DOTALL)

    # Извлечение значений из найденных совпадений
    viwe_qwestion_dkll = en_match.group(1).strip() if en_match else None
    viwe_qwestion_native = native_match.group(1).strip() if native_match else None


#TIMING
    timeing_log.info(f"В функции viwe_chatgpt_api начали заносить в СЕССИЮ viwe_qwestion_dkll, viwe_qwestion_native")
    if viwe_qwestion_dkll:
        session['viwe_qwestion_dkll'] = viwe_qwestion_dkll
        logging.info(f"viwe_qwestion_dkll успешно загружен в СЕССИЮ: {viwe_qwestion_dkll}")
    else:
        logging.warning("Значение 'viwe_qwestion_dkll' не найден")

    if viwe_qwestion_native:
        session['viwe_qwestion_native'] = viwe_qwestion_native
        logging.info(f"viwe_qwestion_native успешно успешно загружен в СЕССИЮ: {viwe_qwestion_native}")
    else:
        logging.warning("Значение 'viwe_qwestion_native' не найден")
#TIMING
    timeing_log.info(f"В функции viwe_chatgpt_api мы закончили заносить в СЕССИЮ viwe_qwestion_dkll, viwe_qwestion_native")

            
#КАССА ЧЕК
    if 'usage' in result:
#TIMING
        timeing_log.info(f"В функции viwe_chatgpt_api начинаем обсчитывать расходы на GPT") 
        prompt_tokens = result['usage'].get('prompt_tokens', 0)  # Входящие токены
        completion_tokens = result['usage'].get('completion_tokens', 0)  # Исходящие токены
        # Стоимость токенов
        input_cost_per_token = 0.150 / 1000000  # $0.150 / 1M input tokens
        output_cost_per_token = 0.600 / 1000000  # $0.600 / 1M output tokens
        # Расчёт стоимости
        prompt_cost = prompt_tokens * input_cost_per_token # Входящие токены
        completion_cost = completion_tokens * output_cost_per_token # Исходящие токены
        total_cost_viwe_chatgpt_api = prompt_cost + completion_cost #ИТОГО СТОИМОСТЬ ВОПРОС_ОТВЕТ
        logging.info(f"Входящие токены: {prompt_tokens}, стоимость: {prompt_cost:.8f}$")
        logging.info(f"Исходящие токены: {completion_tokens}, стоимость: {completion_cost:.8f}$")
        logging.info(f"Общая стоимость: {total_cost_viwe_chatgpt_api:.8f}$")
#                сохраняем в сессию расходы
        session['balance_mony_viwe_chatgpt_api_detal'] = (
            f"promt {prompt_tokens} for {prompt_cost:.8f}$ + "
            f"ansver {completion_tokens} for {total_cost_viwe_chatgpt_api:.8f}$"
        )
        session['balance_mony_viwe_chatgpt_api_amоunt'] = total_cost_viwe_chatgpt_api
#TIMING
        timeing_log.info(f"В функции viwe_chatgpt_api посчитали расходы на GPT, составило {total_cost_viwe_chatgpt_api:.8f}$")
    else:
        logging.warning("Информация о токенах отсутствует в ответе API.")

# ВЫЗЫВАЕМ СЛЕДУЮЩУЮ ФУНКЦИЮ
#TIMING
    timeing_log.info(f"Внутри функции viwe_chatgpt_api вызываем функцию search_ids_and_distances(call_id, session), которая находит методом Faiss в ОП нужные данные, ориентируясь по векторам, и возвращает номера строк дублирующся в БД. Т е уходим в async def search_ids_and_distances(call_id, session) \n\n")
    await search_ids_and_distances(call_id, session)
        
        
        
        
        
async def search_ids_and_distances(call_id, session): #ВОЗВРАЩАЕТ ids и distances
#ВОЗВРАЩАЕТ embedding_ids - ЭТО НАЙДЕННАЯ ГРУППА СМЫСЛОВ ДЛЯ SQL и дистанцию до искомого distances_level

#TIMING
    timeing_log.info(f"Апплодисменты! Мы в async def search_ids_and_distances(call_id, session). \n искать в search_ids_and_distances не на английском, а на языке БЗ, а она не всегда на английском.")
# может изменю, но сейчас здесь отправлю на эмбединги только text_in_lang_knows_base - это текст вопроса в окошко юзеру 

    #    global уже определена эта глобальная переменная выше
    top_k = 3 #скоько подходящих сторк из БД/Faiss будем брать 
    if faiss_index is None:
        raise ValueError("FAISS индекс не загружен в память.")


# Получаем данные из сессии
    stenograf_dialog_dkll = session.get('stenograf_dialog_dkll', [])
    viwe_qwestion_dkll = session.get('viwe_qwestion_dkll', '')
# Преобразуем stenograf_dialog_dkll в строку, если список не пустой
    if isinstance(stenograf_dialog_dkll, list) and stenograf_dialog_dkll:
        stenograf_dialog_dkll = ' '.join(stenograf_dialog_dkll)
        

# Объединяем две строки
#    qwestion_ids_and_distances = viwe_qwestion_dkll + ' ' + stenograf_dialog_dkll if stenograf_dialog_dkll else viwe_qwestion_dkll

#ОН ПЛОХО ПОДГОТАВЛИВАЕТ ЗАПРОС. Т К ДОЛЖЕН БЫТЬ ЯВНО ВЫДЕЛЕН viwe_qwestion_dkll В ГЛАВНОЕ, А stenograf_dialog_dkll КАК ВТОРОСТЕПЕННОЕ Т Е ДОПОЛНИТЕЛЬНОЕ. ...И ЕЩЁ И НА ЯЗЫКЕ БЗ. !О, КАКИЕ-ТО НУЖНЫЕ ФРАЗОЧКИ Я МОГУ ПРЕПОДГОТАВЛИВАТЬ В ХЕЛЛО ТЕКСТ, КОТОРЫЙ ОДИН РАЗ ПЕРЕВОДИТ НА ЯЗЫК БЗ. И значения itsqwestion и usedstenograf БУДУТ ПОДСТАВЛЕНЫ СЮДА НА ЯЗЫКЕ БЗ


# Формируем запрос с явным выделением viwe_qwestion_dkll как основного текста и stenograf_dialog_dkll как дополнительного
#    itsqwestion = f'Вопрос такой: '
    itsqwestion = f' \n Klausimas toks: '
#    usedstenograf = f', а при ответе на этот вопрос дополнительно используй информацию из текста уже произошедшего диалога: '
    usedstenograf = f', \n atsakant į šį klausimą, papildomai naudokite jau vykusio dialogo informaciją: \n '
    if stenograf_dialog_dkll:
        qwestion_ids_and_distances = (
            f'{itsqwestion}"{viwe_qwestion_dkll}"{usedstenograf}{stenograf_dialog_dkll}'
        )
    else:
        qwestion_ids_and_distances = f'Вопрос такой: "{viwe_qwestion_dkll}"'


    text_in_lang_knows_base = qwestion_ids_and_distances
#TIMING
    timeing_log.info(f" \n \n Спасибо за Апплодисменты! Теперь запрос внутри search_ids_and_distances выглядит так text_in_lang_knows_base: {text_in_lang_knows_base}.")  

#    TF-IDF - Инструмент для генерирования ключевых слов сплошного текста 

#TIMING
    timeing_log.info(f"Старт text_in_lang_knows_base_to_vector.")
    embedding = text_in_lang_knows_base_to_vector(call_id, session, text_in_lang_knows_base)
#        logging.info(f"Такой вот эмбединг: {embedding}")
#TIMING
    timeing_log.info(f"Финиш text_in_lang_knows_base_to_vector. Но будем не только viwe_qwestion_dkll трансформировать в эмбеддинги, А ВЕСЬ ПРОМТ!!!, который надо подготовить")
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

#    return embedding



#TIMING
    timeing_log.info(f"Старт faiss_index.search - это поиск по эмбеддингам в ОП.")
# Выполняем поиск в FAISS
    distances, ids = faiss_index.search(embedding, top_k)
#TIMING
    timeing_log.info(f"Завершён поиск по эмбеддингам в ОП посредством faiss_index.search.")
    
    logging.info(f"Такой вот ids: {ids} и вот такой  distances: {distances}  в search_ids_and_distances()")
    
    session['ids'] = ids # номера id в Faiss соответствующие БД.
    session['distances'] = distances # номера id в Faiss соответствующие БД.


    
#TIMING
    timeing_log.info(f"Старт get_text_from_mysql для забирания нужных строк с текстом для промта в general_chatgpt_api.")
# ВЫЗЫВАЕМ ФУНКЦИЮ
#    get_text_from_mysql(call_id, ids)
    loop = asyncio.get_event_loop()  # Получаем текущий цикл событий
    await get_text_from_mysql(call_id, session, ids)
#TIMING
    timeing_log.info(f"Исполнена get_text_from_mysql для забирания нужных строк с текстом для промта в general_chatgpt_api. \n")
    
    # Возвращаем результаты в удобной форме
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
        query = f"SELECT * FROM faiss_migris_info WHERE id IN ({','.join(['%s'] * len(id_list))})"
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
        timeing_log.info(f"Если вы видите эту запись, ...то! Вы внутри функции get_text_from_mysql и достигнут результат search_ids_and_distances, который заключался в том, чтобы добыть ids_sql - это найденные тексты из БЗ для использования их в general_chatgpt_api.")
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
        session['balance_mony_price_ada02_qweation'] = cost
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
        





async def client_can_speak(call_id, session): #Устанавливаем флаги разрешающие обработку Ажур МОЖНО ГОВОРИТЬ В МИКРОФОН
#TIMING
    timeing_log.info(f"Начали ставить флаги в def client_can_speak позволяющие обработку входящего аудио Ажуром - это когда session['stt_can_input'] = True")
    session['stt_can_input'] = True
    session['audio_starting'] = False
    
#TIMING
    timeing_log.info(f"Завершили ставить флаги в def client_can_speak позволяющие обработку входящего аудио Ажуром - это когда session['stt_can_input'] = True и микрофон открыт")

#                    logging.info("Содержимое сессии после связки ВОПРОС-ОТВЕТ:")
#                    for key, value in session.items():
#                        logging.info("  %s: %s", key, value)

    timeing_log.info(f"Флаг (в client_can_speak) stt_can_input на True для call_id {call_id}. МОЖНО ГОВОРИТЬ В МИКРОФОН")
    
#УХОДИМ В ЦИКЛ ВОПРОС-ОТВЕТ
#TIMING
    timeing_log.info(f"УХОДИМ В ЦИКЛ ВОПРОС-ОТВЕТ ИЗ ФУНКЦИИ async def client_can_speak(call_id, session)")
#    await dialog_while_cykl(call_id, session)
#    await list_tasks(call_id, session) 




        
        
        




def ocenka_distances_level(call_id, distances): # оценивает насколько вопрос относится к теме
#    session['distances'] = distances # насколько ответ схож с вопросом.
    logging.info(f"Посмотрим внутри функции ocenka_distances_level на distances - {distances}")
    
    
#Допустим, у вас в индексе 3 вектора, и вы ищете 2 ближайших (top_k=2):
    distances, ids = faiss_index.search(embedding, 2)
    print(distances)
    # [[0.1, 0.5]]  # Расстояния до 2 ближайших векторов
    print(ids)
    # [[12, 45]]    # ID этих двух векторов

#    Если FAISS не находит подходящего вектора (например, для слишком малых данных в индексе), он вернёт -1:
    distances, ids = faiss_index.search(embedding, 2)
    print(distances)
    # [[0.1, inf]]
    print(ids)
    # [[12, -1]]
#Расстояния дают представление о том, насколько найденные векторы (результаты) похожи на запрос. Если расстояние слишком велико (например, больше определённого порога), это может означать, что результат не релевантен.
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


    #    Пороговая фильтрация для улучшения производительности

        # Обрезаем результаты с расстоянием > 1.0
        filtered_ids = [id for id, distance in zip(ids[0], distances[0]) if distance <= 1.0]
        print("Фильтрованные ID:", filtered_ids)


    
    

    


    
    
    

    


    
async def general_chatgpt_api(call_id, session): # ДОЛЖЕН ОТДАТЬ answer_gpt_general_native НА ОЗВУЧКУ
#TIMING
    
    session['t_start_gotovit_promt'] = get_current_time_lt()

    timeing_log.info(f"Вошли в функцию general_chatgpt_api.")
#    ввести понятие ВЕС ИНФОРМАЦИИ на основании которой генерируется запрос.
    MAX_RETRIES = 3  # Максимальное количество повторов при ошибке
    RETRY_DELAY = 0.5  # Задержка между повторами (в секундах)


    id_calls_now = session.get('id_calls_now')
    viwe_qwestion_dkll = session.get('viwe_qwestion_dkll')
    static_user_information = session.get('static_user_information', "")

    role_system_general_chatgpt = session.get('role_system_general_chatgpt')
    role_assistant_general_chatgpt = session.get('role_assistant_general_chatgpt')
    
    lang_native = session.get('lang_native')
    lang_knows_base = session.get('lang_knows_base')
    
    ids_sql = session.get('ids_sql', [])
    logging.info(f"Так выглядит ids_sql достаный из сессии в блоке/функции general_chatgpt_api:  {ids_sql}")
    
    stenograf_dialog_dkll = session.get('stenograf_dialog_dkll', "")
    
    context_dialog_dkll = session.get('context_dialog_dkll', "")
        
    num_world_answer = 80
    
    prompt = viwe_qwestion_dkll
    if not isinstance(prompt, str) or not prompt.strip():
        logging.error("Некорректный или пустой prompt для GPT API.")
        return ""

#СОБИРАЕМ ПЕРЕМЕННУЮ messages, ЧТОБЫ ОТПРАВИТЬ ЕЁ В data
    messages = [
        {   
            "role": "system", # лучше писать на английском
            "content": (
                f"{role_system_general_chatgpt}"
            ).format(num_world_answer=num_world_answer, lang_native=lang_native, lang_knows_base=lang_knows_base)
        },
        
#        {   
#            "role": "system",
#            "content": role_system_general_chatgpt.format(num_world_answer=num_world_answer, lang_native=lang_native, lang_knows_base=lang_knows_base)
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
#TIMING
    timeing_log.info(f"Meeeeessage в general_chatgpt_api: {messages}")

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
                session['t_g_promt_is_ready_and_sent'] = get_current_time_lt()
                async with session_http.post(
                    'https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:

                    logging.info(f"Статус ответа OpenAI API (попытка {attempt + 1}): {response.status}")



                    if response.status != 200:
                        logging.error(f"Ошибка API: {response.status}. Попытка {attempt}.")
                    else:
                        session['t_g_gpt_answer_is_ready'] = get_current_time_lt()
                        result = await response.json()
                        timeing_log.info(f"Ответ general_chatgpt_api: \n{result}")

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

#Надо проверять структуру ответа, и если она не содержит всего ожидаемого, то повторять попытку запроса

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
#==================ЗДЕСЬ ПОСТАВИЛИ В СЕССИЮ ТЕКСТ ДЛЯ generate_tts_audio
    else:
        logging.warning("Ключ 'answer_gpt_general_native' не найден")
#TIMING
        timeing_log.info(f"В функции viwe_chatgpt_api момент начала отправки запроса в https://api.openai.com/v1/chat/completions")

# Сохраняем результаты в стенограмме ЗДЕСЬ и этот answer_gpt_general_dkll и тот, что в сессии сейчас viwe_qwestion_dkll
    if viwe_qwestion_dkll:
        # Добавляем фразу клиента
        user_text = f"user: {viwe_qwestion_dkll} \n "
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
        session['balance_mony_general_chatgpt_api_detal'] = (
            f"promt {prompt_tokens} for {prompt_cost:.8f}$ + "
            f"ansver {completion_tokens} for {completion_cost:.8f}$"
        )
        session['balance_mony_general_chatgpt_api_amоunt'] = total_cost_general_chatgpt_api
    else:
        logging.warning("Информация о токенах general_chatgpt_api отсутствует в ответе API.")

    session['t_g_gpt_answer_was_processed'] = get_current_time_lt()

    
#===================================
#   

#TIMING
    timeing_log.info(f"Начинаем генерировать из функции general_chatgpt_api её результат answer_gpt_general_native в generate_tts_audio.\n\n")
# ЗАПУСК Генерация TTS аудио        
    logging.info(f"БУДЕМ ОЗВУЧИВАТЬ: {answer_gpt_general_native}.\n\n")
    
    session['t_start_generate_tts_audio'] = get_current_time_lt()
    
    audio_data = await generate_tts_audio(session, answer_gpt_general_native, call_id)
#TIMING
    timeing_log.info(f"Завершена генерация answer_gpt_general_native в generate_tts_audio. И это лог из возврата в general_chatgpt_api")
    



async def generate_tts_audio(session, text, call_id): #внутри её обновляется язык и голос из БД
#Здесь можно добавить в аргументы "случай" и если тот случай когда сюда зашли из Нелоу, то уходим в одно направление , а если зашли из Цикла, то в другое направление   
#TIMING
    timeing_log.info(f"\n Зашли в  generate_tts_audio с call_id: {call_id}\n Текст для озвучки уже должен быть распознан . ") 
    # Настройка конфигурации синтеза речи. stt_resalt:  {text}
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_language = session.get('lang_native')  # Используем язык из СЕССИИ
    speech_config.speech_synthesis_voice_name = session.get('voice_name')  # Используем голос из СЕССИИ

    lang_native = session.get('lang_native')
    voice_name = session.get('voice_name')
    id_calls_now = session.get('id_calls_now')
    id_calling_list = session.get('id_calling_list')
    websocket = session.get('websocket')
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)  # Устанавливаем формат аудио

    # Создаем синтезатор речи
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
#TIMIN
    timeing_log.info(f"Настройки SpeechSDK в generate_tts_audio \n для call_id: {call_id} \n в строке (id_calls_now): {id_calls_now} \n на языке (lang_native): {lang_native} \n голосом (voice_name): {voice_name} \n synthesizer (synthesizer): {synthesizer} \n  ") 
#===================================
#TIMING
    timeing_log.info(f"Начинаем рендерить '.txt' в аудио. \n рендерим с помощью synthesizer.speak_text_async(text)") 
    # Запускаем асинхронную функцию синтеза речи : {text}
    result_future = synthesizer.speak_text_async(text)
    # Получаем результат Future асинхронно в отдельном потоке
    result = await asyncio.to_thread(result_future.get)
#TIMING
    timeing_log.info(f"Завершен рендеринг txt") 

    # Проверяем результат синтеза
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data =result.audio_data
#        return audio_data   # Возвращаем аудио данные

        # если всё в порядке, возвращаем
#        return audio_data
#    else:
#        traceback.print_exc()
#        return b""
    else:
        traceback.print_exc()  # Выводим трассировку ошибки
        return b""  # Возвращаем пустой байтовый объект в случае ошибки

#    сгенерированная там audio_data, и она выглядит вот так ЭТО ТАК ВЫГЛЯДИТ ССИНТЕЗИРОВАННЫЙ ЗВУК текста hello_text : ... . Это для строки {id_calling_list}, с call_id: {call_id}
#TIMING
#    timeing_log.info(f"audio_data: {audio_data}") 
    if audio_data:
#!!!!!!!!!! Отправка аудиоданных клиенту через WebSocket
#TIMING
        timeing_log.info(f"Начинаем отправлять audio_data(аудиозвук) через await websocket.send_bytes на Клиент.Размер: {len(audio_data)} байт. Для строки {id_calling_list}, с call_id: {call_id}")
    
        session['t_tts_audio_ready_and_sent_to_klient'] = get_current_time_lt()
    
        await websocket.send_bytes(audio_data)
        logging.info(f"Аудио В generate_tts_audio отправлено для call_id {call_id}. Размер: {len(audio_data)} байт")
        
        session['t_tts_audio_was_send_to_klient'] = get_current_time_lt()
        
#TIMING
        timeing_log.info(f"Завершили отправлять audio_data через await websocket.send_bytes на Клиент. Размер: {len(audio_data)} байт. Для строки {id_calling_list}, с call_id: {call_id}")
    
#TIMING
        timeing_log.info(f"\n Из generate_tts_audio запустаем complit_line_calls_now пытаясь заполнить в calls_now все отметки времени на t_ ")  
        await complit_line_calls_now(call_id, session)
#TIMING
        timeing_log.info(f"\n В generate_tts_audio отработатл complit_line_calls_now ")  
    
    else:
        logging.error("Ошибка генерации TTS call_id: {call_id}, id_calling_list: {id_calling_list}.")
        
        




    



async def complit_line_calls_now(call_id, session):
#Заполняем всю строку в call_now в указанную !!!!!!!!!! строку таблицы
    id_calls_now=session.get("id_calls_now")
    
    
    
# Подсчёт оплаты за STT
    price_azur_stt_per_1chas = 1.0 # в Ажуре цена за 1 час аудио
    stt_sek_stop_countdown = session.get("stt_sek_stop_countdown")
    stt_sek_start_countdown = session.get("stt_sek_start_countdown")
    tts_price_hello_text = session.get("tts_price_hello_text")
#TIMING
    timeing_log.info(f"\n stt_sek_stop_countdown: {stt_sek_stop_countdown}\n stt_sek_start_countdown: {stt_sek_start_countdown}")
    
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
##TIMING
    timeing_log.info(f"\n !$!$!$!$!$!$!$!-->>> СТОИМОСТЬ {stt_sek} секунд транскрибированного аудио, поступившего в Azur_STT СОСТАВЛЯЕТ: {stt_price}$ (stt_price)")

    
    
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
            input_txt_dkll=session.get("viwe_qwestion_dkll"),
            input_txt_nativ=session.get("viwe_qwestion_native"),
            output_txt_native=session.get("answer_gpt_general_native"),
            output_txt_dkll=session.get("answer_gpt_general_dkll"),
            tts_price = tts_price,
        )

    
#Формула расчёта цены за всю линию в БД
    bm_all_line_price = session.get("balance_mony_viwe_chatgpt_api_amоunt") + session.get("balance_mony_price_ada02_qweation") + session.get("balance_mony_general_chatgpt_api_amоunt") + tts_price + stt_price 
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
        t_start_gotovit_promt=session.get("t_start_gotovit_promt"),
        t_g_promt_is_ready_and_sent=session.get("t_g_promt_is_ready_and_sent"),
        t_g_gpt_answer_is_ready=session.get("t_g_gpt_answer_is_ready"),
        t_g_gpt_answer_was_processed=session.get("t_g_gpt_answer_was_processed"),
        t_start_generate_tts_audio=session.get("t_start_generate_tts_audio"),
        t_tts_audio_ready_and_sent_to_klient = session.get("t_tts_audio_ready_and_sent_to_klient"),
        t_tts_audio_was_send_to_klient = session.get("t_tts_audio_was_send_to_klient"),
        
        t_action_audio_starting=session.get("t_action_audio_starting"),
        t_action_audio_will_finish=session.get("t_action_audio_will_finish"),
        
        voice_name=session.get("voice_name"),

        bm_vchat_detal = session.get("balance_mony_viwe_chatgpt_api_detal"),
        bm_vchat_amоunt = session.get("balance_mony_viwe_chatgpt_api_amоunt"),
        bm_ada02_qweation = session.get("balance_mony_price_ada02_qweation"),
        bm_gchat_detal = session.get("balance_mony_general_chatgpt_api_detal"),
        bm_gchat_amоunt = session.get("balance_mony_general_chatgpt_api_amоunt"),
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
# TIMING: С МОМЕНТА НАЧАЛА ОЖИДАНИЯ ЗВУКА ВОПРОСА "bytes" in message, С ДАТЧИКА session['stt_sek_start_countdown'] = get_current_time_lt() ДО НОВОГО ТАКОГО МОМЕНТА, Т Е ПОЛНЫЙ МАКСИМАЛЬНЫЙ ЦИКЛ

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
#            balance_mony_inthestart=session.get("balance_mony_inthestart"), # СТАТИЧНОЕ значение баланса из строки calling_list на начало связи
    
    
#    session['t_action_audio_starting'] = '2025-02-15 00:00:00.000'
#    session['stt_sek_stop_countdown'] = '2025-02-15 00:00:00.000'
        
    
#
## END
#


#    # Логируем содержимое сессии
#    logging.info("\n \n \n  Ф И Н А Л Ь Н А Я     С Е С С И Я ===================================================== \n ")
#    for key, value in session.items():
#        logging.info("  %s: %s", key, value)
#    logging.info("======================================================================================== \n \n ")