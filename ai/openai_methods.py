import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import aiohttp

from common.setup_logs import setup_logger

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
project_prefix = os.getenv("PROJECT_NAME")
vector_dimension = int(os.getenv("VECTOR_DIMENSION"))
embedding_model = os.getenv("EMBEDDING_MODEL")

logger = setup_logger()

index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')


def calculate_cost_of_request(response):
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
        logger.warning("Информация о токенах отсутствует в ответе API.")
        return 0


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


async def send_openai_request(prompt, system_context="", assistant_context="", model="gpt-4o-mini"):
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "user", "content": prompt}
    ]

    if system_context is not "":
        messages.append({"role": "system", "content": system_context})

    if assistant_context is not "":
        messages.append({"role": "assistant", "content": assistant_context})

    data = {
        "model": model,
        "messages": messages
    }

    logger.info(f"Sending OpenAI request: {data}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:
                logger.info(f"📥 OpenAI API Response Status: {response.status}")

                if response.status == 200:
                    response_data = await response.json()
                    return response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                else:
                    error_message = await response.text()
                    logger.error(f"OpenAI API Error: {response.status} - {error_message}")
                    return ""

        except Exception as e:
            logger.error(f"Exception in send_openai_request: {e}")
            return ""