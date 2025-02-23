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

logger = setup_logger(__name__)

index_path = os.path.join(os.path.dirname(__file__), f'{project_prefix}_info.bin')

def calculate_cost_of_request(response):
    logger.info("Calculating request cost...")

    if 'usage' in response:
        input_tokens = response['usage'].get('prompt_tokens', 0)
        output_tokens = response['usage'].get('completion_tokens', 0)

        input_cost_per_token = 0.150 / 1000000  # $0.150 / 1M input tokens
        output_cost_per_token = 0.600 / 1000000  # $0.600 / 1M output tokens

        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost = input_cost + output_cost

        logger.info(f"Request cost calculated: {total_cost:.6f}$")
        return total_cost
    else:
        logger.warning("Token usage info missing in API response.")
        return 0

def text_to_vector(text: str):
    if not text.strip():
        logger.warning("Received empty text for vectorization.")
        return None

    logger.info(f"Generating embedding for text: {text[:30]}...")

    try:
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

async def send_openai_request(prompt, system_context="", assistant_context="", model="gpt-4o-mini"):
    logger.info(f"Preparing OpenAI request for model: {model}")

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "user", "content": prompt}
    ]

    if system_context != "" and system_context is not None:
        messages.append({"role": "system", "content": system_context})

    if assistant_context != "" and assistant_context is not None:
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
            logger.error(f"Exception in send_openai_request: {e}", exc_info=True)
            return ""