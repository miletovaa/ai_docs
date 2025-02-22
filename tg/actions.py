import os

from common.setup_logs import setup_logger
from tg.utils import escape_inside_code_and_pre_tags
from ai.logic import run_pipeline
from tg.keyboard import get_response_inline_keyboard
from db.connector import get_db_connection

logger = setup_logger()
conn = get_db_connection()

project_prefix = os.getenv("PROJECT_NAME")

async def quit_conversation(update):
    try:
        await update.message.reply_text("👌 Okay, this topic is over. Text me if you have any other questions.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def model_selection(update):
    try:
        query = f"SELECT `option_description` FROM {project_prefix}_app_options"

        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        records = cursor.fetchall()
        conn.commit()
        cursor.close()

        buttons = [record['option_description'] for record in records]

        await update.message.reply_text(
            "🔍 Please choose approach:",
            reply_markup=get_response_inline_keyboard(buttons)
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def handle_change_model(query):
    model = query.data
    telegram_id = query.message.chat.id

    try:
        db_query = f"SELECT * FROM {project_prefix}_app_options WHERE `option_description` = %s"

        cursor = conn.cursor(dictionary=True)
        cursor.execute(db_query, (model,))
        record = cursor.fetchone()
        conn.commit()

        if not record:
            await query.message.reply_text("❌ Model not found.")
            cursor.close()
            return

        db_query = f"UPDATE users SET option_id = %s WHERE telegram_id = %s"
        cursor = conn.cursor()
        cursor.execute(db_query, (record["id"], telegram_id))
        conn.commit()
        cursor.close()

        await query.message.reply_text(f"🔄 Approach changed.")

    except Exception as e:
        await query.message.reply_text(f"❌ Error: {str(e)}")
        print(f"Error in handle_change_model: {e}")


async def ai_module_response(update, user_input):
    telegram_id = update.message.from_user.id
    await update.message.reply_text("⏳ Processing...")

    try:
        response = await run_pipeline(user_input, telegram_id)
        logger.info(f"Response for user: {response}")

        await update.message.reply_text(escape_inside_code_and_pre_tags(response), parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
