import os

from common.setup_logs import setup_logger
from tg.utils import escape_inside_code_and_pre_tags
from ai.logic import run_pipeline
from tg.keyboard import get_response_inline_keyboard
from db.connector import get_db_connection

logger = setup_logger(__name__)
conn = get_db_connection()

project_prefix = os.getenv("PROJECT_NAME")

async def quit_conversation(update):
    logger.info(f"User {update.message.from_user.id} is quitting conversation.")
    
    try:
        await update.message.reply_text("👌 Okay, this topic is over. Text me if you have any other questions.")
    except Exception as e:
        logger.error(f"Error in quit_conversation: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def model_selection(update):
    logger.info(f"Fetching model options for user {update.message.from_user.id}.")

    try:
        query = f"SELECT `option_description` FROM {project_prefix}_app_options"
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        records = cursor.fetchall()
        conn.commit()
        cursor.close()

        buttons = [record['option_description'] for record in records]

        logger.info(f"Model options retrieved: {buttons}")

        await update.message.reply_text(
            "🔍 Please choose approach:",
            reply_markup=get_response_inline_keyboard(buttons)
        )
    except Exception as e:
        logger.error(f"Error in model_selection: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def handle_change_model(query):
    model = query.data
    telegram_id = query.message.chat.id

    logger.info(f"User {telegram_id} selected model: {model}")

    try:
        db_query = f"SELECT * FROM {project_prefix}_app_options WHERE `option_description` = %s"

        cursor = conn.cursor(dictionary=True)
        cursor.execute(db_query, (model,))
        record = cursor.fetchone()
        conn.commit()

        if not record:
            logger.warning(f"Model '{model}' not found in database.")
            await query.message.reply_text("❌ Model not found.")
            cursor.close()
            return

        db_query = f"UPDATE users SET option_id = %s WHERE telegram_id = %s"
        cursor = conn.cursor()
        cursor.execute(db_query, (record["id"], telegram_id))
        conn.commit()
        cursor.close()

        logger.info(f"Updated model selection for user {telegram_id}.")

        await query.message.reply_text(f"🔄 Approach changed.")

    except Exception as e:
        logger.error(f"Error in handle_change_model: {e}", exc_info=True)
        await query.message.reply_text(f"❌ Error: {str(e)}")

async def ai_module_response(update, user_input):
    telegram_id = update.message.from_user.id

    logger.info(f"Processing input from user {telegram_id}: {user_input[:30]}...")

    await update.message.reply_text("⏳ Processing...")

    try:
        response = await run_pipeline(user_input, telegram_id)

        logger.info(f"Response generated for user {telegram_id}. Sending message.")

        await update.message.reply_text(escape_inside_code_and_pre_tags(response), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in ai_module_response: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)}")