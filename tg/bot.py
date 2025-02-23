import os
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from dotenv import load_dotenv

from common.setup_logs import setup_logger
from tg.keyboard import reply_keyboard_markup, QUIT_BUTTON, CHOOSE_MODEL_BUTTON
from tg.actions import quit_conversation, model_selection, ai_module_response, handle_change_model
from db.connector import get_db_connection

load_dotenv()

conn = get_db_connection()
logger = setup_logger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
project_name = os.getenv("PROJECT_NAME")

async def start(update, context):
    telegram_id = update.message.from_user.id
    logger.info(f"User {telegram_id} started the bot.")

    try:
        query = f"SELECT * FROM users WHERE telegram_id = '{telegram_id}'"
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        record = cursor.fetchone()
        conn.commit()
        cursor.close()

        if not record:
            logger.info(f"User {telegram_id} not found in database. Creating a new entry.")
            query = f"INSERT INTO users (telegram_id, option_id) VALUES ('{telegram_id}', 1)"
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

        await update.message.reply_text(
            f"👋 Ciao! Send me a question about the {project_name} project.",
            reply_markup=reply_keyboard_markup
        )
        logger.info(f"Sent welcome message to user {telegram_id}.")
    
    except Exception as e:
        logger.error(f"Error in start command for user {telegram_id}: {e}", exc_info=True)
        await update.message.reply_text("❌ An error occurred. Please try again later.")

async def handle_message(update, context):
    telegram_id = update.message.from_user.id
    user_input = update.message.text.strip()
    logger.info(f"Received message from user {telegram_id}: {user_input}")

    try:
        if user_input == QUIT_BUTTON:
            logger.info(f"User {telegram_id} chose to quit the conversation.")
            await quit_conversation(update)
        elif user_input == CHOOSE_MODEL_BUTTON:
            logger.info(f"User {telegram_id} chose to change model.")
            await model_selection(update)
        else:
            logger.info(f"Processing AI response for user {telegram_id}.")
            await ai_module_response(update, user_input)
    
    except Exception as e:
        logger.error(f"Error handling message from user {telegram_id}: {e}", exc_info=True)
        await update.message.reply_text("❌ An error occurred while processing your request.")

async def button_handler(update, context):
    query = update.callback_query
    telegram_id = query.message.chat.id

    try:
        logger.info(f"User {telegram_id} clicked a button: {query.data}")
        await query.answer()
        await handle_change_model(query)
    
    except Exception as e:
        logger.error(f"Error handling button click from user {telegram_id}: {e}", exc_info=True)
        await query.message.reply_text("❌ An error occurred while processing your request.")

def main():
    logger.info("Initializing Telegram bot...")
    
    try:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_handler(CallbackQueryHandler(button_handler))

        print("✅ Bot is running!")
        logger.info("✅ Bot is running!")
        app.run_polling()
    
    except Exception as e:
        logger.critical(f"Fatal error in bot execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()