import os
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from dotenv import load_dotenv

from common.setup_logs import setup_logger
from tg.keyboard import reply_keyboard_markup, QUIT_BUTTON, CHOOSE_MODEL_BUTTON
from tg.actions import quit_conversation, model_selection, ai_module_response, handle_change_model
from db.connector import get_db_connection

load_dotenv()

conn = get_db_connection()
logger = setup_logger()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
project_name = os.getenv("PROJECT_NAME")

async def start(update, context):
    telegram_id = update.message.from_user.id

    query = f"SELECT * FROM users WHERE telegram_id = '{telegram_id}'"
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    record = cursor.fetchone()
    conn.commit()
    cursor.close()

    if not record:
        query = f"INSERT INTO users (telegram_id, option_id) VALUES ('{telegram_id}', 1)"
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        cursor.close()

    await update.message.reply_text(
        f"👋 Ciao! Send me a question about the {project_name} project.",
        reply_markup=reply_keyboard_markup
    )

async def handle_message(update, context):
    user_input = update.message.text

    if user_input.strip() == QUIT_BUTTON:
        await quit_conversation(update)
    elif user_input.strip() == CHOOSE_MODEL_BUTTON:
        await model_selection(update)
    else:
        await ai_module_response(update, user_input)

async def button_handler(update, context):
    query = update.callback_query
    await query.answer()

    await handle_change_model(query)

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("✅ Bot is running!")
    app.run_polling()

if __name__ == "__main__":
    main()