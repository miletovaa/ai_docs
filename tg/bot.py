import re
import os
from ai.logic import run_pipeline
from telegram.helpers import escape_markdown
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

from common.setup_logs import setup_logger

load_dotenv()

logger = setup_logger()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update, context):
    await update.message.reply_text("👋 Ciao! Send me a question about Syncro project.")

async def handle_message(update, context):
    user_input = update.message.text
    await update.message.reply_text("⏳ Processing...")

    try:
        response = await run_pipeline(user_input)
        logger.info(f"Response for user: {response}")
        print(response)

        escaped_response = escape_markdown(response)
        await update.message.reply_text(escaped_response, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running!")
    app.run_polling()

if __name__ == "__main__":
    main()