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

def escape_inside_code_and_pre_tags(text: str) -> str:
    pattern = re.compile(r"<(code|pre)>(.*?)</\1>", flags=re.DOTALL)

    def replacer(match):
        tag = match.group(1)
        inner_content = match.group(2)
        escaped_content = inner_content.replace("<", "&lt;").replace(">", "&gt;")
        return f"<{tag}>{escaped_content}</{tag}>"

    return pattern.sub(replacer, text)

async def handle_message(update, context):
    user_input = update.message.text
    await update.message.reply_text("⏳ Processing...")

    try:
        response = await run_pipeline(user_input)
        logger.info(f"Response for user: {response}")

        await update.message.reply_text(escape_inside_code_and_pre_tags(response), parse_mode="HTML")
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