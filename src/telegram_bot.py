import logging

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.config import get_settings

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Command handlers ──────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the arXiv Paper Curator Bot!\n\n"
        "Ask me any question about ML, AI, or CS research and I will search "
        "indexed arXiv papers to answer.\n\n"
        "Example: What is self-attention in transformers?\n\n"
        "Use /help for tips."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me any research question and I will answer based on indexed arXiv papers.\n\n"
        "Tips:\n"
        "- Be specific (e.g. 'How does BERT handle masked language modeling?')\n"
        "- Ask about CS/AI/ML topics — off-topic questions are rejected by a guardrail\n"
        "- Answers may take up to 30 seconds for complex questions\n\n"
        "Commands:\n"
        "/start - Welcome message\n"
        "/help  - This help text"
    )


# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip()
    if not query:
        return

    settings = get_settings()
    api_url = f"{settings.telegram.api_base_url}/ask-agentic"

    # Show typing indicator while the RAG pipeline runs.
    # Telegram auto-cancels this after 5 s, so for slow models the indicator
    # will disappear before the answer arrives — acceptable for now.
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                api_url,
                json={
                    "query": query,
                    "top_k": 3,
                    "use_hybrid": True,
                },
            )

        if response.status_code != 200:
            await update.message.reply_text(
                f"API error {response.status_code} — is the API server running?"
            )
            return

        data = response.json()
        answer = data.get("answer", "No answer generated.")
        steps = data.get("reasoning_steps", [])
        attempts = data.get("retrieval_attempts", 0)

        # Build reply: answer + optional agent info footer
        parts = [answer]
        if steps:
            parts.append("\n--- Agent steps ---")
            parts.extend(f"- {s}" for s in steps)
        if attempts:
            parts.append(f"Retrieval attempts: {attempts}")

        text = "\n".join(parts)

        # Telegram hard limit is 4096 chars per message
        if len(text) > 4096:
            text = text[:4090] + "\n[truncated]"

        await update.message.reply_text(text)

    except httpx.RequestError as e:
        await update.message.reply_text(
            f"Could not reach the API: {e}\n\n"
            f"Make sure the API server is running at {settings.telegram.api_base_url}"
        )
    except Exception as e:
        logger.error(f"Unexpected error handling message: {e}")
        await update.message.reply_text("Something went wrong. Please try again.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()

    if not settings.telegram.enabled:
        logger.warning("Telegram bot disabled (TELEGRAM__ENABLED=false) — exiting")
        return

    if not settings.telegram.bot_token:
        logger.error("No bot token set (TELEGRAM__BOT_TOKEN) — exiting")
        return

    logger.info(
        f"Starting Telegram bot (API: {settings.telegram.api_base_url})"
    )

    app = (
        Application.builder()
        .token(settings.telegram.bot_token)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
