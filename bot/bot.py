import logging
import os
from pathlib import Path

from chains import get_question_gen_chains
from compilation import compile
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (ApplicationBuilder, CallbackContext,
                          CallbackQueryHandler, CommandHandler,
                          ConversationHandler, MessageHandler, filters)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("./bot.log", "a")
fh.setFormatter(formatter)
logger.addHandler(fh)


MCQ_CHAIN = None
FFQ_CHAIN = None

LAST_QUESTION = None
SAVED_QUESTIONS = []

keyboard = [
    [
        InlineKeyboardButton("MCQ", callback_data="MCQ"),
        InlineKeyboardButton("FFQ", callback_data="FFQ"),
    ],
    [
        InlineKeyboardButton("Explain", callback_data="EXPLAIN"),
        InlineKeyboardButton("Compile", callback_data="COMPILE"),
    ],
    [InlineKeyboardButton("Save", callback_data="SAVE")],
]
reply_markup = InlineKeyboardMarkup(keyboard)
SELECTING_COMMAND = 1
AWAIT_MCQ = 2
AWAIT_FFQ = 3


async def load_document(update: Update, context: CallbackContext):
    global MCQ_CHAIN, FFQ_CHAIN
    file_path = Path(f"user_files/{update.effective_chat.id}/file.pdf")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file = await context.bot.get_file(update.message.document)
    await file.download_to_drive(custom_path=file_path)

    await context.bot.send_message(
        update.effective_chat.id, "File downloaded. Processing to set up..."
    )
    MCQ_CHAIN, FFQ_CHAIN = await get_question_gen_chains(file_path)
    await context.bot.send_message(
        update.effective_chat.id,
        "Document prepared. Ready to process topics.",
        reply_markup=reply_markup,
    )
    return SELECTING_COMMAND


async def generate_mcq(update: Update, context: CallbackContext):
    global MCQ_CHAIN, LAST_QUESTION
    if MCQ_CHAIN is None:
        await context.bot.send_message(
            update.effective_chat.id,
            "You must process a PDF file first",
            reply_markup=reply_markup,
        )
    topic = update.message.text
    while True:
        mcq = await MCQ_CHAIN.ainvoke({"input": topic})
        if mcq.valid:
            break
        logger.warning(f"YAML parsing failed for MCQ: {mcq.reasoning[-400:]}")

    LAST_QUESTION = mcq
    await context.bot.send_message(
        update.effective_chat.id, mcq.mcq.__repr__(), reply_markup=reply_markup
    )
    return SELECTING_COMMAND


async def generate_ffq(update: Update, context: CallbackContext):
    global FFQ_CHAIN, LAST_QUESTION
    if FFQ_CHAIN is None:
        await context.bot.send_message(
            update.effective_chat.id,
            "You must process a PDF file first",
            reply_markup=reply_markup,
        )
    topic = update.message.text
    while True:
        try:
            ffq = FFQ_CHAIN.invoke({"input": topic})
            break
        except Exception as e:
            logger.warning(f"Pasrsing failed with error {e}")
    LAST_QUESTION = ffq
    await context.bot.send_message(
        update.effective_chat.id, ffq.ffq.__repr__(), reply_markup=reply_markup
    )
    return SELECTING_COMMAND


async def explain_reasoning(update: Update, context: CallbackContext):
    global LAST_QUESTION
    if LAST_QUESTION is None:
        await context.bot.send_message(
            update.effective_chat.id,
            "You need to generate a question first!",
            reply_markup=reply_markup,
        )
    await context.bot.send_message(
        update.effective_chat.id, LAST_QUESTION.reasoning, reply_markup=reply_markup
    )
    return SELECTING_COMMAND


async def explain_sources(update: Update, context: CallbackContext):
    global LAST_QUESTION
    if LAST_QUESTION is None:
        await context.bot.send_message(
            update.effective_chat.id,
            "You need to generate a question first!",
            reply_markup=reply_markup,
        )
    await context.bot.send_message(
        update.effective_chat.id,
        "\n\n".join(
            [
                f"REFERENCE {i}:\n{source}"
                for i, source in enumerate(LAST_QUESTION.sources)
            ]
        ),
        reply_markup=reply_markup,
    )
    return SELECTING_COMMAND


async def save_question(update: Update, context: CallbackContext):
    global LAST_QUESTION, SAVED_QUESTIONS
    if LAST_QUESTION is None:
        await context.bot.send_message(
            update.effective_chat.id,
            "You need to generate a question first!",
            reply_markup=reply_markup,
        )
    SAVED_QUESTIONS.append(LAST_QUESTION)
    await context.bot.send_message(
        update.effective_chat.id, "Question added to buffer.", reply_markup=reply_markup
    )
    return SELECTING_COMMAND


async def compile_questions(update: Update, context: CallbackContext):
    global SAVED_QUESTIONS
    if len(SAVED_QUESTIONS) == 0:
        await context.bot.send_message(
            update.effective_chat.id,
            "You need to generate and save questions before compiling!",
            reply_markup=reply_markup,
        )
    path = Path(f"user_files/{update.effective_chat.id}/compilation.md")
    compile(path, SAVED_QUESTIONS)
    await context.bot.send_document(
        update.effective_chat.id, path, reply_markup=reply_markup
    )
    return SELECTING_COMMAND


async def InlineKeyboardHandler(update: Update, context: CallbackContext):
    query = update.callback_query
    if query.data == "MCQ":
        await context.bot.send_message(
            update.effective_chat.id, "Enter the topic for the MCQ"
        )
        return AWAIT_MCQ
    elif query.data == "FFQ":
        await context.bot.send_message(
            update.effective_chat.id, "Enter the topic for the FFQ"
        )
        return AWAIT_FFQ
    elif query.data == "EXPLAIN":
        return await explain_reasoning(update, context)
    elif query.data == "SAVE":
        return await save_question(update, context)
    elif query.data == "COMPILE":
        return await compile_questions(update, context)
    else:
        raise NotImplementedError("Unknown command")


async def cancel(update: Update, context: CallbackContext):
    global MCQ_CHAIN, FFQ_CHAIN, LAST_QUESTION, SAVED_QUESTIONS
    MCQ_CHAIN = None
    FFQ_CHAIN = None
    LAST_QUESTION = None
    SAVED_QUESTIONS = []
    return ConversationHandler.END


def main():
    logger.info("Bot restarted")
    app = (
        ApplicationBuilder()
        .token(os.environ["TELEGRAM_TOKEN"])
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(
        ConversationHandler(
            entry_points=[MessageHandler(filters.Document.PDF, load_document)],
            states={
                SELECTING_COMMAND: [CallbackQueryHandler(InlineKeyboardHandler)],
                AWAIT_MCQ: [MessageHandler(filters.TEXT, generate_mcq)],
                AWAIT_FFQ: [MessageHandler(filters.TEXT, generate_ffq)],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
        )
    )
    logger.info("Ready to poll")
    app.run_polling()
    logger.info("Polling")


if __name__ == "__main__":
    main()
