from telegram.ext import Updater, CommandHandler, MessageHandler, filters
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import telegram
import os

os.environ["OPENAI_API_KEY"] = "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"

def create_index(path):
    max_input = 4096
    tokens = 200
    chunk_size = 600 #for LLM, we need to define chunk size
    max_chunk_overlap = 20
  
    #define prompt
    promptHelper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
  
    #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
  
    #load data — it will take all the .txtx files, if there are more than 1
    docs = SimpleDirectoryReader(path).load_data()

    #create vector index
    vectorindex = GPTSimpleVectorIndex(documents=docs, llm_predictor=llmPredictor, prompt_helper=promptHelper)
    vectorindex.save_to_disk("vectorIndex.json")
    return "vectorIndex.json"

def answer_me(prompt, vectorindex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorindex)
    response = vIndex.query(prompt, response_mode="compact")
    return response

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hi! I'm an AI-powered chatbot. You can ask me anything!")

def help(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="You can ask me anything and I'll do my best to answer!")

def handle_message(update, context):
    path = 'Knowledge' # The path where the documents for the bot are stored

    # Create vector index and save to disk
    vectorindex_path = create_index(path)
    # Get user input
    user_input = update.message.text.strip()

    # Get response from the bot
    response = answer_me(user_input, vectorindex_path)

    # Send the response back to the user
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

# Define Telegram bot
# Replace YOUR_BOT_TOKEN with your bot token
bot_token = '6220985502:AAEQAkZr-DE8rXRYApoAD5FVCNBM6LOSzP8'

# Create a bot instance
bot = telegram.Bot(token=bot_token)

# Create an updater instance
updater = Updater(bot=bot, use_context=True)

# Get the dispatcher to register handlers
dispatcher = updater.dispatcher

# Register command handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help))

# Register message handler
dispatcher.add_handler(MessageHandler(filters.text & ~filters.command, handle_message))

# Define the main function
def main():
    path = 'Knowledge' # The path where the documents for the bot are stored

    # Create vector index and save to disk
    vectorindex_path = create_index(path)

    # Start the bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

