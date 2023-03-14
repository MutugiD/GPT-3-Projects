from gpt_index import SimpleDirectoryReader,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os
import time

# Set up OpenAI API credentials
os.environ["OPENAI_API_KEY"] =  "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"

def create_index(path):
    max_input = 4096
    tokens = 200
    chunk_size = 600 #for LLM, we need to define chunk size
    max_chunk_overlap = 20
  
    #define prompt
    promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size) 
    #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001",max_tokens=tokens))
    #load data — it will take all the .txtx files, if there are more than 1
    docs = SimpleDirectoryReader(path).load_data()
    #create vector index
    vectorindex = GPTSimpleVectorIndex(documents=docs,llm_predictor=llmPredictor,prompt_helper=promptHelper)
    vectorindex.save_to_disk("vectorIndex.json")
    return "vectorIndex.json"
def answerMe(prompt, vectorindex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorindex)
    response = vIndex.query(prompt,response_mode="compact")
    return response

# Define a function to run the conversational loop
def run_conversation():
    # Initialize the conversation
    print("Bot: Hi, how can I help you today?")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Generate a response using GPT-3
        prompt = "Me: " + user_input + "\nBot:"
        bot_response = answerMe(prompt)
        
        # Print the bot's response
        print("Bot:", bot_response)
        
        # Pause for a moment to give the user a chance to respond
        time.sleep(1)

# Run the conversational loop
run_conversation()
