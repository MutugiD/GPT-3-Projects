import streamlit as st
from gpt_index import SimpleDirectoryReader,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os

os.environ["OPENAI_API_KEY"] = ""

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

# Define Streamlit app
def app():
    st.title("Welcome to Diani App, ask me any qiestions about hotels in Diani")
    path = 'Knowledge' # The path where the documents for the bot are stored

    # Create vector index and save to disk
    vectorindex_path = create_index(path)

    # Get user input
    user_input = st.text_input("Ask me a question:")

    if user_input:
        # Get response from the bot
        response = answerMe(user_input, vectorindex_path)
        st.write("Response:", response)

# Run the app
if __name__ == '__main__':
    app()