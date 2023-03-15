from flask import Flask, request
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "k-hbEUHDU2LjAFDwRSaYvmT3BlbkFJg31qqZfZ2VMF9CCrdORd"

app = Flask(__name__)

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

#@app.route('/')
#def index():
  #  return 'Welcome to Diani App, ask me any questions about hotels in Diani'

@app.route('/', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['input']
    vectorindex_path = create_index('Knowledge')
    response = answerMe(user_input, vectorindex_path)
    return response

if __name__ == '__main__':
    app.run()
