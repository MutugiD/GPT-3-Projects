from flask import Flask
from gpt_index import SimpleDirectoryReader,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os

app = Flask(__name__)

@app.route('/test', methods=['POST'])

def answerMe(vectorindex):
  vIndex = GPTSimpleVectorIndex.load_from_disk(vectorindex)
  while True:
    prompt = input("Please ask: ")
    response = vIndex.query(prompt,response_mode="compact")
    print(f"Response: {response} \n ")
    answerMe('vectorIndex.json')
    
    
if __name__ == '__main__':
    app.run(debug=True)
    


