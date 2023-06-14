


import os
import pprint

os.environ["OPENAI_API_KEY"] = "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"
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import pandas as pd
import requests

def request_chatgpt(prompt):
    # Define your API key
    api_key = "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"

    # Define the API endpoint URL
    endpoint = "https://api.openai.com/v1/chat/completions"

    # Define the API request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Define the API request data
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [{"role": "user", "content": prompt}]
    }

    # Send the API request
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=80)
    except Exception:
        return ""
    response_text = ""

    # Check the API response status code
    if response.status_code == 200:
        # Print the API response text
        response_text = response.json()['choices'][0]['message']['content']
        # print(response_text)
    else:
        # Print the API error message
        print(f"Request failed with status code: {response.status_code}")

    return response_text


data = pd.read_excel("data/data.xlsx")
# data = data[["Solution Name", "The Problem", "The Solution"]]
list_data = data.values.tolist()


loader = PyPDFLoader("data\CPDs\CPD Somalia.pdf")
documents = loader.load()

text = " ".join([_.page_content for _ in documents])

prompt = f"Summarize the following text (which has been filtered of stop words) in triple brackets in about 5000 words:\n\n((({text})))"

out = request_chatgpt(prompt)

print(out)

prompt = f"((({out})))\n"

for d in list_data:
    prompt += "\n\n\n[[["

    for cat, datapoint in zip(data.columns, d):
        prompt += f"\n{cat.upper()} : {datapoint}\n"

    prompt += "]]]\n=====\n"

with open("temp_text.txt", "w+", encoding="utf-8") as f:
    f.write(prompt)

from langchain.document_loaders import TextLoader

loader = TextLoader("solution_summaries.txt")
documents = loader.load()

# split the documents into chunks
text_splitter = CharacterTextSplitter(separator="\n\n====================", chunk_size=500)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
print(texts)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 8})
# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True
)
query = f"Which five of the following [[[solutions]]] best answer the problem posed in the initial (((project proposal))), and why? Consider relevance of the solutions, geography, economics, and any other potential reasons when answering. Return the 5 top solutions ranked from best to worst.\n\n((({out})))"
result = qa({"query": query})
pprint.pprint(result)
