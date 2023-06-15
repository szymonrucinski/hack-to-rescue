from dataclasses import dataclass
import os
import pprint
import json
from langchain.document_loaders import TextLoader
import time
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQA


def get_problem_summary(documents):
    """Summarize the problem from the documents"""
    # split the documents into chunks
    print("documents", documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    problems_query = 'What are the problems mentioned in the text? List problems that need to be solved  do it in the following format: {"problem1": "description of the problem1...", "problem2":"..."} conscisely and precisely formatted as a json.'
    problems = qa({"query": problems_query})

    # pprint.pprint(summarizations)
    pprint.pprint(problems)

    # output = {"summarization": summarizations["result"], "problems": problems["result"]}

    # get answer & citation
    return problems["result"]


def get_text_summary(documents):
    """Summarize the problem from the documents"""
    time.sleep(5)
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    summarization_query = "You are the United Nations crisis meeting: summarize the mentioned text in max. 3 sentences to the UN crisis meeting. "
    summarizations = qa({"query": summarization_query})
    # pprint.pprint(summarizations)
    pprint.pprint(summarizations)
    return summarizations["result"]


def suggest(summary):
    loader = TextLoader("solution_summaries.txt")
    documents = loader.load()

    # split the documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n====================", chunk_size=500
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 32})
    # create a chain to answer questions

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    query = 'Which five of the following [[[solutions]]] best answer the problem posed in the initial (((project proposal))), and why? Consider relevance of the solutions, geography, economics, and any other potential reasons when answering. Return the 5 top solutions ranked from best to worst, in the following format:\n{"NAME": "solutionname", "CATEGORY": "solutioncategory", "RECOMMENDATION": "how to apply the solution to the specific initiatives mentioned", "OUTCOME": "the expected effects of the application of the solution in the context of the proposal", "ISSUES": "potential issues with the application of the solution to the problem or areas which this solution does not address"}. Make sure to output each solution in json format!\n\n'
    query += f"((({summary})))"

    result = qa({"query": query})

    result_text = result["result"]

    ret = []
    for r in result_text.split("\n"):
        try:
            r = json.loads(r)
            ret.append(r)
            print(r)
        except:
            pass

    return ret
