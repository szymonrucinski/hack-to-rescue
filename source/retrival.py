import os
import pprint

from langchain.document_loaders import PyPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

loader = PyPDFLoader("../data/CPDs/CPD Somalia.pdf")
documents = loader.load()

from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=2000)
texts = text_splitter.split_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True
)
query = "What are the problems mentioned in the text?"
result = qa({"query": query})
pprint.pprint(result)
