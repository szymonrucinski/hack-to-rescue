import os
import pprint

os.environ["OPENAI_API_KEY"] = "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"
from langchain.document_loaders import PyPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# loader = PyPDFLoader("..\data\CPDs\CPD Pakistan.pdf")
# pages = loader.load_and_split()
# text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
# embeddings = OpenAIEmbeddings()

# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings

# faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
# docs = faiss_index.similarity_search(
#     "What are the problems of the coutry mentioned in the text?", k=5
# )

# from langchain.indexes import VectorstoreIndexCreator

# index = VectorstoreIndexCreator().from_loaders([loader])
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


from langchain.document_loaders import TextLoader

loader = PyPDFLoader("..\data\CPDs\CPD Pakistan.pdf")
# pages = loader.load_and_split()
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator().from_loaders([loader])
# query = "What did the president say about Ketanji Brown Jackson"
# index.query(query)
# query = "What specific problem is this project proposing to address?"
query = "What is the estimated resource of programme?"
# x = index.query_with_sources(query)
# print(x)

from langchain.chains.question_answering import load_qa_chain

# chain = load_qa_chain(llm, chain_type="stuff")
# chain.run(input_documents=docs, question=query)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

chain = load_qa_with_sources_chain(
    llm=OpenAI(batch_size=4), chain_type="map_reduce", return_source_documents=True
)
result = chain(
    {"input_documents": loader.load(), "question": query},
    return_source_documents=True,
)
# for d in result["input_documents"]:
#     pprint.pprint(d.metadata)
pprint.pprint(result)
