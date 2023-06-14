from dataclasses import dataclass
import os
import pprint
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "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"


@dataclass
class Recommendation:
    """Class for mapping output from  the model to a dataclass"""

    solution: str
    category: str
    recommendation: str
    outcome: str


def get_problem_summary(documents):
    """Summarize the problem from the documents"""
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    query = "What are the problems mentioned in the text?"
    result = qa({"query": query})
    pprint.pprint(result["result"])
    return result["result"]


# gets list of strings and returns list of recommendations
# def get_recommendations(list_of_pages: list[str]) -> list[Recommendation]:
#     """Mocked function to return recommendations for a user"""
#     recommendation_one = Recommendation(
#         solution="Solution 1",
#         category="Category 1",
#         recommendation="Recommendation 1",
#         outcome="Outcome 1",
#     )
#     recommendation_two = Recommendation(
#         solution="Solution 2",
#         category="Category 2",
#         recommendation="Recommendation 2",
#         outcome="Outcome 2",
#     )
#     recommendation_three = Recommendation(
#         solution="Solution 3",
#         category="Category 3",
#         recommendation="Recommendation 3",
#         outcome="Outcome 3",
#     )
#     return [recommendation_one, recommendation_two, recommendation_three]
