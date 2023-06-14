from langchain.docstore.document import Document
import pandas as pd
import os
import pprint

os.environ["OPENAI_API_KEY"] = "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"

df = pd.read_excel(
    "..\data\Digital X metadata\Digital X Solution Catalog Metadata for Hack to the Rescue.xlsx"
)
sources = []
# iterate over each row of the dataframe
for index, row in df.iterrows():
    doc = Document(
        page_content=row["The Solution"],
    )
    # append the Document object to the list
    sources.append(doc)

print(sources)


from langchain.text_splitter import RecursiveCharacterTextSplitter

chunks = []
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ".", "!", "?", ",", " ", "<br>"], chunk_size=1024, chunk_overlap=0
)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        chunks.append(Document(page_content=chunk, metadata=source.metadata))

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

index = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key="xxx"))
