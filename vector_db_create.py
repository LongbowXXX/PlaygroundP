from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import chromadb
from chromadb.config import Settings

persist_directory = './tmp/chromadb'
client = chromadb.PersistentClient(path=persist_directory)

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./resources/state_of_the_union.txt', "utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
# db = Chroma.from_documents(documents, OpenAIEmbeddings())

# 新しいDBの作成
db = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    client=client,
)
db.add_documents(documents=documents, embedding=embeddings)

query = "大統領はケタンジ・ブラウン・ジャクソンについて何と言いましたか"
docs = db.similarity_search(query)
print(docs[0].page_content)
