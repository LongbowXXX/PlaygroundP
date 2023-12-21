#!/usr/bin/env python
import chromadb
from fastapi import FastAPI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.pydantic_v1 import BaseModel
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

chat_model = ChatOpenAI(
    model_name="gpt-4-1106-preview"
)

prompt = ChatPromptTemplate.from_template("{topic}についての冗談を教えてください")
add_routes(
    app,
    prompt | chat_model,
    path="/joke",
)

persist_directory = './tmp/chromadb'
client = chromadb.PersistentClient(path=persist_directory)
embeddings = OpenAIEmbeddings()
db = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    client=client,
)

retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "一般教書に関する文書を検索して返却します。",
)
tools = [tool]

llm = ChatOpenAI(temperature=0)


class AgentInput(BaseModel):
    input: str


agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
new_executor = agent_executor.with_types(
    input_type=AgentInput
)

add_routes(
    app,
    new_executor | (lambda x: x["output"]),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
