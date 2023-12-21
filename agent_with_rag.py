import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI


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

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

result = agent_executor(
    {
        "input": "大統領は最新の一般教書でケタンジ・ブラウン・ジャクソンについて何と言いましたか?簡潔に答えてください。"
    }
)

print(result["output"])
