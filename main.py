from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
import logging

# ログの設定：デバッグレベル以上をコンソールに出力
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

chat_model = ChatOpenAI(
    model_name="gpt-4-1106-preview"
)

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

logging.debug("chat model in")
response = chat_model.invoke(messages)
logging.debug("chat model out")
print(response)
# >> AIMessage(content="Socks O'Color")
