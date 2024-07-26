from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain

llm = ChatOpenAI(model="gpt-4")

chatbot = ConversationChain(llm=llm, verbose=True)

chatbot.predict(input="Hello")
chatbot.predict(input="Can I ask you a question? Are you an AI?")