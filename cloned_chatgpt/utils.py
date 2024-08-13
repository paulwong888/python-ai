from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain

def get_chat_response(prompt, memory, openai_api_key):
    model = ChatOpenAI(openai_api_key=openai_api_key)

    chain = ConversationChain(
        llm=model,
        memory=memory
    )

    response = chain.invoke({"input": prompt})

    return response["response"]

# from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
# import os

# load_dotenv()

# memory = ConversationBufferMemory(return_messages=True)
# print(get_chat_response("牛顿提出过哪些知名的定律？", memory, os.getenv("OPENAI_API_KEY")))
# print(get_chat_response("我上一个问题是什么？", memory, os.getenv("OPENAI_API_KEY")))