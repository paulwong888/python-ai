from langchain_openai import ChatOpenAI
from my_config import set_environment

#https://python.langchain.com/v0.2/docs/integrations/chat/openai/

set_environment()

llm = ChatOpenAI(
    #model="gpt-3.5-turbo",
    #temperature=0,
    #max_tokens=None,
    #timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "tell me a joke."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)