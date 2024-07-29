from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

for s in chain.stream({"topic": "bears"}):
    print(s.content, end="", flush=True)
print("\n")