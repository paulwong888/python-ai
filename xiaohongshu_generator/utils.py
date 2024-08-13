from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from prompt_template import system_template_text, user_template_text
from xiaohongshu_model import Xiaohongshu

def generate_xiaohongshu(topic, openai_api_key):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template_text),
        ("user", user_template_text)
    ])

    model = ChatOpenAI(api_key=openai_api_key)

    output_parser = PydanticOutputParser(pydantic_object=Xiaohongshu)

    chain = prompt | model | output_parser

    # result就是Xiaohongshu的实例
    result = chain.invoke({
        "parser_instructions": output_parser.get_format_instructions(),
        "topic": topic
    })

    return result

# from dotenv import load_dotenv
# import os

# load_dotenv()
# print(generate_xiaohongshu(topic="大模型", openai_api_key=os.getenv("OPENAI_API_KEY")))