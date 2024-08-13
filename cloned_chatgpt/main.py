import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from utils import get_chat_response
from dotenv import load_dotenv

from init_path import init

# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
# parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# print(parent_dir_path)
# sys.path.insert(0, parent_dir_path)
init()
from commons.sidebar import display_sidebar, remove_button

remove_button()
load_dotenv()

st.title("克隆ChatGPT")

openai_api_key = display_sidebar()

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
    st.session_state["messages"] = [
        {"role": "ai", "content": "您好，我是您的AI助手，有什么可以帮到您？"}
    ]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    st.session_state["messages"].append(
        {"role": "human", "content": prompt}
    )
    st.chat_message("human").write(prompt)

    with st.spinner("AI正在思考中，请稍等..."):
        response = get_chat_response(prompt, st.session_state["memory"], openai_api_key)

    st.session_state["messages"].append(
        {"role": "ai", "content": response}
    )
    st.chat_message("ai").write(response)