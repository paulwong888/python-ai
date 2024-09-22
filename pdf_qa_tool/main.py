from init_path import init
init()
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from commons.sidebar import display_sidebar, remove_button
from utils import qa_agent

st.title("📑 AI智能PDF問答工具")
st.subheader("Present by Paul Wong")

openai_api_key = display_sidebar()
remove_button()
load_dotenv()

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

has_uploaded_file = False
uploaded_file = st.file_uploader("上傳你的PDF檔案： ", type="pdf")
use_sample_file = st.checkbox("使用樣例檔案")
has_uploaded_file = use_sample_file or uploaded_file
# sample_file_button = st.button("使用样例文件")

# if use_sample_file:
#     uploaded_file = open("pdf_qa_tool/temp.pdf", "r")
question = st.text_input("對PDF的內容進行提問(例：transformer有多少層)", disabled=(not has_uploaded_file))

if not openai_api_key:
   openai_api_key = os.getenv("OPENAI_API_KEY")

# print(f"uploaded_file={uploaded_file}, has_uploaded_file={has_uploaded_file}")

if has_uploaded_file and question:
    with st.spinner("AI正在思考中，請稍候..."):
        response = qa_agent(
            uploaded_file=uploaded_file,
            question=question,
            memory=st.session_state["memory"],
            openai_api_key=openai_api_key
        )
    
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("歷史消息"):
        chat_history_list = st.session_state["chat_history"]
        for i in range(0, len(chat_history_list), 2):
            human_message = chat_history_list[i]
            ai_message = chat_history_list[i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(chat_history_list) -2:
                st.divider()