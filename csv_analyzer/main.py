import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from utils import dataframe_agent
from streamlit.runtime.uploaded_file_manager import UploadedFile
from init_path import init

init()
from commons.sidebar import display_sidebar, remove_button

remove_button()
load_dotenv()

def create_char(response_dict):
    chart_type = response_dict["chart_type"]
    input_data = response_dict["datas"]
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    if chart_type == "scatter":
        st.scatter_chart(df_data)

st.title("💡 CSV資料分析智能工具")
st.subheader("Present by Paul Wong")

openai_api_key = display_sidebar()

data = st.file_uploader("上傳你的資料檔案(CSV格式)", type="csv")
sample_data_button = st.button("使用樣例資料檔案")

if data:
    print(type(data))
    st.session_state["df"] = pd.read_csv(data)
elif sample_data_button:
    sample_data = "csv_analyzer/personal_data.csv"
    st.session_state["df"] = pd.read_csv(sample_data)

if "df" in st.session_state:
    with st.expander("原始數據"):
        st.dataframe(st.session_state["df"])

query = st.text_area(
    "請輸入你關於以上表格的問題，或數據選取請求，或視覺化要求(支持散點圖，折線圖，條形圖)，如請選取年齡大於30的數據 / 繪製出職業的條形圖 / 繪製出客戶年收入和年齡之間的散點圖",
    value="繪製出職業的條形圖"
)
button = st.button("生成回答")

if button:
    if not openai_api_key:
        openai_api_key=os.getenv("OPENAI_API_KEY")
    if "df" not in st.session_state:
        st.info("請先上傳資料檔案")
        st.stop()
    if not query:
        st.info("請輸入要統計的問題")
        st.stop()
    if "df" in st.session_state:
        with st.spinner("AI正在思考中，請稍等..."):
            response_dict = dataframe_agent(openai_api_key, st.session_state["df"], query)
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        elif "table" in response_dict:
            st.table(
                pd.DataFrame(
                    response_dict["table"]["data"],
                    columns=response_dict["table"]["columns"]
                )
            )
        else:
            create_char(response_dict)
        