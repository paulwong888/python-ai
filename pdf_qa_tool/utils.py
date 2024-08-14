from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

def qa_agent(uploaded_file: UploadedFile, question, memory, openai_api_key):
    model = ChatOpenAI(api_key=openai_api_key)

    if uploaded_file:
        file_content = uploaded_file.read()
    else:
        file_content = open("pdf_qa_tool/temp.pdf", "rb").read()
        
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50,
        separators = ["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    embedding_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = chain.invoke(
        {"chat_history": memory, "question": question}
    )

    return response

from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory

load_dotenv()

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)
print(
    qa_agent(
        uploaded_file=None, 
        question="transformer有多少层？", 
        memory=memory, 
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )["answer"]
)