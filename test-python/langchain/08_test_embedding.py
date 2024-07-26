from langchain.document_loaders import PyPDFLoader

pdfLoader = PyPDFLoader("/Users/mac/PAUL/WORK/WORKSPACES/PYTHON-WORKSPACE/langchain/laverne-resume.pdf")
pages = pdfLoader.load_and_split()


from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


from langchain.vectorstores import FAISS

db = FAISS.from_documents(embedding=embeddings, documents=pages)

q = "what is the phone number of Leo?"
result = db.similarity_search(q)

from langchain.chains import RetrievalQA
from langchain import OpenAI

llm = OpenAI()

chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
chain(inputs=q, return_only_outputs=True)