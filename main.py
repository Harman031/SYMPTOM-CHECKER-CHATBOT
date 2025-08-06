from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os

os.environ["TRANSFORMER_NO_TF"] = "1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

class SBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding_model = SBERTEmbeddings("all-MiniLM-L6-v2")

dataset = [
    "Cramping",
    "Sharp, stabbing pain",
    "Bloating",
    "Nausea",
    "Vomiting",
    "Fever",
    "Loss of appetite",
    "Pain that comes and goes",
    "Pain that worsens with movement",
    "Pain with urination",
    "Yellowing of the skin (jaundice)",
    "When to seek immediate medical care: Severe pain, blood in stool, persistent vomiting, chest pain, yellow skin, swelling in abdomen.",
    "Common causes: Gas, indigestion, food poisoning, stomach virus, menstrual cramps, urinary tract infection, gallstones, appendicitis."
]

docs = [Document(page_content=text) for text in dataset]

vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="chroma_db")

prompt_template = """
You are a medical chatbot for abdominal pain in adults.
You have access to the Mayo Clinic abdominal pain dataset.
Answer ONLY from this dataset.
If the user asks something outside this dataset, say:
"I don't know. Please ask about abdominal pain symptoms."

Question: {question}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["question"])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": PROMPT}
)


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

class Query(BaseModel):
    text: str

@app.post("/chat")
def chat(query: Query):
    result = qa_chain.run(query.text)
    return {"answer": result}