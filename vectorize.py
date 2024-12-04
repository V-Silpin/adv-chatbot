from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder

google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key = pinecone_key)

class Vectorize():
    def __init__(self):
        self.docs = None
        self.index_name = "hybrid-search-langchain-pinecone"
        self.index = pc.Index(self.index_name)
    def embedder(url):
        loader = WebBaseLoader(url)
        data = []
        data.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", r"(?<=\.)", " "], length_function=len)
        docs = text_splitter.split_documents(data)
    def store():
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
        bm25_encoder=BM25Encoder().default()
        bm25_encoder.fit(docs)
        