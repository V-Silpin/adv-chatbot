from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

class Vectorize():
    def __init__(self):
        self.docs = None
        load_dotenv()
        self.index_name = os.getenv("INDEX_NAME")
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
    def chunker(self, url):
        loader = WebBaseLoader(url)
        data = []
        data.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", r"(?<=\.)", " "], length_function=len)
        self.docs = text_splitter.split_documents(data)
        print("Processing Done")
    def store(self):
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key, model="models/embedding-001")
        pc = Pinecone(api_key = self.pinecone_key)
        self.index = pc.Index(self.index_name)
        bm25_encoder=BM25Encoder().default()
        retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=self.index)
        texts = [doc.page_content for doc in self.docs]
        retriever.add_texts(texts)
        print("Storage Done")