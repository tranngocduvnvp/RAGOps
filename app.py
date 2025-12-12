import time
import pandas as pd
from dotenv import load_dotenv
import os
# import google.generativeai as genai
from rag.core import RAG
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
# import google.generativeai as genai
import openai
from reflection import Reflection
from re_rank import Reranker
from llms.llms import LLMs
import argparse
from insert_data import load_csv_to_chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import yaml
from types import SimpleNamespace
import uvicorn

# Load environment variables from .env file
load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class URLNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class APINotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class ValueNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class DataNotFoundError(Exception):
    def __init__(self):
        super().__init__(f"Please make sure you have valid CSV file in folder data")

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = SimpleNamespace(**cfg)
    return args

def map_environment_variables(args):
    if args.mode == "online" and args.model_name == "gemini":
        MODEL_API_KEY = os.getenv('GEMINI_API_KEY', None)  
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('GEMINI_API_KEY')

    elif args.mode == "online" and args.model_name == "openai":
        MODEL_API_KEY = os.getenv('OPENAI_API_KEY')
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('OPENAI_API_KEY')

    elif args.mode == "online" and args.model_name == "together":
        MODEL_API_KEY = os.getenv('TOGETHER_API_KEY', None)
        MODEL_BASE_URL = os.getenv("TOGETHER_BASE_URL", None)

        if not MODEL_API_KEY:
            raise APINotFoundError('TOGETHER_API_KEY')
        if not MODEL_BASE_URL:
            raise URLNotFoundError('TOGETHER_BASE_URL')

    elif args.mode == "offline" and args.model_engine == "ollama":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("OLLAMA_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("OLLAMA_BASE_URL")
    
    elif args.mode == "offline" and args.model_engine == "vllm":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("VLLM_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("VLLM_BASE_URL")

    elif args.mode == "offline" and args.model_engine == "huggingface":
        MODEL_API_KEY = None
        MODEL_BASE_URL = None
    
    return MODEL_API_KEY, MODEL_BASE_URL

def setup_db_and_rag(args, llm):
    # Initialize RAG
    if args.db == 'qdrant':
        QDRANT_API = os.getenv('QDRANT_API', None)
        QDRANT_URL = os.getenv('QDRANT_URL', None)
        if not QDRANT_API:
            raise APINotFoundError("QDRANT_API")
        if not QDRANT_URL:
            raise URLNotFoundError("QDRANT_URL")
        
        rag = RAG(
            type='qdrant',
            qdrant_api=QDRANT_API,
            qdrant_url=QDRANT_URL,
            embeddingName=args.embedding_model,
            llm=llm,
        )

    elif args.db == 'mongodb':
        MONGODB_URI = os.getenv('MONGODB_URI')
        MONGODB_NAME = os.getenv('MONGODB_NAME')
        MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')
        if not MONGODB_URI:
            raise URLNotFoundError("MONGODB_URI")
        if (not MONGODB_NAME) or (not MONGODB_COLLECTION):
            raise ValueNotFoundError(f"MONGODB_NAME and MONGODB_COLLECTION")
        
        rag = RAG(
            type='mongodb',
            mongodbUri=MONGODB_URI,
            dbName=MONGODB_NAME,
            dbCollection=MONGODB_COLLECTION,
            embeddingName=args.embedding_model,
            llm=llm,
        )
    else:

        def chromadb_collection_exists(collection_name: str, persist_dir: str = "./chroma_db") -> bool:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=persist_dir)
                collections = client.list_collections()
                return any(col.name == collection_name for col in collections)
            except Exception as e:
                print(f"Error checking ChromaDB collection: {e}")
                return False
        def csv_exists(folder_path: str) -> list:
            """
            Check if any CSV file exists in the given folder and return their file path.

            Args:
                folder_path (str): Path to the folder (e.g., "./data")

            Returns:
                list: list of csv files.    
            """
            if not os.path.isdir(folder_path):
                return []

            csv_paths = [
                os.path.abspath(os.path.join(folder_path, file))
                for file in os.listdir(folder_path)
                if file.lower().endswith(".csv") and os.path.isfile(os.path.join(folder_path, file))
            ]
            return csv_paths
        
        if not chromadb_collection_exists(collection_name=args.embedding_model.split('/')[-1]):
            csv_files = csv_exists(folder_path="data")
            if len(csv_files) == 0:
                raise DataNotFoundError
            else:
                print(f"The collection {args.embedding_model.split('/')[-1]} does not exist.\n")
                print("Starting to create new collection. Please make sure you have a valid CSV file in data folder.\n")
                print(f"Detected {len(csv_files)} csv files.\n")
                for i in  range(len(csv_files)):              
                    load_csv_to_chromadb(csv_path=csv_files[i], persist_dir="./chroma_db", model_name=args.embedding_model)
                    print(f"Processed {i+1} files.\n")  
                print("The data insert process is complete.")

        rag = RAG(
            type='chromadb',
            embeddingName=args.embedding_model,
            llm=llm
        )
    return rag 

def setup_pipeline(args):

    # --- Semantic Router Setup --- #

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'

    sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=EmbeddingConfig(name=args.embedding_model))
    productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
    chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
    semanticRouter = SemanticRouter(sentenceTransformerEmbedding, routes=[productRoute, chitchatRoute])
    
    # --- End Semantic Router Setup --- #

    # --- Set up LLMs --- #

    MODEL_API_KEY, MODEL_BASE_URL = map_environment_variables(args)

    llm = LLMs(type=args.mode, model_version=args.model_version, model_name=args.model_name, engine=args.model_engine, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    # --- End Set up LLMs --- #

    # --- Relection Setup --- #

    reflection = Reflection(llm=llm)

    # --- End Reflection Setup --- #

    # Initialize RAG
    rag = setup_db_and_rag(args, llm)

    # Initialize ReRanker 
    reranker = Reranker(model_name=args.reranker)

    return semanticRouter, llm, reflection, rag, reranker

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG Server ....")
    args = load_config("config.yaml")
    print("Config Loaded")
    print(args)
    semanticRouter, llm, reflection, rag, reranker = setup_pipeline(args)
    app_state["semanticRouter"] = semanticRouter
    app_state["llm"] = llm
    app_state["reflection"] = reflection
    app_state["rag"] = rag
    app_state["reranker"] = reranker

    print("Load Pipeline sucessfully!")
    yield

    print("Shutdown Complete")




app = FastAPI(
    title="RAG Server",
    version="1.0.0",
    lifespan=lifespan
)

#make schema
class Query(BaseModel):
    query:str

class Response(BaseModel):
    answer: str
    route: str
    passages: list | None
    ranked_passages: list | None


@app.post("/rag", response_model=Response)
def rag_inference(request: Query):
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'
    try:
        query = request.query
        data = {"role": "user", "content": query}

        semanticRouter = app_state["semanticRouter"]
        llm = app_state["llm"]
        reflection = app_state["reflection"]
        rag = app_state["rag"]
        reranker = app_state["reranker"]

        # Reflection phase 
        # reflected_query = reflection([data])
        # query = reflected_query

        guidedRoute = semanticRouter.guide(query)[1]
        
        if guidedRoute == PRODUCT_ROUTE_NAME:
            print("Using RAG")
            # Take relevant documents from RAG system
            passages = [passage for passage in rag.vector_search(query)]

            # Rerannk retrieved documents
            scores, ranked_passages = reranker(query, passages)


            source_information = ""
            for i in range(len(ranked_passages)):
                # print(ranked_passages[i])
                source_information += f"Thông cung cấp {i+1}:\n Thông tin sản phẩm {ranked_passages[i]["combined_information"]}\n\
                    Giá sản phẩm: {ranked_passages[i]["metatdata"]["product_price"]}\n"
            
            # print("\n\nsource information:\n", source_information)
            # exit()

            combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng bán hoa. \
            Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin được cung cấp: {source_information}. Câu trả lời ngắn gọn không được bịa đặt ngoài các thông tin được cung cấp."
            data = []
            data.append({
                "role": "user",
                "content": combined_information
            })
            response = rag.generate_content(data) 
        else: 
            # Guide to LLMs
            print("Guide to LLMs")
            response = llm.generate_content([data])
            

        return Response(
            answer=response,
            route=guidedRoute,
            passages=passages,
            ranked_passages=ranked_passages
        )

    except Exception as e:
        raise HTTPException(status_code=505, detail=str(e))



if __name__ == "__main__": 
    uvicorn.run("app:app", host="0.0.0.0", port=7070, reload=True)