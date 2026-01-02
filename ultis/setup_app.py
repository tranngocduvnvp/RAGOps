import time
import uuid
import pandas as pd
from dotenv import load_dotenv
import os
# import google.generativeai as genai
from rag.core import RAG
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
# import google.generativeai as genai
from ultis.prompt_caching import PromptCache
from ultis.memory import MemoryModule
import openai
from reflection import Reflection
from re_rank import Reranker
from llms.llms import LLMs
import argparse
from insert_data import load_csv_to_chromadb
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import yaml
from types import SimpleNamespace
import uvicorn
from typing import List, Optional, Any, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultis.logging_config import setup_console_logging



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
    
    REDIS_SECRET = os.getenv('REDIS_PASSWORD', None)

    return MODEL_API_KEY, MODEL_BASE_URL, REDIS_SECRET


def setup_db_and_rag(args, llm, emb=None, logger_app=None):
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
            emb=emb
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
            emb=emb
        )
    else:

        def chromadb_collection_exists(collection_name: str, persist_dir: str = "./checkpoint_model/chroma_db") -> bool:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=persist_dir)
                collections = client.list_collections()
                return any(col.name == collection_name for col in collections)
            except Exception as e:
                logger_app.info(f"Error checking ChromaDB collection: {e}")
                return False
        def csv_exists(folder_path: str) -> list:
            """
            Check if any CSV file exists in the given folder and return their file path.
            """
            if not os.path.isdir(folder_path):
                return []

            csv_paths = [
                os.path.abspath(os.path.join(folder_path, file))
                for file in os.listdir(folder_path)
                if file.lower().endswith(".csv") and os.path.isfile(os.path.join(folder_path, file))
            ]
            return csv_paths
        
        collection_name = args.embedding_model.split('/')[-1]
        # if not chromadb_collection_exists(collection_name=collection_name):
        #     csv_files = csv_exists(folder_path="data")
        #     if len(csv_files) == 0:
        #         raise DataNotFoundError
        #     else:
        #         logger_app.info(f"The collection {collection_name} does not exist.\n")
        #         logger_app.info("Starting to create new collection. Please make sure you have a valid CSV file in data folder.\n")
        #         logger_app.info(f"Detected {len(csv_files)} csv files.\n")
        #         for i in  range(len(csv_files)):              
        #             load_csv_to_chromadb(csv_path=csv_files[i], persist_dir="./checkpoint_model/chroma_db", model_name=args.embedding_model, emb=emb)
        #             logger_app.info(f"Processed {i+1} files.\n")  
        #         logger_app.info("The data insert process is complete.")

        rag = RAG(
            type='chromadb',
            embeddingName=args.embedding_model,
            llm=llm,
            emb=emb,
            dbCollection="rag_collection"
        )
    return rag 