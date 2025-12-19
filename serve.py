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

def setup_db_and_rag(args, llm, emb=None):
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
        if not chromadb_collection_exists(collection_name=collection_name):
            csv_files = csv_exists(folder_path="data")
            if len(csv_files) == 0:
                raise DataNotFoundError
            else:
                print(f"The collection {collection_name} does not exist.\n")
                print("Starting to create new collection. Please make sure you have a valid CSV file in data folder.\n")
                print(f"Detected {len(csv_files)} csv files.\n")
                for i in  range(len(csv_files)):              
                    load_csv_to_chromadb(csv_path=csv_files[i], persist_dir="./chroma_db", model_name=args.embedding_model, emb=emb)
                    print(f"Processed {i+1} files.\n")  
                print("The data insert process is complete.")

        rag = RAG(
            type='chromadb',
            embeddingName=args.embedding_model,
            llm=llm,
            emb=emb
        )
    return rag 

def setup_pipeline(args):

    # --- Semantic Router Setup --- #

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'

    print("Load model sentenceTransformerEmbedding embedding")
    sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=EmbeddingConfig(name=args.embedding_model))
    productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
    chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
    semanticRouter = SemanticRouter(sentenceTransformerEmbedding, routes=[productRoute, chitchatRoute])
    
    # --- End Semantic Router Setup --- #

    # --- Set up LLMs --- #

    MODEL_API_KEY, MODEL_BASE_URL = map_environment_variables(args)

    llm = LLMs(type=args.mode, model_version=args.model_version, model_name=args.model_name, engine=args.model_engine, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    # --- End Set up LLMs --- #

    # --- Reflection Setup --- #

    reflection = Reflection(llm=llm)

    # --- End Reflection Setup --- #

    # Initialize RAG
    rag = setup_db_and_rag(args, llm, emb=sentenceTransformerEmbedding)

    print("Load model Reranker")
    # Initialize ReRanker 
    reranker = Reranker(model_name=args.reranker)
    memorymodule = MemoryModule(sentenceTransformerEmbedding, reranker, args.faiss_base_path, max_short_term_messages=args.max_short_term_messages)

    cachemodule = PromptCache(
        embedding_instance=sentenceTransformerEmbedding,
        expire_seconds=args.expire_seconds,
        similarity_threshold=args.similarity_threshold
    )


    return semanticRouter, llm, reflection, rag, reranker, memorymodule, cachemodule

app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG Server ....")
    args = load_config("config.yaml")
    print("Config Loaded")
    semanticRouter, llm, reflection, rag, reranker, memorymodule, cachemodule = setup_pipeline(args)
    app_state["semanticRouter"] = semanticRouter
    app_state["llm"] = llm
    app_state["reflection"] = reflection
    app_state["rag"] = rag
    app_state["reranker"] = reranker
    app_state["memorymodule"] = memorymodule
    app_state["cachemodule"] = cachemodule

    print("Load Pipeline successfully!")
    yield
    print("Shutdown Complete")

app = FastAPI(
    title="RAG Server (OpenAI-compatible)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:3000"] nếu cần
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Helper: map OpenAI Chat request to our pipeline ----
def choose_message_content(messages: List[Dict[str, str]]) -> str:
    """
    From OpenAI-style messages array, pick a reasonable user message:
    - prefer the last message with role 'user'
    - fallback to last message content
    """
    if not messages:
        return ""
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    # fallback
    return messages[-1].get("content", "")

def build_openai_response(model_name: str, assistant_content: str, usage=None) -> Dict[str, Any]:
    """
    Build a minimal OpenAI-compatible completions response.
    """
    now_ts = int(time.time())
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now_ts,
        "model": model_name,
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": assistant_content
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
    return resp

# Raw Pydantic model for OpenAI-like request (we accept flexible data)
class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint
    """
    models = [
        {
            "id": "shopping-rag",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "permission": [],
            "root": "rag",
            "parent": None
        },
        
    ]
    return JSONResponse(content={"object": "list", "data": models})

@app.post("/v1/chat/completions")
async def openai_compatible_chat(req: OpenAIChatRequest):
    """
    Endpoint compatible with OpenAI Chat Completions:
    Accepts JSON with 'model' and 'messages' and returns a JSON similar to OpenAI response.
    Internally uses your semantic router + RAG pipeline.
    """
    try:
        # Grab pipeline
        semanticRouter = app_state.get("semanticRouter")
        llm = app_state.get("llm")
        reflection = app_state.get("reflection")
        rag = app_state.get("rag")
        reranker = app_state.get("reranker")
        memorymodule = app_state.get("memorymodule")
        cachemodule = app_state.get("cachemodule")

        # Build OpenAI-like response
        model_name = req.model or getattr(llm, "model_name", "unknown-model")
        # usage estimation left as zeros currently. If your LLM returns token usage, plug it here.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        iduser = "001"
        passages = None
        ranked_passages = None

        if semanticRouter is None or llm is None:
            raise HTTPException(status_code=500, detail="Server pipeline not initialized")

        messages = req.messages or []
        user_content = choose_message_content(messages)
        origin_query = user_content

        response, is_exact, sim = cachemodule.get_cached_response(iduser, user_content)
        if response:
            openai_resp = build_openai_response(model_name=model_name, assistant_content=response, usage=usage)
            # additionally, include our internal route and passages in top-level field "internal" for debugging (optional)
            openai_resp["internal"] = {
                "route": "cache load",
                "passages": passages,
                "ranked_passages": ranked_passages
            }
            return openai_resp

        # Build data for internal pipeline (keeps compatibility with original code)
        data = {"role": "user", "content": user_content}

        stm = memorymodule.get_short_term_history(iduser)
        ltm = memorymodule.search(iduser, query=user_content)
        print("start rewrite query") 
        reflected_query = reflection(data, stm, ltm)
        user_content = reflected_query
        

        # Decide route via semantic router
        print("start semanticRouter") 
        guidedRoute = semanticRouter.guide(user_content)[1]

        PRODUCT_ROUTE_NAME = 'products'
        CHITCHAT_ROUTE_NAME = 'chitchat'

        

        if guidedRoute == PRODUCT_ROUTE_NAME:
            print("using query RAG")
            # RAG path
            passages = [passage for passage in rag.vector_search(user_content)]
            # Re-rank
            print("using reranking")
            scores, ranked_passages = reranker(user_content, passages)

            # Build source information safely (fix nested quoting)
            source_information = ""
            for i, p in enumerate(ranked_passages):
                # defensive extraction - some keys may differ depending on your RAG metadata shape
                combined_info = p.get("combined_information", str(p.get("content", "")))
                metadata = p.get("metatdata", p.get("metadata", {}))
                price = metadata.get("product_price", metadata.get("price", "N/A"))
                source_information += f"Thông cung cấp {i+1}:\n Thông tin sản phẩm: {combined_info}\n Giá sản phẩm: {price}\n\n"

            combined_information = (
                "Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng bán hoa. "
                f"Câu hỏi của khách hàng: {user_content}\n"
                f"Trả lời câu hỏi dựa vào các thông tin được cung cấp: {source_information}. "
                "Câu trả lời ngắn gọn không được bịa đặt ngoài các thông tin được cung cấp."
            )
            prompt_messages = [{"role": "user", "content": combined_information}]
            # Use RAG.generate_content (keeps original behaviour)
            print("generate answer")
            assistant_response = rag.generate_content(prompt_messages)
            print("generate content succesfully")
            memorymodule.ingress(iduser, prompt_messages + [{"role": "assistant", "content": assistant_response}])
        else:
            # LLM path (chitchat or default)
            assistant_response = llm.generate_content([data])
            memorymodule.ingress("001", [data] + [{"role": "assistant", "content": assistant_response}])

        cachemodule.cache_response(iduser, origin_query, assistant_response)
        

        openai_resp = build_openai_response(model_name=model_name, assistant_content=assistant_response, usage=usage)
        # additionally, include our internal route and passages in top-level field "internal" for debugging (optional)
        openai_resp["internal"] = {
            "route": guidedRoute,
            "passages": passages,
            "ranked_passages": ranked_passages
        }

        return openai_resp

    except Exception as e:
        # keep consistent with OpenAI-like error
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__": 
    uvicorn.run("serve:app", host="0.0.0.0", port=7070, reload=True)
