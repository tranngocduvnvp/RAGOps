import time
import uuid
from dotenv import load_dotenv
import os
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
from ultis.prompt_caching import PromptCache
from ultis.memory import MemoryModule
from reflection import Reflection
from re_rank import Reranker
from llms.llms import LLMs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Optional, Any, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultis.logging_config import setup_console_logging
from ultis.setup_app import load_config, map_environment_variables, setup_db_and_rag
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME 

from langchain_openai import ChatOpenAI
from langfuse import observe
# from langfuse.decorators import observe, langfuse_context 
from langfuse import Langfuse
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset  # From Hugging Face datasets

# Load environment variables from .env file
load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# setup logging config
logger_app = setup_console_logging(app_name=os.getenv("HOSTNAME"), log_level="INFO") 

# setup trace 
# Khởi tạo TracerProvider với resource (tên service)
resource = Resource(attributes={SERVICE_NAME: "my-fastapi-app"})
trace.set_tracer_provider(TracerProvider(resource=resource))

# Cấu hình OTLP exporter để gửi traces đến OpenTelemetry Collector
# Thay 'otel-collector.monitoring.svc.cluster.local:4317' bằng service thực tế của collector (sẽ deploy sau)
otlp_exporter = OTLPSpanExporter(
    endpoint="otel-collector-opentelemetry-collector.monitoring.svc.cluster.local:4317",
    insecure=True  # Sử dụng insecure cho dev; enable TLS cho production
)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Khởi tạo tracer
tracer = trace.get_tracer(__name__)


# Set environment variables for API keys (add Langfuse keys)
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # If using OpenAI for Ragas/LLM
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-ff3cd06f-79f7-456e-8b7a-94aa51caba5c"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-e0bab1a6-cc75-4f96-a07f-623d95e56567"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # Or self-hosted if applicable

gemini_llm = ChatOpenAI(
    model="gemini-2.0-flash-lite",
    openai_api_key="AIzaSyAoJU8okYfRigs2q8B_slIkFTmpe8-BGMs",
    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0
)


langfuse = Langfuse()

def setup_pipeline(args):
    # --- Semantic Router Setup --- #

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'
    
    with tracer.start_as_current_span("setup_pipeline") as processors:
        with tracer.start_as_current_span("Load model sentenceTransformerEmbedding embedding",\
                                          links=[trace.Link(processors.get_span_context())]):
            logger_app.info("Load model sentenceTransformerEmbedding embedding")
            sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=EmbeddingConfig(name=args.embedding_model))
            productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
            chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
            semanticRouter = SemanticRouter(sentenceTransformerEmbedding, routes=[productRoute, chitchatRoute])
        
        # --- End Semantic Router Setup --- #
        
        with tracer.start_as_current_span("Set up LLMs",\
                                          links=[trace.Link(processors.get_span_context())]):
            # --- Set up LLMs --- #

            MODEL_API_KEY, MODEL_BASE_URL, REDIS_SECRET = map_environment_variables(args)

            llm = LLMs(type=args.mode, model_version=args.model_version, model_name=args.model_name, engine=args.model_engine, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

            # --- End Set up LLMs --- #

            # --- Reflection Setup --- #

            reflection = Reflection(llm=llm)

            # --- End Reflection Setup --- #

        with tracer.start_as_current_span("Set up database and RAG",\
                                          links=[trace.Link(processors.get_span_context())]):
            # Initialize RAG
            rag = setup_db_and_rag(args, llm, emb=sentenceTransformerEmbedding, logger_app=logger_app)

            logger_app.info("Load model Reranker")
        
        with tracer.start_as_current_span("Set up Reranker model",\
                                          links=[trace.Link(processors.get_span_context())]):
            # Initialize ReRanker 
            reranker = Reranker(model_name=args.reranker)
        
        with tracer.start_as_current_span("Set up Memory module and PromptCache",\
                                          links=[trace.Link(processors.get_span_context())]):
            memorymodule = MemoryModule(sentenceTransformerEmbedding, reranker, args.faiss_base_path, max_short_term_messages=args.max_short_term_messages)

            cachemodule = PromptCache(
                redis_host="redis-master.redis.svc.cluster.local",
                embedding_instance=sentenceTransformerEmbedding,
                expire_seconds=args.expire_seconds,
                similarity_threshold=args.similarity_threshold,
                password=REDIS_SECRET
            )

    return semanticRouter, llm, reflection, rag, reranker, memorymodule, cachemodule



app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger_app.info("Starting RAG Server ....")
    args = load_config("config.yaml")
    logger_app.info("Config Loaded")
    semanticRouter, llm, reflection, rag, reranker, memorymodule, cachemodule = setup_pipeline(args)
    app_state["semanticRouter"] = semanticRouter
    app_state["llm"] = llm
    app_state["reflection"] = reflection
    app_state["rag"] = rag
    app_state["reranker"] = reranker
    app_state["memorymodule"] = memorymodule
    app_state["cachemodule"] = cachemodule
    logger_app.info("Load Pipeline successfully!")
    yield
    logger_app.info("Shutdown Complete")

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

evaluation_samples = {
    'question': [],
    'answer': [],
    'contexts': [],
    # 'ground_truth': []
}

# New endpoint to trigger Ragas evaluation (for manual or periodic evaluation)
@app.get("/evaluate")
async def run_evaluation():
    global evaluation_samples
    if not evaluation_samples['question']:
        return JSONResponse(content={"message": "No samples collected yet."})

    dataset = Dataset.from_dict(evaluation_samples)

    # Evaluate with Ragas
    # Note: This uses OpenAI by default; configure if needed
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,         # For generation quality
            answer_relevancy,     # For generation relevancy
            context_precision,    # For retrieval/rerank precision
            # context_recall,       # For retrieval recall
            # answer_correctness    # Overall correctness
        ],
        llm=gemini_llm
    )

    # Clear samples after evaluation (optional)
    evaluation_samples = {'question': [], 'answer': [], 'contexts': [], 'ground_truth': []}

    return JSONResponse(content={"ragas_results": result})

# Wrap phases with Langfuse for monitoring
@observe(name="Query Phase")
def monitored_retrieve(rag, query):
    passages = [passage for passage in rag.vector_search(query)]
    return passages

@observe(name="Rerank Phase", capture_input=False)
def monitored_rerank(reranker, query, passages):
    scores, ranked_passages = reranker(query, passages)
    return scores, ranked_passages

@observe(name="Generation Phase", as_type="generation")
def monitored_generate(rag, prompt_messages):
    response = rag.generate_content(prompt_messages)
    return response


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

        with tracer.start_as_current_span("chat request") as processors:
            with tracer.start_as_current_span("Get Cache Respone ",\
                                          links=[trace.Link(processors.get_span_context())]):
                
                messages = req.messages or []
                user_content = choose_message_content(messages)
                origin_query = user_content


                response, is_exact, sim = cachemodule.get_cached_response(iduser, user_content)
                if response:
                    logger_app.info("Using cache")
                    openai_resp = build_openai_response(model_name=model_name, assistant_content=response, usage=usage)
                    # additionally, include our internal route and passages in top-level field "internal" for debugging (optional)
                    openai_resp["internal"] = {
                        "route": "cache load",
                        "passages": passages,
                        "ranked_passages": ranked_passages
                    }
                    return openai_resp

            with tracer.start_as_current_span("Get memory history ",\
                                          links=[trace.Link(processors.get_span_context())]):
                # Build data for internal pipeline (keeps compatibility with original code)
                data = {"role": "user", "content": user_content}

                stm = memorymodule.get_short_term_history(iduser)
                ltm = memorymodule.search(iduser, query=user_content)
        
            with tracer.start_as_current_span("Rewrite query",\
                                          links=[trace.Link(processors.get_span_context())]):
                logger_app.info("start rewrite query") 
                reflected_query = reflection(data, stm, ltm)
                user_content = reflected_query
            
        
            with tracer.start_as_current_span("Semantic Router",\
                                          links=[trace.Link(processors.get_span_context())]):
                # Decide route via semantic router
                logger_app.info("start semanticRouter") 
                guidedRoute = semanticRouter.guide(user_content)[1]

                PRODUCT_ROUTE_NAME = 'products'
                CHITCHAT_ROUTE_NAME = 'chitchat'

        

            if guidedRoute == PRODUCT_ROUTE_NAME:
                with tracer.start_as_current_span("RAG pipeline ",\
                                          links=[trace.Link(processors.get_span_context())]):
                    
                    logger_app.info("using query RAG")
                    # RAG path
                    # passages = [passage for passage in rag.vector_search(user_content)]
                    # Monitored retrieve
                    passages = monitored_retrieve(rag, user_content)
                    # Re-rank
                    logger_app.info("using reranking")
                    scores, ranked_passages = monitored_rerank(reranker, user_content, passages)
                    # scores, ranked_passages = reranker(user_content, passages)

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
                    logger_app.info("generate answer")
                    assistant_response = monitored_generate(rag, prompt_messages)
                    # assistant_response = rag.generate_content(prompt_messages)
                    logger_app.info("generate content succesfully")
                    memorymodule.ingress(iduser, prompt_messages + [{"role": "assistant", "content": assistant_response}])

                    # Collect for Ragas evaluation (add ground_truth manually or via feedback loop in production)
                    evaluation_samples['question'].append(user_content)
                    evaluation_samples['answer'].append(assistant_response)
                    evaluation_samples['contexts'].append([source_information])  # Contexts as list of strings
                    # evaluation_samples['ground_truth'].append("Ground truth here")  # Replace with actual ground truth
            else:
                with tracer.start_as_current_span("Chatchit pipeline ",\
                                          links=[trace.Link(processors.get_span_context())]):
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
        logger_app.exception("Error server when processing request")
        # keep consistent with OpenAI-like error
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__": 
    uvicorn.run("serve:app", host="0.0.0.0", port=7070, reload=True)
