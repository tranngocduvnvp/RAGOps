import time
import uuid
from dotenv import load_dotenv
import os
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig, CustomSentenceTransformerEmbeddings
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

# from langfuse.decorators import observe, langfuse_context 
from langfuse import Langfuse

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount

# Load environment variables from .env file
load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# setup logging config
logger_app = setup_console_logging(app_name=os.getenv("HOSTNAME"), log_level="INFO") 

# setup trace 
resource = Resource(attributes={SERVICE_NAME: "ragops"})
trace.set_tracer_provider(TracerProvider(resource=resource))

# Configure the OTLP exporter to send traces to the OpenTelemetry Collector.
otlp_exporter = OTLPSpanExporter(
    endpoint="opentelemetry-collector.monitoring.svc.cluster.local:4317",
    insecure=True  
)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Initialize the tracer.
tracer = trace.get_tracer(__name__)


# Set environment variables for API keys (add Langfuse keys)
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", None)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", None)
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", None)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", None)




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
            # sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=EmbeddingConfig(name=args.embedding_model))
            sentenceTransformerEmbedding = CustomSentenceTransformerEmbeddings(base_url="http://embeddings-embedding-service:80")
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
            reranker = Reranker(base_url="http://reranking-rerank-service:80")
        
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
    allow_origins=["*"],  
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

# Endpoint for evidently detecting data drift.
@app.get("/data_drift")
def run_evaluation():
    # global evaluation_samples
    df_reference = pd.read_csv('evaluation_samples.csv')
    df_current = pd.DataFrame({"question":productsSample})
    drift_report = Report([
        ValueDrift(column="question", method="perc_text_content_drift"),
        # DriftedColumnsCount(),
    ], include_tests=False)
    drift_snapshot = drift_report.run(df_current, df_reference)
    return drift_snapshot.dict()
    


@app.post("/v1/chat/completions")
async def openai_compatible_chat(req: OpenAIChatRequest):
    """
    Endpoint compatible with OpenAI for connecting to Open WebUI
    """
    try:
        semanticRouter = app_state.get("semanticRouter")
        llm = app_state.get("llm")
        reflection = app_state.get("reflection")
        rag = app_state.get("rag")
        reranker = app_state.get("reranker")
        memorymodule = app_state.get("memorymodule")
        cachemodule = app_state.get("cachemodule")


        model_name = req.model or getattr(llm, "model_name", "unknown-model")
        reranker_name = getattr(reranker, "reranker_name", "unknown-reranker")
        logger_app.info(f"Using model: {model_name}")

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        iduser = "001"
        passages = None
        ranked_passages = None

        if semanticRouter is None or llm is None:
            raise HTTPException(status_code=500, detail="Server pipeline not initialized")

        with langfuse.start_as_current_observation(
            as_type="span",
            name="chat request",
            input={"messages": req.messages}  
        ) as root_span:
            with langfuse.start_as_current_observation(as_type="span", name="Get Cache Response") as cache_span:
                messages = req.messages or []
                user_content = choose_message_content(messages)
                origin_query = user_content

                response, is_exact, sim = cachemodule.get_cached_response(iduser, user_content)
                if response:
                    logger_app.info("Using cache")
                    cache_span.update(
                        input={"query": user_content},
                        output={"response": response, "is_exact": is_exact, "similarity": sim},
                        metadata={"route": "cache_hit"}
                    )
                    openai_resp = build_openai_response(model_name=model_name, assistant_content=response, usage=usage)
                    openai_resp["internal"] = {
                        "route": "cache load",
                        "passages": passages,
                        "ranked_passages": ranked_passages
                    }
                    
                    root_span.update(output={"assistant_message": response})
                    return openai_resp

            with langfuse.start_as_current_observation(as_type="span", name="Get memory history") as memory_span:
                data = {"role": "user", "content": user_content}
                stm = memorymodule.get_short_term_history(iduser)
                ltm = memorymodule.search(iduser, query=user_content)
                memory_span.update(
                    input={"user_id": iduser, "query": user_content},
                    output={"short_term": stm, "long_term": ltm}
                )

            with langfuse.start_as_current_observation(as_type="generation", model=model_name, name="Rewrite query") as rewrite_span:
                logger_app.info("start rewrite query")
                (reflected_query, usage_dict_rewrite) = reflection(data, stm, ltm)
                user_content = reflected_query
                rewrite_span.update(
                    input={"original": origin_query},
                    output={"rewritten": reflected_query},
                    usage_details=usage_dict_rewrite,
                    cost_details={
                        "input": 0.00000030,
                        "cache_read_input_tokens": 0.00000003,
                        "output": 0.00000250
                    }
                )

            with langfuse.start_as_current_observation(as_type="span", name="Semantic Router") as router_span:
                logger_app.info("start semanticRouter")
                guidedRoute = semanticRouter.guide(user_content)[1]
                router_span.update(
                    input={"query": user_content},
                    output={"route": guidedRoute}
                )

                PRODUCT_ROUTE_NAME = 'products'
                CHITCHAT_ROUTE_NAME = 'chitchat'

            if guidedRoute == PRODUCT_ROUTE_NAME:
                with langfuse.start_as_current_observation(as_type="span", name="RAG pipeline") as rag_span:
                    logger_app.info("using query RAG")
                    passages = [passage for passage in rag.vector_search(user_content)]

                    with langfuse.start_as_current_observation(as_type="span", name="Reranking") as rarank_span:
                        logger_app.info("using reranking")
                        scores, ranked_passages = reranker(user_content, passages)
                        rarank_span.update(input=passages, output=ranked_passages, metadata={"scores": scores, "reranker_name":reranker_name, "route": "rerank"})

                    source_information = ""
                    for i, p in enumerate(ranked_passages):
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
                    prompt_messages_memory = [{"role": "user", "content": user_content}]

                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name="RAG LLM Generation",
                        model=model_name,  
                        input=prompt_messages
                    ) as gen:
                        logger_app.info("generate answer")
                        (assistant_response, usage_dict_answer) = rag.generate_content(prompt_messages)
                        logger_app.info("generate content succesfully")

                        gen.update(
                            output=assistant_response,
                            usage_details=usage_dict_answer,
                            metadata={"route": "rag"},
                            cost_details={
                                "input": 0.00000030,
                                "cache_read_input_tokens": 0.00000003,
                                "output": 0.00000250
                            }
                        )

                    memorymodule.ingest(iduser, prompt_messages_memory + [{"role": "assistant", "content": assistant_response}])

                    evaluation_samples['question'].append(user_content)
                    evaluation_samples['answer'].append(assistant_response)
                    evaluation_samples['contexts'].append([source_information])
                    df = pd.DataFrame(evaluation_samples)
                    df.to_csv('evaluation_samples.csv', index=False)

                    rag_span.update(metadata={"retrieved_passages": len(passages), "reranked": len(ranked_passages)})

            else:
                with langfuse.start_as_current_observation(as_type="span", name="Chatchit pipeline") as chitchat_span:
                    # Generation riêng cho chitchat LLM call
                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name="Chitchat LLM Generation",
                        model=model_name,
                        input=[data]
                    ) as gen:
                        (assistant_response, usage_dict_chitchat) = llm.generate_content([data])

                        gen.update(output=assistant_response,\
                                    usage_details=usage_dict_chitchat,\
                                    cost_details={
                                        "input": 0.00000030,
                                        "cache_read_input_tokens": 0.00000003,
                                        "output": 0.00000250
                                    },
                                    metadata={"route": "chitchat"})

                    memorymodule.ingest("001", [data] + [{"role": "assistant", "content": assistant_response}])

            cachemodule.cache_response(iduser, origin_query, assistant_response)

            openai_resp = build_openai_response(model_name=model_name, assistant_content=assistant_response, usage=usage)
            openai_resp["internal"] = {
                "route": guidedRoute,
                "passages": passages if guidedRoute == PRODUCT_ROUTE_NAME else None,
                "ranked_passages": ranked_passages if guidedRoute == PRODUCT_ROUTE_NAME else None
            }

            root_span.update(output={"assistant_message": assistant_response})

            return openai_resp

    except Exception as e:
        logger_app.exception("Error server when processing request")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__": 
    uvicorn.run("serve:app", host="0.0.0.0", port=7070, reload=True)
