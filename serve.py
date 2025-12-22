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
import pandas as pd
import ast
from pandas import isna
import numpy as np

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
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-d2389b4b-7a68-42b8-b1db-5fa0da5cccb3"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-6c32632d-ede0-4a10-938f-dd7b02511462"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # Or self-hosted if applicable
os.environ["OPENAI_API_KEY"] = "AIzaSyAoJU8okYfRigs2q8B_slIkFTmpe8-BGMs"

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
def run_evaluation():
    # global evaluation_samples
    df = pd.read_csv('evaluation_samples.csv')
    df["contexts"] = df["contexts"].apply(ast.literal_eval)

    # Chuyển DataFrame về dict đúng format ban đầu
    evaluation_samples = {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist(),
    }

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
            # context_precision,    # For retrieval/rerank precision
            # context_recall,       # For retrieval recall
            # answer_correctness    # Overall correctness
        ],
        llm=gemini_llm
    )

    # # Clear samples after evaluation (optional)
    # evaluation_samples = {'question': [], 'answer': [], 'contexts': [], 'ground_truth': []}
    print(result)
    # Chuyển result (là Ragas Result object) sang dict, xử lý NaN
    result_df = result.to_pandas()  # Chuyển thành DataFrame

    # Thay thế NaN bằng None (JSON sẽ hiểu là null)
    result_dict = result_df.replace({np.nan: None}).to_dict(orient="records")

    # Hoặc nếu muốn tổng hợp scores trung bình
    scores = result.scores  # Đây là dict với key là tên metric
    print(scores)
    scores_clean = {k: float(v) if not isna(v) else None for k, v in scores.items()}

    return {
        "ragas_results": {
            "individual_scores": result_dict,  # Chi tiết từng sample
            "average_scores": scores_clean,     # Trung bình các metric
            "total_samples": len(dataset)
        }
    }



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
        reranker_name = getattr(reranker, "reranker_name", "unknown-reranker")
        logger_app.info(f"Using model: {model_name}")

        # usage estimation left as zeros currently. If your LLM returns token usage, plug it here.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        iduser = "001"
        passages = None
        ranked_passages = None

        if semanticRouter is None or llm is None:
            raise HTTPException(status_code=500, detail="Server pipeline not initialized")

        with langfuse.start_as_current_observation(
        as_type="span",
        name="chat request",
        input={"messages": req.messages}  # hoặc bất kỳ input nào bạn muốn hiển thị ở trace level
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
                    # Set output cho trace khi dùng cache
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

                    # Generation riêng cho LLM call trong RAG (giả sử rag.generate_content trả về response + usage nếu có)
                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name="RAG LLM Generation",
                        model=model_name,  # thay bằng model thực tế nếu khác
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

                    memorymodule.ingress(iduser, prompt_messages_memory + [{"role": "assistant", "content": assistant_response}])

                    evaluation_samples['question'].append(user_content)
                    evaluation_samples['answer'].append(assistant_response)
                    evaluation_samples['contexts'].append([source_information])
                    df = pd.DataFrame(evaluation_samples)
                    df.to_csv('evaluation_samples.csv', index=False)

                    # Thêm metadata cho span RAG nếu cần
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

                        # Tương tự, update usage nếu có
                        gen.update(output=assistant_response,\
                                    usage_details=usage_dict_chitchat,\
                                    cost_details={
                                        "input": 0.00000030,
                                        "cache_read_input_tokens": 0.00000003,
                                        "output": 0.00000250
                                    },
                                    metadata={"route": "chitchat"})

                    memorymodule.ingress("001", [data] + [{"role": "assistant", "content": assistant_response}])

            cachemodule.cache_response(iduser, origin_query, assistant_response)

            openai_resp = build_openai_response(model_name=model_name, assistant_content=assistant_response, usage=usage)
            openai_resp["internal"] = {
                "route": guidedRoute,
                "passages": passages if guidedRoute == PRODUCT_ROUTE_NAME else None,
                "ranked_passages": ranked_passages if guidedRoute == PRODUCT_ROUTE_NAME else None
            }

            # Set output cho toàn bộ trace (assistant response cuối cùng)
            root_span.update(output={"assistant_message": assistant_response})

            return openai_resp

    except Exception as e:
        logger_app.exception("Error server when processing request")
        # keep consistent with OpenAI-like error
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__": 
    uvicorn.run("serve:app", host="0.0.0.0", port=7070, reload=True)
