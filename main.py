import os
import re
import json
from tqdm import tqdm
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# ========== ENV VARIABLES ==========
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API = os.getenv("QDRANT_API")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ========== CLEANING ==========
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


# ========== EMBEDDING + VECTORSTORE ==========
embed_model = FastEmbedEmbedding(model_name="thenlper/gte-large")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)

# Create collection if not exists
try:
    client.get_collection("tutor")
except:
    client.create_collection(
        collection_name="tutor",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="tutor",
    embedding=embed_model,
    stores_text=True,
)

# ========== INDEX INIT ==========
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


# ========== LOAD DATA ==========
# def load_data(json_path="test.json"):
#     with open(json_path, "r") as f:
#         data = json.load(f)
#
#     nodes = []
#     for item in data:
#         content = f"Problem: {item['Problem']} \nFormula: {item['annotated_formula']} \nSteps: {item['linear_formula']}"
#         content = clean_text(content)
#
#         metadata = {
#             "category": item.get("category", ""),
#             "correct": item.get("correct", ""),
#             "options": item.get("options", ""),
#             "rationale": item.get("Rationale", ""),
#         }
#
#         metadata = {k: clean_text(v) if isinstance(v, str) else v for k, v in metadata.items()}
#         node = TextNode(text=content, metadata=metadata)
#         nodes.append(node)
#
#     batch_size = 50
#     for i in tqdm(range(0, len(nodes), batch_size)):
#         batch = nodes[i:i + batch_size]
#         index.insert_nodes(batch)


# ========== SETUP LLM ==========
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
react_template = """
You are a helpful and safe AI assistant designed specifically to answer mathematics-related questions.
You must follow input and output guardrails to ensure educational integrity, accuracy, and user safety.

## TOOL USAGE STRATEGY:
- Always try to answer using the `MathTutor` tool first.
- If the answer is not found or unclear, fall back to the `search` tool.
- If the query is unrelated to mathematics, respond with a message: "This query is not related to mathematics."

## INPUT GUARDRAILS:
- Only accept questions that are related to mathematics (e.g., algebra, geometry, calculus, statistics).
- If the query is unrelated to mathematics, politely decline to answer and guide the user back.
- Do not accept questions that ask for personal advice, opinions, or anything unethical.
- Avoid hallucinations — rely on CONTEXT or Web Search only.
- If the question contains inappropriate or harmful content, block the response.

## OUTPUT GUARDRAILS:
- Your answers must be safe, factual, and age-appropriate.
- Always respond in a respectful and educational tone.
- Begin by briefly restating the question in your own words for clarity.
- Then provide a **step-by-step explanation** of the solution.
- If the answer is not found in the CONTEXT, trigger a web search module (if available).
- If it still cannot be answered accurately, say “I don’t know” instead of guessing.
- Encourage students to ask follow-up questions if clarification is needed.

---------------------
CONTEXT: {context}
---------------------
QUESTION: {query}

Based on the above, provide a step-by-step mathematical explanation:
"""
llm_with_guardrails = LangChainLLM(llm=llm, system_prompt=react_template)
Settings.llm = llm_with_guardrails

# ========== QUERY ENGINE ==========
retriever = index.as_retriever(search_kwargs={"k": 4})
query_engine = RetrieverQueryEngine(retriever=retriever)

# ========== TOOLS ==========
math_tool = FunctionTool.from_defaults(
    fn=query_engine.query,
    name="MathTutor",
    description="Use this tool to answer math questions from the internal knowledge base."
)


def search(query: str) -> str:
    req = DDGS()
    results = req.text(query, max_results=5)
    return "\n".join([r['body'] for r in results])


search_tool = FunctionTool.from_defaults(
    fn=search,
    name="search",
    description="Fallback web search tool for math queries."
)

# ========== AGENT ==========
agent = ReActAgent.from_tools(
    tools=[math_tool, search_tool],
    llm=llm_with_guardrails,
    system_prompt=react_template,
    verbose=True
)


# ========== LLM CLASSIFIER ==========
def is_math_query_llm(query: str) -> bool:
    try:
        prompt = f"""
You are an assistant that determines if a question is related to mathematics.
Question: "{query}"
Answer ONLY "YES" or "NO".
"""
        response = llm.invoke(prompt).content.strip().upper()
        return response == "YES"
    except:
        return False


# ========== MAIN FUNCTION ==========
def math_precheck_llm_agent(query: str) -> str:
    if not is_math_query_llm(query):
        return "I don't know. This question is not related to mathematics."
    return str(agent.query(query))
