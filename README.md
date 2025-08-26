# MathTutor
Math Tutor AI Agent
An AI-powered mathematics tutor that combines RAG (Retrieval-Augmented Generation) with real-time web search to provide accurate, step-by-step explanations for math problems.This system is designed with educational guardrails, ensuring safe, factual, and age-appropriate responses. It uses a vector database (Qdrant) for knowledge retrieval and Gemini (Google Generative AI) as the reasoning engine.


**Features:**


->Math-only Focus → Accepts and answers only mathematics-related queries (algebra, calculus, geometry, statistics, etc.).


-> RAG-Powered Learning → Retrieves answers from a curated math knowledge base stored in Qdrant Vector DB.


->Smart Web Search Fallback → Uses DuckDuckGo search when the knowledge base lacks an answer.


->Safety & Guardrails → Filters non-math queries, avoids hallucinations, and ensures respectful, educational explanations.


->Step-by-Step Explanations → Always restates the question clearly and explains the solution process step by step.


->Fast & Scalable → Uses FastEmbed embeddings + Gemini-2.0-Flash for lightning-fast performance.


**Tech Stack:**


->LLM: Google Gemini(via langchain-google-genai)


->Vector DB: Qdrant


->Embeddings: FastEmbed


->Agent Framework: LlamaIndex,ReAct Agent


->Search API: DuckDuckGo Search


->Environment Management: python-dotenv


**⚙️ How It Works:**


1)Query Pre-Check → A classifier determines if the question is math-related.


2)RAG Retrieval → Searches Qdrant for relevant problems, formulas, and explanations.


3)LLM Reasoning → Gemini processes the retrieved context with guardrails.


4)Fallback Search → If no answer is found, triggers a DuckDuckGo web search.


5)Safe Response → Provides a clear, factual, and step-by-step explanation.
