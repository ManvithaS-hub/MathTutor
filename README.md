# Ganit AI - RAG-Powered Math Assistant

An AI-powered mathematics tutor that combines RAG (Retrieval-Augmented Generation) with real-time web search to provide accurate, step-by-step explanations for math problems.This system is designed with educational guardrails, ensuring safe, factual, and age-appropriate responses. It uses a vector database (Qdrant) for knowledge retrieval and Gemini (Google Generative AI) as the reasoning engine.

---

## 🚀 Features
- **Replicate a Math Professor:** Build an AI agent capable of solving mathematical problems with clear, step-by-step explanations.

- **RAG-Driven Reasoning:** Retrieve solutions from a curated knowledge base using Qdrant Vector DB before relying on external sources.

- **Web Search Fallback:** Perform real-time search (DuckDuckGo) when the knowledge base lacks sufficient context.

- **Simplified Learning:** Present complex mathematical concepts in a simplified and structured manner for students.

- **Guardrails & Safety:**  Ensure the system only processes mathematics-related queries while avoiding hallucinations.

- **Scalable & Fast:** Use FastEmbed embeddings + Gemini LLM for efficient query understanding and reasoning.

- **Agentic Architecture:** Leverage LlamaIndex + ReAct Agent to orchestrate retrieval, reasoning, and safe response generation.

---

## 🛠️ Tech Stack

| Component | Details |
|-----------|---------|
| LLM | Google Gemini(via langchain-google-genai) |
| Vector DB | Qdrant |
| Embeddings | FastEmbed |
| Agent Framework | LlamaIndex,ReAct Agent |
| Search API | DuckDuckGo Search |

---

## ⚙️ How It Works


1)**Query Pre-Check:**  A classifier determines if the question is math-related.


2)**RAG Retrieval:** Searches Qdrant for relevant problems, formulas, and explanations.


3)**LLM Reasoning:** Gemini processes the retrieved context with guardrails.


4)**Fallback Search:** If no answer is found, triggers a DuckDuckGo web search.


5)**Safe Response:** Provides a clear, factual, and step-by-step explanation.

---

## 🌐 Usage
### 1. Clone the Repository
```bash
https://github.com/ManvithaS-hub/MathTutor.git
cd MathTutor
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Add Credentials
Create a .env file in the root folder:
```
QDRANT_URL=your_url
QDRANT_API=your_apiKey
GOOGLE_API_KEY=your_apiKey
```

### 4. Run the main.py file:
```
python main.py
```
### 5. 📂 Project Structure
```Iden_Challenge/
├── main.py        # Main automation script
├── .env               # Environment variables (ignored)
├── venv/              # Python virtual environment (ignored)
├── README.md          # Project documentation
├── test.json          #Project Dataset
├──app.py              #Project frontent using streamlit
└── requirements.txt         # Ignore .env and venv

```
---
## 6. Future Enhancements
### 1) Enhanced Frontend UI
- Develop an interactive web-based interface for students.
- Add visualization support (graphs, equations rendering with LaTeX, step-by-step highlights).
- Mobile-friendly responsive design for wider accessibility.

  
### 2) Human Feedback Loop Integration
- Incorporate Reinforcement Learning from Human Feedback (RLHF) or a simple thumbs-up/down system.
- Allow users (students/teachers) to rate answers, helping the agent continuously improve.
- Store feedback for iterative fine-tuning of the knowledge base and reasoning pipeline.

---


