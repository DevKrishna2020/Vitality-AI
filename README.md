Vitality AI 🩺 — Your Personal Health OS

Vitality AI is an intelligent, multimodal health assistant built with Streamlit, RAG (Vector DB Search), Tavily Web Search, and multiple LLM providers.
It helps users understand symptoms, analyze health reports, calculate macros, plan workouts, and chat naturally using a beautiful, streamlined UI.

🚀 Key Features
🔥 1. Multi-Model Support

Switch between AI providers instantly:

OpenAI GPT Models

Google Gemini

Groq Llama / Mixtral

Llama (via Groq API)
Model switching is seamless — your chat history is preserved and injected into the next model.

🎨 2. Elegant Light/Dark Theme Toggle

A smooth, modern UI with:

Custom CSS

Dark & light theme modes

Instant theme switching without UI flicker

📂 3. Health Report Uploads (PDF)

Upload lab reports or health PDFs:

Text-layer extraction

Automatic ingestion into RAG pipeline

Context-aware answers referencing uploaded data

🧠 4. RAG Pipeline (Vector DB Search)

Embeds your uploaded PDFs

Retrieves relevant segments

Combines with LLM response for accurate answers

🌐 5. Tavily Web Search Tool

For up-to-date medical references:

Keyword-based search

Summarized results from trusted sources

💬 6. Smart AI Health Chatbot

Supports:

Symptom triage

Health questions

Explanations of lab values

Safety-first responses

🏋️‍♂️ 7. Fitness Toolkit

Macro calculator

BMI calculator

Personalized workout planning

🔁 8. Multi-Model Switching

Switch providers mid-conversation:

Chat history is preserved

Fed back into the selected LLM

No conversation resets

🖼 9. Streamlined UI Design

Bubble-style chat

Smooth placeholder animations ("Vitality is thinking...")

Consistent spacing, shadows, and rounded UI

Centered chat layout

Sidebar with utilities (history, uploads, API keys)


🛠️ Tech Stack

Component	       Tech
Frontend	       Streamlit + Custom CSS
LLMs	           OpenAI, Google Gemini, Groq Llama
RAG	               FAISS / Local Vector DB
Web Search	       Tavily Search API
Voice Input (optional)	Browser speech-to-text
File Parsing	   PyPDF
Backend Helpers	   Python, Streamlit Session State