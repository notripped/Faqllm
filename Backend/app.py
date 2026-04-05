# flask_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import faiss
import os
import re
import ollama
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL_NAME = "llama3.2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- Global variables ---
df = None
faiss_index = None
faq_embeddings = None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def initialize_model():
    global df, faiss_index, faq_embeddings

    print("DEBUG: initialize_model() started.")

    data_file_path = os.path.join(os.path.dirname(__file__), '..', 'Kaggle related questions on Qoura - Questions.csv')
    print(f"DEBUG: Attempting to load data from: {data_file_path}")

    try:
        df = pd.read_csv(data_file_path)
        print("DEBUG: CSV data loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: CSV file not found at {data_file_path}.")
        exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Could not load CSV data: {e}")
        exit(1)

    print("DEBUG: Data loaded. Starting preprocessing and dropping NaNs.")
    df.dropna(subset=['Questions', 'Answered', 'Link'], inplace=True)
    df['ProcessedQuestions'] = df['Questions'].apply(preprocess_text)
    faq_questions_processed = df['ProcessedQuestions'].tolist()
    print(f"DEBUG: Preprocessing complete. Number of processed FAQs: {len(faq_questions_processed)}")

    print(f"DEBUG: Generating embeddings using {EMBEDDING_MODEL_NAME}...")
    faq_embeddings = embedding_model.encode(faq_questions_processed, batch_size=64, show_progress_bar=True)
    faq_embeddings = np.array(faq_embeddings).astype('float32')
    faq_embeddings = np.ascontiguousarray(faq_embeddings)
    faiss.normalize_L2(faq_embeddings)

    embedding_dim = faq_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(faq_embeddings)
    print("DEBUG: Embeddings generated and FAISS index built successfully.")

def find_best_match(user_query, k=1, index=None, dataframe=None):
    if index is None or dataframe is None:
        return []

    processed_user_query = preprocess_text(user_query)
    query_embedding = embedding_model.encode([processed_user_query])
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding = np.ascontiguousarray(query_embedding)
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k=k)

    def build_link(raw):
        if pd.notna(raw) and isinstance(raw, str):
            if raw.startswith('/'):
                return "https://www.quora.com" + raw
            elif not raw.startswith('http://') and not raw.startswith('https://'):
                return "https://www.quora.com/" + raw.lstrip('/')
            return raw
        return None

    results = []
    for i in range(min(k, len(I[0]))):
        idx = I[0][i]
        results.append({
            'question': str(dataframe['Questions'].iloc[idx]),
            'answer': str(dataframe['Answered'].iloc[idx]),
            'link': build_link(dataframe['Link'].iloc[idx]),
            'confidence': float(1 - (D[0][i] / 2))
        })
    return results

def generate_answer_with_rag(user_query, num_retrieved_faqs=3, index=None, dataframe=None):
    if index is None or dataframe is None:
        return "An internal error occurred."

    faqs = find_best_match(user_query, k=num_retrieved_faqs, index=index, dataframe=dataframe)

    if not faqs:
        return "No relevant FAQs found."

    context_str = ""
    for i, faq in enumerate(faqs):
        context_str += f"FAQ {i+1} (Confidence: {faq['confidence']:.4f}):\n"
        context_str += f"Question: {faq['question']}\n"
        context_str += f"Answer: {faq['answer']}\n"
        if faq['link']:
            context_str += f"Link: {faq['link']}\n"
        context_str += "---\n"

    prompt = f"""You are a Kaggle FAQ assistant. Answer the user's question directly using the FAQ content below.

Rules:
- Give a direct, confident answer using the FAQ answer text
- Do NOT say "based on the provided FAQs" or "I couldn't find"
- Do NOT hedge or qualify unless the FAQ itself is vague
- Keep it to 2-3 sentences max
- If a link is available, end with: "Learn more: <link>"

User's Question: {user_query}

FAQs:
{context_str}

Answer:"""

    try:
        response = ollama.chat(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 300, "temperature": 0.1}
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        fallback = faqs[0]['answer'] if faqs else "No answer available."
        return f"[Ollama unavailable — showing direct FAQ match]\n\n{fallback}"

def get_brief_explanation(user_query):
    try:
        response = ollama.chat(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": f"In 1 sentence, directly answer: {user_query}"}],
            options={"num_predict": 100, "temperature": 0.2}
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "[Ollama unavailable]"

# --- API Endpoints ---
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    mode = request.json.get('mode', 'rag')
    k = request.json.get('k', 3)
    try:
        k = int(k)
        if k < 1 or k > 10:
            return jsonify({"error": "k must be between 1 and 10"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "k must be an integer"}), 400

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    if faiss_index is None or df is None:
        return jsonify({"error": "Backend not fully initialized. Please try again later."}), 500

    if mode == 'rag':
        answer = generate_answer_with_rag(user_question, num_retrieved_faqs=k, index=faiss_index, dataframe=df)
        explanation = get_brief_explanation(user_question)
        return jsonify({"answer": answer, "explanation": explanation})

    elif mode == 'embedding':
        results = find_best_match(user_question, k=1, index=faiss_index, dataframe=df)
        if results:
            match = results[0]
            explanation = get_brief_explanation(user_question)
            return jsonify({
                "matched_question": match['question'],
                "answer": match['answer'],
                "link": match['link'],
                "confidence_score": match['confidence'],
                "explanation": explanation
            })
        else:
            return jsonify({"answer": "Could not find a direct match for your question."})
    else:
        return jsonify({"error": "Invalid mode specified"}), 400

if __name__ == '__main__':
    initialize_model()
    app.run(debug=False, port=5000)
