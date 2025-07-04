# flask_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import faiss
import time
import re
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# --- Configuration (load from environment variables) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the app.")

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash") # Using gemini-1.5-flash as a safe default
# --- End Configuration ---

genai.configure(api_key=GEMINI_API_KEY)

# --- Global variables for data and index (initialized to None, populated on startup) ---
df = None
faiss_index = None
faq_embeddings = None

# --- Data Loading and Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define your initialization function (without the decorator)
def initialize_model():
    global df, faiss_index, faq_embeddings # Declare globals to be assigned
    data_file_path = r'C:\Users\ravik\Faqllm\Kaggle related questions on Qoura - Questions.csv'

    # Load and preprocess data
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: CSV file not found at {data_file_path}. Please ensure the path is correct.")
        exit(1) # Exit with an error code if the data file isn't found
    df.dropna(subset=['Questions', 'Answered', 'Link'], inplace=True)
    df['ProcessedQuestions'] = df['Questions'].apply(preprocess_text)
    faq_questions_processed = df['ProcessedQuestions'].tolist()

    # Generate embeddings and build FAISS index
    print(f"Generating embeddings for {len(faq_questions_processed)} FAQs using {GEMINI_EMBEDDING_MODEL}...")
    faq_embeddings_list = []
    batch_size = 100

    for i in range(0, len(faq_questions_processed), batch_size):
        batch = faq_questions_processed[i:i + batch_size]
        try:
            response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            if 'embedding' in response:
                faq_embeddings_list.extend(response['embedding'])
            else:
                print(f"Warning: 'embedding' key missing in response for batch {i}-{i+len(batch)}")
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i+len(batch)}: {e}")
            time.sleep(1)

    if not faq_embeddings_list:
        print("FATAL ERROR: No embeddings were generated for FAQs. Check API key, model name, and network.")
        exit(1) # Exit if we can't build the core component

    faq_embeddings = np.array(faq_embeddings_list).astype('float32')
    faq_embeddings = np.ascontiguousarray(faq_embeddings)
    faiss.normalize_L2(faq_embeddings)

    embedding_dim = faq_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(faq_embeddings)
    print("Embeddings generated and FAISS index built.")


# --- Matching Function (using Gemini Embeddings) ---
def find_best_match_gemini_embeddings_only(user_query, k=1, index=None, df=None):
    if index is None or df is None:
        print("Error: FAISS index or DataFrame not provided to find_best_match_gemini_embeddings_only.")
        return None, None, None, 0.0

    processed_user_query = preprocess_text(user_query)
    query_embedding_np = None

    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=[processed_user_query],
            task_type="RETRIEVAL_QUERY"
        )
        if 'embedding' in response and response['embedding'] and isinstance(response['embedding'][0], list):
            temp_embedding = np.array([response['embedding'][0]]).astype('float32')
            query_embedding_np = np.ascontiguousarray(temp_embedding)
            faiss.normalize_L2(query_embedding_np)
        else:
            print(f"Warning: Invalid or empty embedding response for query: '{user_query}'")
            return None, None, None, 0.0
    except Exception as e:
        print(f"Error generating query embedding for '{user_query}': {e}")
        return None, None, None, 0.0

    if query_embedding_np is None:
        return None, None, None, 0.0

    D, I = index.search(query_embedding_np, k=k)

    if k == 1:
        best_match_index = I[0][0]
        confidence_score = float(1 - (D[0][0] / 2)) # Ensure it's a standard float

        matched_question = str(df['Questions'].iloc[best_match_index])
        matched_answer = str(df['Answered'].iloc[best_match_index])

        # --- MODIFICATION START ---
        matched_link_raw = df['Link'].iloc[best_match_index]
        matched_link = None # Initialize as None
        if pd.notna(matched_link_raw) and isinstance(matched_link_raw, str):
            if matched_link_raw.startswith('/'):
                matched_link = "https://www.quora.com" + matched_link_raw
            elif not matched_link_raw.startswith('http://') and not matched_link_raw.startswith('https://'):
                # If it's not a relative path and not already a full URL, treat it as a Quora path
                # This handles cases like "What-is-Kaggle" without a leading slash
                matched_link = "https://www.quora.com/" + matched_link_raw.lstrip('/')
            else:
                matched_link = matched_link_raw # It's already a full URL
        # --- MODIFICATION END ---

        return matched_question, matched_answer, matched_link, confidence_score
    else:
        results = []
        num_results = min(k, len(I[0]))
        for i in range(num_results):
            matched_idx = I[0][i]
            confidence_score = float(1 - (D[0][i] / 2)) # Ensure it's a standard float

            matched_link_raw = df['Link'].iloc[matched_idx]
            full_link = None # Initialize as None
            if pd.notna(matched_link_raw) and isinstance(matched_link_raw, str):
                if matched_link_raw.startswith('/'):
                    full_link = "https://www.quora.com" + matched_link_raw
                elif not matched_link_raw.startswith('http://') and not matched_link_raw.startswith('https://'):
                    full_link = "https://www.quora.com/" + matched_link_raw.lstrip('/')
                else:
                    full_link = matched_link_raw

            results.append({
                'question': str(df['Questions'].iloc[matched_idx]),
                'answer': str(df['Answered'].iloc[matched_idx]),
                'link': full_link, # Use the full_link here
                'confidence': float(confidence_score)
            })
        return results

# --- RAG-based Generation Function (using Gemini Chat Model) ---
def generate_answer_with_rag_gemini(user_query, num_retrieved_faqs=3, index=None, df=None):
    if index is None or df is None:
        print("Error: FAISS index or DataFrame not provided to generate_answer_with_rag_gemini.")
        return "An internal error occurred."

    retrieved_faqs_result = find_best_match_gemini_embeddings_only(user_query, k=num_retrieved_faqs, index=index, df=df)

    if num_retrieved_faqs == 1:
        if retrieved_faqs_result[0] is None:
            return "No relevant FAQs could be retrieved via embedding search or an error occurred during query embedding."
        retrieved_faqs = [{
            'question': retrieved_faqs_result[0],
            'answer': retrieved_faqs_result[1],
            'link': retrieved_faqs_result[2],
            'confidence': retrieved_faqs_result[3]
        }]
    else:
        if not retrieved_faqs_result or any(item is None for item in retrieved_faqs_result):
             return "No relevant FAQs could be retrieved via embedding search or an error occurred during query embedding."
        retrieved_faqs = retrieved_faqs_result


    context_str = ""
    for i, faq in enumerate(retrieved_faqs):
        context_str += f"FAQ {i+1} (Confidence: {faq['confidence']:.4f}):\n"
        context_str += f"Question: {faq['question']}\n"
        context_str += f"Answer: {faq['answer']}\n"
        if faq['link']:
            context_str += f"Link: {faq['link']}\n"
        context_str += "---\n"

    model = genai.GenerativeModel(GEMINI_CHAT_MODEL)

    prompt_message = f"""
    You are a helpful assistant for a Kaggle FAQ system. Use the provided FAQs to answer the user's question concisely and accurately.
    If the provided FAQs do not contain enough information to answer the question, state that you cannot answer based on the given information.

    User's Question: {user_query}

    Relevant FAQs:
    {context_str}

    Based on the provided FAQs, please answer the user's question. If a link is available in the relevant FAQ, include it at the end of your answer. If no relevant FAQ is found, state that you cannot provide an answer.
    Also if you are able to answer the question, provide a link to the relevant FAQ if available.
    """

    try:
        response = model.generate_content(
            prompt_message,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.1,
            ),
        )
        llm_answer = response.text
        return llm_answer
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        return "An error occurred while generating the answer."

# --- API Endpoints ---
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    mode = request.json.get('mode', 'rag')
    k = request.json.get('k', 3)

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Ensure global variables are populated (should be after initialize_model() runs)
    if faiss_index is None or df is None:
        print("Backend not fully initialized. FAISS index or DataFrame is None.")
        return jsonify({"error": "Backend not fully initialized. Please try again later."}), 500

    if mode == 'rag':
        answer = generate_answer_with_rag_gemini(user_question, num_retrieved_faqs=k, index=faiss_index, df=df)
        return jsonify({"answer": answer})
    elif mode == 'embedding':
        question, answer, link, score = find_best_match_gemini_embeddings_only(user_question, k=1, index=faiss_index, df=df)
        if question:
            return jsonify({
                "matched_question": question,
                "answer": float(answer),
                "link": link,
                "confidence_score": float(score)
            })
        else:
            return jsonify({"answer": "Could not find a direct match for your question."})
    else:
        return jsonify({"error": "Invalid mode specified"}), 400

if __name__ == '__main__':
    # --- FIX: Call initialization directly here ---
    initialize_model()
    app.run(debug=True, port=5000)