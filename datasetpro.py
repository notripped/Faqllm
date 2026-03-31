import numpy as np
import pandas as pd
import faiss
import re
import os
import ollama
from sentence_transformers import SentenceTransformer

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL_NAME = "llama3.2"

# --- Model & Data Initialization ---
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

data_file_path = os.path.join(os.path.dirname(__file__), 'Kaggle related questions on Qoura - Questions.csv')
df = pd.read_csv(data_file_path)
df.dropna(subset=['Questions', 'Answered', 'Link'], inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['ProcessedQuestions'] = df['Questions'].apply(preprocess_text)
faq_questions_processed = df['ProcessedQuestions'].tolist()

# --- Embedding Generation ---
print(f"Generating embeddings for {len(faq_questions_processed)} FAQs using {EMBEDDING_MODEL_NAME}...")
faq_embeddings = embedding_model.encode(faq_questions_processed, batch_size=64, show_progress_bar=True)
faq_embeddings = np.array(faq_embeddings).astype('float32')
faq_embeddings = np.ascontiguousarray(faq_embeddings)
faiss.normalize_L2(faq_embeddings)

# --- FAISS Indexing ---
embedding_dim = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(faq_embeddings)
print("Embeddings generated and FAISS index built.")

# --- Matching Function ---
def find_best_match(user_query, k=1):
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

    if k == 1:
        idx = I[0][0]
        confidence = float(1 - (D[0][0] / 2))
        return (
            str(df['Questions'].iloc[idx]),
            str(df['Answered'].iloc[idx]),
            build_link(df['Link'].iloc[idx]),
            confidence
        )
    else:
        results = []
        for i in range(min(k, len(I[0]))):
            idx = I[0][i]
            results.append({
                'question': str(df['Questions'].iloc[idx]),
                'answer': str(df['Answered'].iloc[idx]),
                'link': build_link(df['Link'].iloc[idx]),
                'confidence': float(1 - (D[0][i] / 2))
            })
        return results

# --- RAG Generation Function ---
def generate_answer_with_rag(user_query, num_retrieved_faqs=3):
    retrieved = find_best_match(user_query, k=num_retrieved_faqs)

    if num_retrieved_faqs == 1:
        if retrieved[0] is None:
            return "No relevant FAQs found."
        faqs = [{'question': retrieved[0], 'answer': retrieved[1], 'link': retrieved[2], 'confidence': retrieved[3]}]
    else:
        if not retrieved:
            return "No relevant FAQs found."
        faqs = retrieved

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
        return faqs[0]['answer'] if faqs else "An error occurred while generating the answer."

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("\n--- FAQ Matching System ---")
    print("Choose a mode:")
    print("1. Embedding-based Matching (direct FAQ lookup)")
    print("2. Generation-based Matching (RAG with Ollama LLM)")

    while True:
        mode_choice = input("Enter mode (1 or 2, or 'exit' to quit): ")
        if mode_choice.lower() == 'exit':
            break
        if mode_choice not in ['1', '2']:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        while True:
            user_question = input("Ask a question (or type 'back' to change mode, 'exit' to quit): ")
            if user_question.lower() == 'exit':
                exit()
            if user_question.lower() == 'back':
                break

            if mode_choice == '1':
                question, answer, link, score = find_best_match(user_question, k=1)
                if question:
                    print(f"\nMatched Question: {question}")
                    print(f"Answer: {answer}")
                    if link:
                        print(f"Link: {link}")
                    print(f"Confidence Score: {score:.4f}\n")
                else:
                    print("Could not find a match.\n")

            elif mode_choice == '2':
                print("\n--- Generating answer with Ollama LLM ---")
                generated_answer = generate_answer_with_rag(user_question, num_retrieved_faqs=3)
                print(f"\nUser Question: {user_question}")
                print(f"Generated Answer: {generated_answer}\n")
