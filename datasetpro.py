import numpy as np
import pandas as pd
import faiss
import time
import re
import google.generativeai as genai # Import the Google Generative AI client
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---
# Replace with your actual API key
# IMPORTANT: It's HIGHLY recommended to load API keys from environment variables
# or a secure configuration system, not hardcode them directly in the script.
# For example: import os; GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the app.")

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash") # Using gemini-1.5-flash as a safe default
# --- End Configuration ---

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# --- Data Loading and Preprocessing (same as before) ---
df = pd.read_csv(r'C:\Users\ravik\Faqllm\Kaggle related questions on Qoura - Questions.csv')
df.dropna(subset=['Questions', 'Answered', 'Link'], inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['ProcessedQuestions'] = df['Questions'].apply(preprocess_text)
faq_questions_processed = df['ProcessedQuestions'].tolist()

# --- Embedding Generation using Gemini ---
print(f"Generating embeddings for {len(faq_questions_processed)} FAQs using {GEMINI_EMBEDDING_MODEL}...")
faq_embeddings_list = []
batch_size = 100 # Adjust batch size if needed, Gemini also has limits

for i in range(0, len(faq_questions_processed), batch_size):
    batch = faq_questions_processed[i:i + batch_size]
    try:
        # For retrieval, it's good practice to specify task_type
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=batch,
            task_type="RETRIEVAL_DOCUMENT" # Optimize embeddings for documents in retrieval
        )
        # Extract embeddings from the response
        if 'embedding' in response:
            faq_embeddings_list.extend(response['embedding'])
        else:
            print(f"Warning: 'embedding' key missing in response for batch {i}-{i+len(batch)}")
    except Exception as e:
        print(f"Error generating embeddings for batch {i}-{i+len(batch)}: {e}")
        time.sleep(1)

# Essential check after embedding generation for FAQs
if not faq_embeddings_list:
    print("FATAL ERROR: No embeddings were generated for FAQs. Exiting.")
    exit()

faq_embeddings = np.array(faq_embeddings_list).astype('float32')
faq_embeddings = np.ascontiguousarray(faq_embeddings) # Ensure C-contiguous
faiss.normalize_L2(faq_embeddings) # Normalize the main FAQ embeddings

# --- FAISS Indexing (same as before) ---
embedding_dim = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(faq_embeddings)
print("Embeddings generated and FAISS index built.")

# --- Matching Function (using Gemini Embeddings) ---
def find_best_match_gemini_embeddings_only(user_query, k=1):
    processed_user_query = preprocess_text(user_query)
    query_embedding_np = None # Initialize to None

    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=[processed_user_query],
            task_type="RETRIEVAL_QUERY" # Optimize embeddings for queries in retrieval
        )
        # Ensure 'embedding' key exists and the embedding is not empty
        if 'embedding' in response and response['embedding'] and isinstance(response['embedding'][0], list):
            # Convert the list of floats to a NumPy array, cast to float32
            temp_embedding = np.array([response['embedding'][0]]).astype('float32')
            # Ensure it's C-contiguous for FAISS
            query_embedding_np = np.ascontiguousarray(temp_embedding)
            # Normalize the query embedding (crucial for L2 distance with normalized vectors)
            faiss.normalize_L2(query_embedding_np)
        else:
            print(f"Warning: Invalid or empty embedding response for query: '{user_query}'")
            return None, None, None, 0.0 # Return None for all values on warning
    except Exception as e:
        print(f"Error generating query embedding for '{user_query}': {e}")
        return None, None, None, 0.0 # Return None for all values on error

    # --- CRUCIAL FIX: Check if query_embedding_np is None before proceeding to FAISS search ---
    if query_embedding_np is None:
        return None, None, None, 0.0 # Return None for all values if embedding failed

    D, I = index.search(query_embedding_np, k=k) # Use the correctly processed numpy array

    if k == 1:
        best_match_index = I[0][0]
        # Same confidence score calculation for L2 normalized embeddings
        # D[0][0] is the squared L2 distance. 1 - (distance / 2) is common for similarity from L2.
        confidence_score = 1 - (D[0][0] / 2)

        matched_question = df['Questions'].iloc[best_match_index]
        matched_answer = df['Answered'].iloc[best_match_index]
        matched_link = df['Link'].iloc[best_match_index] if 'Link' in df.columns else None
        return matched_question, matched_answer, matched_link, confidence_score
    else:
        results = []
        # Ensure that D and I contain enough results for k
        num_results = min(k, len(I[0]))
        for i in range(num_results):
            matched_idx = I[0][i]
            confidence_score = 1 - (D[0][i] / 2)
            results.append({
                'question': df['Questions'].iloc[matched_idx],
                'answer': df['Answered'].iloc[matched_idx],
                'link': df['Link'].iloc[matched_idx] if 'Link' in df.columns else None,
                'confidence': confidence_score
            })
        return results

# --- RAG-based Generation Function (using Gemini Chat Model) ---
def generate_answer_with_rag_gemini(user_query, num_retrieved_faqs=3):
    # Use the Gemini embedding function to retrieve top FAQs
    retrieved_faqs_result = find_best_match_gemini_embeddings_only(user_query, k=num_retrieved_faqs)

    # Handle the case where find_best_match_gemini_embeddings_only returns None for k=1
    if num_retrieved_faqs == 1:
        # Check if the first element (question) is None, which indicates an error/no match
        if retrieved_faqs_result[0] is None:
            return "No relevant FAQs could be retrieved via embedding search or an error occurred during query embedding."
        # If it's not None, convert the single result into a list of dictionaries for consistent processing
        retrieved_faqs = [{
            'question': retrieved_faqs_result[0],
            'answer': retrieved_faqs_result[1],
            'link': retrieved_faqs_result[2],
            'confidence': retrieved_faqs_result[3]
        }]
    else:
        # For k > 1, it should already return a list of dictionaries.
        # Check if the list is empty or if it somehow contains None
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

    # Initialize Gemini GenerativeModel for chat
    model = genai.GenerativeModel(GEMINI_CHAT_MODEL)

    # Construct the prompt for the LLM
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
        # Use generate_content for a single turn, text-based interaction
        response = model.generate_content(
            prompt_message,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1000, # Limit the length of the generated answer
                temperature=0.1,      # Lower temperature for more factual answers
            ),
            # safety_settings=... # You might want to add safety settings
        )
        llm_answer = response.text
        return llm_answer
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        return "An error occurred while generating the answer."

# --- Main Execution Loop (with Gemini options) ---
if __name__ == "__main__":
    print("\n--- FAQ Matching System ---")
    print("Choose a mode:")
    print("1. Embedding-based Matching (direct FAQ lookup using Gemini Embeddings)")
    print("2. Generation-based Matching (Hybrid RAG with Gemini LLM)")

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
                # Embedding-based Matching with Gemini embeddings
                # Check the first element of the returned tuple
                question, answer, link, score = find_best_match_gemini_embeddings_only(user_question, k=1)
                if question is not None: # Check if question is not None to signify a valid result
                    print(f"\nMatched Question: {question}")
                    print(f"Answer: {answer}")
                    if link:
                        print(f"Link: quora.com/{link}")
                    print(f"Confidence Score: {score:.4f}\n")
                else:
                    print("Could not find a match or an error occurred during embedding the query.\n")
            elif mode_choice == '2':
                # Generation-based Matching (Hybrid RAG with Gemini LLM)
                print("\n--- Generating answer with LLM (based on top retrieved FAQs) ---")
                generated_answer = generate_answer_with_rag_gemini(user_question, num_retrieved_faqs=3)
                print(f"\nUser Question: {user_question}")
                print(f"Generated Answer: {generated_answer}")
                print("\n")