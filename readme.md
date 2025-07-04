# Kaggle FAQ Chatbot

A Retrieval-Augmented Generation (RAG) system designed to answer user questions about Kaggle, leveraging Google Gemini's embedding and generative models combined with FAISS for efficient similarity search. The application features a Flask backend API and a simple HTML/CSS/JavaScript frontend for interaction.

## Features

- **Intelligent FAQ Retrieval**: Uses Gemini embeddings and FAISS to find the most relevant Kaggle FAQs based on user queries
- **Hybrid RAG Approach**:
  - **Embedding-based Matching**: Directly retrieves and displays the closest matching FAQ, its answer, and a link
  - **Generation-based Matching**: Retrieves relevant FAQs as context and uses a Gemini chat model to generate a concise, human-like answer
- **Clean Frontend**: A simple, aesthetic web interface for asking questions and viewing responses
- **Scalable Backend**: Flask API separates the core logic from the frontend, allowing for flexible deployment

## Technical Stack

### Backend
- **Python**: Core programming language
- **Flask**: Lightweight web framework for building the API
- **Flask-CORS**: Handles Cross-Origin Resource Sharing for frontend communication
- **Google Generative AI (Gemini API)**:
  - `models/text-embedding-004`: For generating embeddings of questions and answers
  - `gemini-pro`: For generating natural language answers (the LLM)
- **FAISS**: (Facebook AI Similarity Search) For efficient similarity search over the FAQ embeddings
- **Pandas & NumPy**: For data handling and numerical operations

### Frontend
- **HTML5**: Structure of the web page
- **CSS3**: Styling for an aesthetic user interface
- **JavaScript**: Handles user interaction and communication with the Flask backend

## Important Note on Virtual Environment

⚠️ **Your virtual environment (`faq-env/`) contains large binary files** (like `torch_cpu.dll` and `dnnl.lib` from deep learning libraries) that exceed GitHub's 100MB file size limit. This folder must not be committed to Git. It is correctly listed in `.gitignore`. If you previously attempted to push it, you need to clean your Git history using `git filter-repo` to remove these large files from past commits.

## Setup Instructions (Local Development)

### Prerequisites
- **Python 3.8+**: Download and install from [python.org](https://python.org)
- **Git**: Download and install from [git-scm.com](https://git-scm.com)
- **Git LFS**: Download and install from [git-lfs.github.com](https://git-lfs.github.com). Run `git lfs install` in your terminal after installation
- **Google Gemini API Key**: Obtain one from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Faqllm.git
   cd Faqllm
   ```
   *(Replace `your-username` with your actual GitHub username and `Faqllm` with your repository name if different)*

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv faq-env
   
   # On Windows:
   .\faq-env\Scripts\activate
   
   # On macOS/Linux:
   source faq-env/bin/activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place Your Dataset**:
   Ensure your dataset file `Kaggle related questions on Qoura - Questions.csv` is located in the root of your `Faqllm` project directory, or update the `data_file_path` variable in `Backend/flask_app.py` to its correct location.

### Environment Variables

The Flask backend requires your Google Gemini API key to be set as an environment variable.

**On Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
set FLASK_APP=Backend/flask_app.py
set FLASK_ENV=development
```

**On Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
$env:FLASK_APP="Backend/flask_app.py"
$env:FLASK_ENV="development"
```

**On macOS/Linux (Bash/Zsh):**
```bash
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
export FLASK_APP=Backend/flask_app.py
export FLASK_ENV=development
```

*Replace `"YOUR_GEMINI_API_KEY"` with the actual API key you obtained.*

### Running the Application

#### 1. Start the Backend
1. Open your terminal or VS Code's integrated terminal
2. Navigate to the `Faqllm` root directory
3. Ensure your virtual environment is activated and environment variables are set
4. Run the Flask application:
   ```bash
   flask run --port 5000 --debug
   ```

The backend will start and begin loading embeddings and building the FAISS index. This might take a few minutes depending on your dataset size and API response times. You should see output indicating that the server is running on `http://127.0.0.1:5000`. Keep this terminal open.

#### 2. Start the Frontend
1. Open a new terminal window (separate from the one running the Flask backend)
2. Navigate to the `Faqllm` root directory
3. Serve the static frontend files using Python's built-in HTTP server:
   ```bash
   python -m http.server 8000
   ```

This will serve your `index.html` and `styles.css` from `http://127.0.0.1:8000`. Keep this terminal open.

4. Open your web browser and go to `http://localhost:8000/Frontend/index.html`

You can now interact with your chatbot! Check both terminal windows and your browser's developer console (F12) for any errors.

## Deployment Instructions

### Option 1: Streamlit Cloud (Easiest for quick deployment)

While this project is structured with a separate backend/frontend, for simplicity and quick sharing, you could adapt the core RAG logic into a single Streamlit application.

1. **Create a `streamlit_app.py`**: Consolidate all your Python logic (data loading, embedding, FAISS, Gemini calls) and Streamlit UI code into a single file
2. **Update `requirements.txt`**: Add `streamlit` to your requirements.txt
3. **Use Streamlit Secrets**: Create a `.streamlit/secrets.toml` file:
   ```toml
   GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
   ```
4. **Deploy on Streamlit Community Cloud**: Go to [share.streamlit.io](https://share.streamlit.io), log in, and connect your GitHub repository

### Option 2: Flask Backend + Static Frontend (More Control & Scalability)

#### Backend Deployment (Flask API)

**Recommended: Google Cloud Run** (Serverless, scales automatically, pay-per-use)

1. **Containerize your Flask App**: Create a `Dockerfile` in your root directory:
   ```dockerfile
   FROM python:3.10-slim-buster

   WORKDIR /app

   # Copy requirements.txt and install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy the rest of your application code
   COPY . .

   # Set environment variables for the container
   ENV GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
   ENV FLASK_APP="Backend/flask_app.py"
   ENV FLASK_ENV="production"

   # Expose the port Flask runs on
   EXPOSE 5000

   # Command to run the application using Gunicorn
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "Backend.flask_app:app"]
   ```

2. **Build and Push Docker Image**:
   - Install Docker Desktop
   - Install Google Cloud CLI (gcloud)
   - Authenticate and deploy to Cloud Run

#### Frontend Deployment

Deploy your static HTML/CSS/JS files to:
- **Netlify** (recommended for static sites)
- **Vercel**
- **GitHub Pages**
- **Google Cloud Storage** (with CDN)

## Project Structure

```
Faqllm/
├── Backend/
│   └── flask_app.py
├── Frontend/
│   ├── index.html
│   └── styles.css
├── faq-env/                 # Virtual environment (not in Git)
├── requirements.txt
├── .gitignore
├── Dockerfile
└── Kaggle related questions on Qoura - Questions.csv
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini AI for embedding and generation capabilities
- FAISS for efficient similarity search
- Kaggle community for the FAQ dataset