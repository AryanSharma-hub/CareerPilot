# CareerPilot AI v2

AI-powered Resume Analysis, ATS Scoring, Job Matching, and Career Guidance Platform.

---

## Project Structure

```
careerpilot/
├── backend/
│   ├── app/
│   │   ├── data/career_data.txt
│   │   ├── rag/rag_engine.py
│   │   ├── routes/resume.py
│   │   ├── services/          (all AI service modules)
│   │   ├── utils/llm_client.py
│   │   └── main.py
│   ├── uploads/               (auto-created on first run)
│   ├── .env                   (add your API key here)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

---

## Setup Instructions

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API key to .env
# Open .env and replace: your_openrouter_api_key_here

# Run the backend
uvicorn app.main:app --reload
```

Backend runs at: http://127.0.0.1:8000

---

### 2. Frontend

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Run the frontend
npm run dev
```

Frontend runs at: http://localhost:5173

---

## API Key

Get your OpenRouter API key from: https://openrouter.ai

Add it to `backend/.env`:
```
OPENROUTER_API_KEY=your_actual_key_here
```

## Model

This project uses **google/gemini-flash-1.5** via OpenRouter.
To change the model, edit `backend/app/utils/llm_client.py`:
```python
PRIMARY_MODEL  = "google/gemini-flash-1.5"
FALLBACK_MODEL = "google/gemini-flash-1.5-8b"
```

---

## Important Notes

- Always run uvicorn from inside the `backend/` folder
- Never run it from inside `venv/`
- The `uploads/` folder is created automatically
- `career_memory.json` is created automatically for chat memory
