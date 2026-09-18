# ResumeMatch AI

A polished NLP-based Resume & Job Matching web application built with **FastAPI, scikit-learn and modern HTML/CSS/JavaScript**, designed for local use and Vercel deployment.

## Features
- PDF, DOCX and TXT resume upload
- Job-description matching with TF-IDF + cosine similarity
- Matched and missing skill extraction
- Skill categories and coverage analysis
- ATS-style readiness and keyword coverage indicators
- Smart resume improvement suggestions
- AI-style career assistant based on the latest analysis
- Resume summary draft
- Career learning roadmap from detected gaps
- Downloadable analysis report
- Responsive professional dashboard UI

## Local run
Use Python 3.11.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn api.index:app --reload
```

Open `http://127.0.0.1:8000/`.

API docs: `http://127.0.0.1:8000/docs`

## Vercel
The project uses `api/index.py` as the Python serverless entry point and `vercel.json` for routing. Push the project to GitHub without the `venv` folder, then import the repository into Vercel.

> Scores are analytical indicators, not hiring decisions. Users should only add skills or experience they genuinely have.
