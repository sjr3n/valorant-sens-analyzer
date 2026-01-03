# Valorant Sensitivity Analyzer

An AI-powered tool to analyze Valorant gameplay VODs and provide objective sensitivity recommendations based on your actual performance.

## Project Structure

- `frontend/` - Web interface for uploading VODs and viewing results
- `backend/` - Flask API server handling requests and file management
- `logic/` - Core video processing and analysis algorithms

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`

## Running the Application

**Backend:**
```bash
python backend/app.py
```

**Frontend:**
```bash
cd frontend
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Tech Stack
- Python, OpenCV, NumPy
- Flask (REST API)
- Vanilla JavaScript (frontend)
- SQLite (database)