# Resume Match & Ranking — Streamlit

A recruiter-focused Streamlit application for:
- PDF resume text extraction
- Essential/preferred skill matching
- TF-IDF job-description similarity
- Combined candidate scoring
- Optional OpenAI recruiter analysis
- CSV export
- Session-based login
- Streamlit Community Cloud deployment

## Local setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

using `secrets.example.toml` as the template.

Run:

```bash
streamlit run app.py
```

## Important security note

Never commit `.streamlit/secrets.toml`, API keys, passwords, candidate resumes, or other confidential information.

## Scanned PDFs

The current version extracts embedded PDF text. Image-only/scanned resumes are reported as "No Text". OCR can be added as a separate deployment layer if required.

## Scoring

If skills are supplied:

- 70% skill-match score
- 30% JD/resume TF-IDF similarity

If no skills are supplied:

- 100% JD/resume similarity

The score is a screening aid and should not be treated as an automated hiring decision.
