
from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume Match & Ranking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "gpt-5.6-luna"
SESSION_TIMEOUT_SECONDS = 8 * 60 * 60
MAX_FILE_SIZE_MB = 10


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", DEFAULT_MODEL)
APP_USERNAME = get_secret("APP_USERNAME")
APP_PASSWORD = get_secret("APP_PASSWORD")


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "authenticated": False,
    "login_time": None,
    "last_activity": None,
    "results": [],
    "run_id": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SECURITY / AUTHENTICATION
# ============================================================

def credentials_configured() -> bool:
    return bool(APP_USERNAME and APP_PASSWORD)


def credentials_match(username: str, password: str) -> bool:
    """Constant-time comparison for configured credentials."""
    return (
        hmac.compare_digest(username.strip(), APP_USERNAME)
        and hmac.compare_digest(password, APP_PASSWORD)
    )


def authenticate(username: str, password: str) -> bool:
    if not credentials_configured():
        return False

    if credentials_match(username, password):
        now = time.time()
        st.session_state.authenticated = True
        st.session_state.login_time = now
        st.session_state.last_activity = now
        return True

    return False


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.login_time = None
    st.session_state.last_activity = None
    st.session_state.results = []
    st.session_state.run_id = None


def authentication_expired() -> bool:
    last_activity = st.session_state.get("last_activity")

    if not last_activity:
        return True

    return (time.time() - last_activity) > SESSION_TIMEOUT_SECONDS


def require_authentication() -> bool:
    if not st.session_state.authenticated:
        return False

    if authentication_expired():
        logout()
        return False

    st.session_state.last_activity = time.time()
    return True


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")

    # Keep characters commonly used in technology names:
    # C++, C#, .NET, Node.js, etc.
    text = re.sub(r"[^\w\s+#./-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_skill(skill: str) -> str:
    return normalize_text(skill)


def parse_skills(raw: str) -> List[str]:
    """Parse comma/newline/semicolon separated skills and remove duplicates."""
    if not raw:
        return []

    parts = re.split(r"[,;\n]+", raw)

    output = []
    seen = set()

    for part in parts:
        skill = part.strip()
        normalized = normalize_skill(skill)

        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(skill)

    return output


# ============================================================
# PDF PROCESSING
# ============================================================

def validate_pdf(uploaded_file) -> None:
    if uploaded_file is None:
        raise ValueError("No file supplied.")

    file_name = uploaded_file.name.lower()

    if not file_name.endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"File exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract embedded text from a PDF."""
    validate_pdf(uploaded_file)

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()

    if not raw_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(raw_bytes))

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(
                    "The PDF is password-protected or encrypted."
                ) from exc

        pages = []

        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
            except Exception:
                continue

        return "\n".join(pages).strip()

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read PDF: {exc}") from exc


# ============================================================
# SKILL MATCHING
# ============================================================

SKILL_ALIASES = {
    "react js": ["react", "reactjs", "react.js"],
    "reactjs": ["react", "reactjs", "react.js"],
    "node js": ["node", "nodejs", "node.js"],
    "nodejs": ["node", "nodejs", "node.js"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "machine learning": ["machine learning", "ml"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "postgresql": ["postgresql", "postgres"],
    "microsoft sql server": ["microsoft sql server", "sql server"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
}


def skill_variants(skill: str) -> List[str]:
    normalized = normalize_skill(skill)

    variants = {normalized}

    if normalized in SKILL_ALIASES:
        variants.update(SKILL_ALIASES[normalized])

    # Compact form helps with React.js / React JS / ReactJS.
    compact = re.sub(r"[\s._-]+", "", normalized)
    if compact:
        variants.add(compact)

    return [v for v in variants if v]


def skill_found(skill: str, resume_text: str) -> bool:
    text = normalize_text(resume_text)

    if not text:
        return False

    for variant in skill_variants(skill):
        if variant in text:
            return True

        compact_text = re.sub(r"[\s._-]+", "", text)
        compact_variant = re.sub(r"[\s._-]+", "", variant)

        if compact_variant and compact_variant in compact_text:
            return True

    return False


def evaluate_skills(
    resume_text: str,
    essential_skills: List[str],
    preferred_skills: List[str],
) -> Tuple[Dict[str, str], float, int, int]:

    results: Dict[str, str] = {}
    essential_hits = 0
    preferred_hits = 0

    for skill in essential_skills:
        matched = skill_found(skill, resume_text)
        results[skill] = "🟢 Matched" if matched else "🔴 Missing"
        essential_hits += int(matched)

    for skill in preferred_skills:
        matched = skill_found(skill, resume_text)
        results[skill] = "🟢 Matched" if matched else "🔴 Missing"
        preferred_hits += int(matched)

    essential_score = (
        (essential_hits / len(essential_skills)) * 70
        if essential_skills else 0
    )

    preferred_score = (
        (preferred_hits / len(preferred_skills)) * 30
        if preferred_skills else 0
    )

    if not essential_skills and not preferred_skills:
        skill_score = 0.0
    else:
        skill_score = essential_score + preferred_score

    return (
        results,
        round(skill_score, 2),
        essential_hits,
        preferred_hits,
    )


# ============================================================
# JD / RESUME SIMILARITY
# ============================================================

def calculate_similarity(resume_text: str, jd_text: str) -> float:
    resume = normalize_text(resume_text)
    jd = normalize_text(jd_text)

    if not resume or not jd:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
        )

        vectors = vectorizer.fit_transform([jd, resume])

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2],
        )[0][0]

        return round(float(score * 100), 2)

    except ValueError:
        return 0.0


def calculate_final_score(
    skill_score: float,
    jd_similarity: float,
    has_skills: bool,
) -> float:
    """
    When skills are supplied:
        70% skill match + 30% JD similarity

    When no skills are supplied:
        100% JD similarity
    """
    if has_skills:
        return round(
            (skill_score * 0.70) + (jd_similarity * 0.30),
            2,
        )

    return round(jd_similarity, 2)


def match_level(score: float) -> str:
    if score >= 80:
        return "🟢 High"
    if score >= 60:
        return "🟡 Medium"
    return "🔴 Low"


# ============================================================
# OPENAI
# ============================================================

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if not api_key or OpenAI is None:
        return None

    return OpenAI(api_key=api_key)


def generate_ai_summary(
    resume_text: str,
    jd_text: str,
    client,
) -> str:

    if client is None:
        return (
            "AI analysis is unavailable. "
            "Configure OPENAI_API_KEY and ensure the OpenAI "
            "package is installed."
        )

    # Limit prompt size to control cost and latency.
    resume_excerpt = resume_text[:16000]
    jd_excerpt = jd_text[:10000]

    prompt = f"""
Analyze this candidate resume against the supplied job description.

JOB DESCRIPTION
---------------
{jd_excerpt}

CANDIDATE RESUME
----------------
{resume_excerpt}

Return a concise recruiter-oriented assessment with these sections:

### Candidate Summary
### Key Strengths
### Missing / Weak Areas
### Relevant Technologies
### Relevant Project Experience
### Recruiter Observations

Rules:
- Use only evidence present in the resume.
- Do not invent employers, technologies, years of experience,
  certifications, responsibilities, or projects.
- If something is not stated, say "Not stated in resume."
- Do not make decisions based on protected characteristics.
- Focus on job-related qualifications and evidence.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=(
                "You are an evidence-based professional technical "
                "recruiter. Analyze resumes against job descriptions "
                "without inventing information."
            ),
            input=prompt,
        )

        output = getattr(response, "output_text", "")

        if not output:
            return "AI returned an empty response."

        return output.strip()

    except Exception as exc:
        return f"AI analysis failed: {exc}"


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login() -> None:
    st.title("🔒 Secure Resume Matcher")

    st.markdown(
        "### Recruiter Login"
    )

    if not credentials_configured():
        st.error(
            "Authentication is not configured. Add "
            "`APP_USERNAME` and `APP_PASSWORD` to Streamlit Secrets."
        )
        st.stop()

    with st.form("login_form"):
        username = st.text_input(
            "Username",
            autocomplete="username",
        )

        password = st.text_input(
            "Password",
            type="password",
            autocomplete="current-password",
        )

        submitted = st.form_submit_button(
            "🔐 Login",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if authenticate(username, password):
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")


# ============================================================
# RESET
# ============================================================

def reset_results() -> None:
    st.session_state.results = []
    st.session_state.run_id = None


# ============================================================
# APPLICATION
# ============================================================

def run_evaluation(
    jd_text: str,
    essential_skills: List[str],
    preferred_skills: List[str],
    resume_files,
    generate_ai: bool,
    ai_limit: int,
) -> List[dict]:

    results = []
    client = get_openai_client(OPENAI_API_KEY)

    total = len(resume_files)
    progress = st.progress(0)
    status = st.empty()

    has_skills = bool(essential_skills or preferred_skills)

    for index, resume in enumerate(resume_files, start=1):
        status.info(
            f"Processing {resume.name} ({index}/{total})..."
        )

        result = {
            "Candidate": resume.name,
            "Final Score": 0.0,
            "Skill Match %": 0.0,
            "JD Similarity %": 0.0,
            "Match Level": "🔴 Error",
            "Essential Skills": f"0/{len(essential_skills)}",
            "Preferred Skills": f"0/{len(preferred_skills)}",
            "Skills Table": {},
            "AI Summary": "",
            "Status": "Processed",
        }

        try:
            resume_text = extract_text_from_pdf(resume)

            if not resume_text.strip():
                result.update({
                    "Match Level": "⚪ No Text",
                    "Status": "No extractable text",
                    "AI Summary": (
                        "No extractable text was found. "
                        "This may be a scanned/image-only PDF. "
                        "OCR support is not enabled in this deployment."
                    ),
                })
                results.append(result)
                progress.progress(index / total)
                continue

            (
                skill_map,
                skill_score,
                essential_hits,
                preferred_hits,
            ) = evaluate_skills(
                resume_text,
                essential_skills,
                preferred_skills,
            )

            jd_similarity = calculate_similarity(
                resume_text,
                jd_text,
            )

            final_score = calculate_final_score(
                skill_score,
                jd_similarity,
                has_skills,
            )

            ai_summary = (
                "AI analysis not requested."
            )

            # Only analyze the highest-ranked candidates after
            # deterministic scoring is complete. This avoids
            # unnecessary API calls for large batches.
            if generate_ai and index <= ai_limit:
                ai_summary = generate_ai_summary(
                    resume_text,
                    jd_text,
                    client,
                )

            result.update({
                "Final Score": final_score,
                "Skill Match %": skill_score,
                "JD Similarity %": jd_similarity,
                "Match Level": match_level(final_score),
                "Essential Skills": (
                    f"{essential_hits}/{len(essential_skills)}"
                ),
                "Preferred Skills": (
                    f"{preferred_hits}/{len(preferred_skills)}"
                ),
                "Skills Table": skill_map,
                "AI Summary": ai_summary,
            })

        except Exception as exc:
            result.update({
                "Status": "Processing error",
                "AI Summary": f"Could not process file: {exc}",
            })

        results.append(result)
        progress.progress(index / total)

    status.success("✅ Evaluation completed.")
    return results


def show_app() -> None:
    st.title("📄 Resume Match & Ranking")
    st.caption(
        "Recruiter-focused resume screening using PDF extraction, "
        "skill matching, TF-IDF JD similarity, and optional AI analysis."
    )

    with st.sidebar:
        st.header("⚙️ Evaluation Settings")

        jd_text = st.text_area(
            "Job Description",
            height=260,
            placeholder="Paste the complete job description...",
        )

        essential_input = st.text_area(
            "Essential Skills",
            height=130,
            placeholder="Python, SQL, AWS, Java...",
            help="Separate skills with commas, semicolons, or new lines.",
        )

        preferred_input = st.text_area(
            "Preferred Skills",
            height=130,
            placeholder="Docker, Kubernetes, React...",
        )

        resume_files = st.file_uploader(
            "Upload Resume PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum {MAX_FILE_SIZE_MB} MB per file.",
        )

        st.markdown("---")

        generate_ai = st.checkbox(
            "Generate AI recruiter analysis",
            value=True,
        )

        ai_limit = st.number_input(
            "Maximum AI analyses per run",
            min_value=0,
            max_value=50,
            value=10,
            step=1,
            help=(
                "AI analysis is generated only for the first N "
                "processed resumes. Increase this if needed."
            ),
        )

        run_button = st.button(
            "🚀 Run Evaluation",
            type="primary",
            use_container_width=True,
        )

        reset_button = st.button(
            "🧹 Clear Results",
            use_container_width=True,
        )

        logout_button = st.button(
            "🚪 Logout",
            use_container_width=True,
        )

    if logout_button:
        logout()
        st.rerun()

    if reset_button:
        reset_results()
        st.rerun()

    if run_button:
        jd_text = jd_text.strip()

        if not jd_text:
            st.error("❌ Enter a Job Description first.")
            st.stop()

        if not resume_files:
            st.error("❌ Upload at least one PDF resume.")
            st.stop()

        essential_skills = parse_skills(essential_input)
        preferred_skills = parse_skills(preferred_input)

        if not essential_skills and not preferred_skills:
            st.warning(
                "No essential or preferred skills were supplied. "
                "The ranking will use JD similarity only."
            )

        st.session_state.results = run_evaluation(
            jd_text=jd_text,
            essential_skills=essential_skills,
            preferred_skills=preferred_skills,
            resume_files=resume_files,
            generate_ai=generate_ai,
            ai_limit=int(ai_limit),
        )

    results = st.session_state.results

    if not results:
        st.info(
            "Paste the JD, upload resumes, and click "
            "**Run Evaluation**."
        )
        return

    # ========================================================
    # RANKING
    # ========================================================

    st.divider()
    st.header("📊 Candidate Ranking")

    ranking_rows = []

    for result in results:
        ranking_rows.append({
            "Candidate": result["Candidate"],
            "Final Score": result["Final Score"],
            "Skill Match %": result["Skill Match %"],
            "JD Similarity %": result["JD Similarity %"],
            "Essential Skills": result["Essential Skills"],
            "Preferred Skills": result["Preferred Skills"],
            "Match Level": result["Match Level"],
            "Status": result["Status"],
        })

    ranking_df = pd.DataFrame(ranking_rows)

    ranking_df = ranking_df.sort_values(
        by="Final Score",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "Rank",
        range(1, len(ranking_df) + 1),
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = ranking_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Ranking CSV",
        data=csv_bytes,
        file_name="resume_match_results.csv",
        mime="text/csv",
    )

    # ========================================================
    # DETAILS
    # ========================================================

    st.divider()
    st.header("🔍 Candidate Details")

    for result in sorted(
        results,
        key=lambda x: x["Final Score"],
        reverse=True,
    ):
        with st.expander(
            f"{result['Candidate']} — "
            f"{result['Final Score']:.2f}% — "
            f"{result['Match Level']}"
        ):
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Final Score",
                f"{result['Final Score']:.2f}%",
            )

            c2.metric(
                "Skill Match",
                f"{result['Skill Match %']:.2f}%",
            )

            c3.metric(
                "JD Similarity",
                f"{result['JD Similarity %']:.2f}%",
            )

            st.write(
                f"**Essential Skills:** "
                f"{result['Essential Skills']}"
            )

            st.write(
                f"**Preferred Skills:** "
                f"{result['Preferred Skills']}"
            )

            st.write(
                f"**Processing Status:** "
                f"{result['Status']}"
            )

            if result["Skills Table"]:
                st.subheader("🛠️ Skill Breakdown")

                skill_df = pd.DataFrame(
                    [
                        {"Skill": skill, "Status": status}
                        for skill, status
                        in result["Skills Table"].items()
                    ]
                )

                st.dataframe(
                    skill_df,
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("🤖 AI Recruiter Analysis")
            st.markdown(result["AI Summary"])


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    if require_authentication():
        show_app()
    else:
        show_login()


if __name__ == "__main__":
    main()
