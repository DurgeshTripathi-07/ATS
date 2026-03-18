import gradio as gr
import pandas as pd
import re
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
import torch

print("🚀 Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 1. LOAD DATA ---
csv_files = ['dataset.csv', 'resume_dataset_1200.csv']
df_list = []

for f in csv_files:
    try:
        df = pd.read_csv(f).fillna("")
        df.columns = [c.lower() for c in df.columns]

        title = next((c for c in df.columns if "title" in c or "role" in c), None)
        jd = next((c for c in df.columns if "description" in c or "jd" in c), None)
        skill = next((c for c in df.columns if "skill" in c), None)

        if title and jd:
            temp = pd.DataFrame({
                'Title': df[title],
                'JD': df[jd],
                'Skills': df[skill] if skill else ""
            })
            df_list.append(temp)
    except:
        pass

if df_list:
    job_db = pd.concat(df_list, ignore_index=True)
    all_skills = " ".join(job_db['Skills'].astype(str).tolist())
    valid_skills = set(all_skills.lower().replace(",", " ").split())

    print("⏳ Encoding Job Database...")
    job_embeddings = model.encode(job_db['JD'].astype(str).tolist(), convert_to_tensor=True)
else:
    job_db = pd.DataFrame()
    job_embeddings = None
    valid_skills = set()

# --- 2. TEXT EXTRACTION ---
def extract_text(file_obj):
    try:
        path = file_obj.name
        if path.endswith('.pdf'):
            reader = PyPDF2.PdfReader(path)
            return "".join([p.extract_text() or "" for p in reader.pages])
        elif path.endswith('.docx'):
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except:
        return ""

# --- 3. QUALITY SCORING ---
def calculate_quality_score(text):
    score = 0
    feedback = []

    words = len(text.split())
    if words > 300:
        score += 10; feedback.append("✅ Length: Good depth.")
    else:
        feedback.append("⚠️ Length: Too short.")

    if re.search(r'@', text) and re.search(r'\d{10}', text):
        score += 10; feedback.append("✅ Contact info present.")
    else:
        feedback.append("⚠️ Missing email/phone.")

    sections = ['education','experience','skills','projects']
    if all(s in text.lower() for s in sections):
        score += 20; feedback.append("✅ Resume structure complete.")
    else:
        feedback.append("⚠️ Missing sections.")

    found_skills = [w for w in text.lower().split() if w in valid_skills]
    unique = len(set(found_skills))
    if unique >= 20:
        score += 20; feedback.append("✅ Strong skills section.")
    elif unique >= 10:
        score += 10; feedback.append("⚠️ Moderate skills.")
    else:
        feedback.append("❌ Weak skills.")

    metrics = re.findall(r'\d+%|\$\d+|\+\d+', text)
    if len(metrics) >= 3:
        score += 20; feedback.append("✅ Quantified impact.")
    else:
        feedback.append("❌ No metrics found.")

    verbs = ['led','built','developed','created','managed','optimized']
    if sum(v in text.lower() for v in verbs) >= 2:
        score += 20; feedback.append("✅ Action-oriented language.")
    else:
        feedback.append("❌ Passive wording.")

    return score, "\n".join(feedback)

# --- 4. JOB RECOMMENDATION (FIXED) ---
def find_jobs(text):
    if job_embeddings is None or job_db.empty:
        return "Database empty."

    resume_emb = model.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(resume_emb, job_embeddings)[0]

    career_scores = {}

    for idx, score in enumerate(scores):
        title = str(job_db.iloc[idx]['Title']).strip().title()
        score_val = score.item()
        career_scores[title] = max(career_scores.get(title, 0), score_val)

    sorted_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)
    top_careers = sorted_careers[:5]

    total = sum(s for _, s in top_careers)
    results = [
        f"• {t} ({round((s/total)*100,1)}%)"
        for t, s in top_careers
    ]

    return "\n".join(results)

# --- 5. MAIN API ---
def analyze_resume(file_obj):
    if file_obj is None:
        return "0", "No file uploaded", "No jobs"

    text = extract_text(file_obj)
    if not text:
        return "0", "Could not read file", ""

    score, feedback = calculate_quality_score(text)
    jobs = find_jobs(text)

    return str(score), feedback, jobs

# --- 6. LAUNCH ---
demo = gr.Interface(
    fn=analyze_resume,
    inputs=gr.File(label="Upload Resume"),
    outputs=[
        gr.Textbox(label="Resume Score"),
        gr.Textbox(label="Improvement Feedback"),
        gr.Textbox(label="Recommended Careers")
    ]
)

demo.launch()