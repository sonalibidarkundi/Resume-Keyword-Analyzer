import streamlit as st
import PyPDF2
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Resume Keyword Analyzer",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Functions
# -----------------------------------

def extract_text_from_pdf(pdf_file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# Stop Words
stop_words = {
    "and", "or", "the", "a", "an", "to", "for",
    "with", "in", "on", "of", "is", "are", "be",
    "as", "by", "from", "at", "this", "that",
    "will", "can", "your", "you", "we", "our",
    "their", "has", "have", "had", "job", "role",
    "experience", "skills"
}


def extract_keywords(text):

    words = clean_text(text).split()

    keywords = []

    for word in words:

        if word not in stop_words and len(word) > 2:
            keywords.append(word)

    return keywords


# -----------------------------------
# Header
# -----------------------------------

st.markdown(
    '<div class="title">📄 Resume Keyword Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Compare Resume with Job Description using ATS</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# Inputs
# -----------------------------------

col1, col2 = st.columns(2)

with col1:

    resume_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "Paste Job Description",
        height=250
    )

# -----------------------------------
# Analyze Button
# -----------------------------------

if st.button("Analyze Resume"):

    if resume_file is None:

        st.error("Please upload your resume.")

    elif job_description.strip() == "":

        st.error("Please enter a job description.")

    else:

        # Resume Text
        resume_text = extract_text_from_pdf(resume_file)

        # Clean Text
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        # Keywords
        resume_keywords = extract_keywords(resume_clean)
        jd_keywords = extract_keywords(jd_clean)

        resume_set = set(resume_keywords)
        jd_set = set(jd_keywords)

        # Matched Keywords
        matched_keywords = sorted(
            resume_set.intersection(jd_set)
        )

        # Missing Keywords
        missing_keywords = sorted(
            jd_set.difference(resume_set)
        )

        # ATS Score
        documents = [resume_clean, jd_clean]

        cv = CountVectorizer()

        matrix = cv.fit_transform(documents)

        similarity = cosine_similarity(matrix)[0][1]

        ats_score = round(similarity * 100, 2)

        # -----------------------------------
        # ATS Score
        # -----------------------------------

        st.subheader("📊 ATS Match Score")

        st.markdown(
            f"""
            <div class="card">
                <h1 style="text-align:center; color:green;">
                    {ats_score}%
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------------
        # Resume Keywords
        # -----------------------------------

        st.subheader("📌 Resume Keywords")

        resume_counter = Counter(resume_keywords)

        top_resume_keywords = [
            word for word, count
            in resume_counter.most_common(30)
        ]

        st.markdown(
            f"""
            <div class="card">
                {", ".join(top_resume_keywords)}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------------
        # Job Description Keywords
        # -----------------------------------

        st.subheader("📌 Job Description Keywords")

        jd_counter = Counter(jd_keywords)

        top_jd_keywords = [
            word for word, count
            in jd_counter.most_common(30)
        ]

        st.markdown(
            f"""
            <div class="card">
                {", ".join(top_jd_keywords)}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------------
        # Matched Keywords
        # -----------------------------------

        st.subheader("✅ Matched Keywords")

        if matched_keywords:

            st.markdown(
                f"""
                <div class="card">
                    {", ".join(matched_keywords)}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning("No matched keywords found.")

        # -----------------------------------
        # Missing Keywords
        # -----------------------------------

        st.subheader("❌ Missing Keywords")

        if missing_keywords:

            st.markdown(
                f"""
                <div class="card">
                    {", ".join(missing_keywords)}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.success("No missing keywords found.")

        # -----------------------------------
        # Suggestions
        # -----------------------------------

        st.subheader("💡 Suggestions")

        if ats_score >= 80:

            st.success("Excellent Resume Match!")

        elif ats_score >= 60:

            st.warning(
                "Good match. Add more keywords to improve ATS score."
            )

        else:

            st.error(
                "Low ATS match. Add missing skills and technologies."
            )

        if missing_keywords:

            st.write(
                "Try adding these keywords to your resume:"
            )

            st.write(", ".join(missing_keywords[:15]))

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Built using Python, Streamlit, NLP and Machine Learning"
)