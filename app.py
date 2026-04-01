import re
from typing import Tuple

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# Initial setup
# ---------------------------
st.set_page_config(
    page_title="Research Paper Recommendation System",
    page_icon="📘",
    layout="wide"
)

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ---------------------------
# Constants
# ---------------------------
REQUIRED_COLUMNS = ["title", "domain", "abstract", "keywords"]

SAMPLE_TOPICS = [
    "Deep Learning in Healthcare",
    "Natural Language Processing",
    "Cybersecurity Threat Detection",
    "IoT in Agriculture",
    "Sentiment Analysis",
    "Recommendation Systems",
    "Artificial Intelligence",
    "Blockchain Security",
    "Speech Recognition",
    "Weather Forecasting"
]


# ---------------------------
# NLP tools
# ---------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ---------------------------
# Helper functions
# ---------------------------
def preprocess_text(text: str) -> str:
    """
    Clean and normalize input text for NLP processing.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


@st.cache_data
def load_dataset(file_path: str = "papers.csv") -> pd.DataFrame:
    """
    Load the dataset and validate required columns.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = df.fillna("")

    df["content"] = (
        df["title"].astype(str) + " " +
        df["domain"].astype(str) + " " +
        df["abstract"].astype(str) + " " +
        df["keywords"].astype(str)
    )

    df["processed_content"] = df["content"].apply(preprocess_text)
    return df


@st.cache_resource
def build_vector_model(processed_text: pd.Series) -> Tuple[TfidfVectorizer, any]:
    """
    Create TF-IDF vectorizer and fit it on processed content.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_text)
    return vectorizer, tfidf_matrix


def get_recommendations(
    user_query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Return top matching research papers based on cosine similarity.
    """
    processed_query = preprocess_text(user_query)

    if not processed_query.strip():
        return pd.DataFrame()

    query_vector = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = df.copy()
    results["similarity_score"] = similarity_scores
    results = results.sort_values(by="similarity_score", ascending=False)

    return results.head(top_n)


def display_recommendations(results: pd.DataFrame) -> None:
    """
    Display the recommended papers in a structured format.
    """
    if results.empty:
        st.warning("No relevant papers found for the given topic.")
        return

    st.subheader("Top Recommended Papers")

    for _, row in results.iterrows():
        with st.container():
            st.markdown(f"### {row['title']}")
            st.write(f"**Domain:** {row['domain']}")
            st.write(f"**Abstract:** {row['abstract']}")
            st.write(f"**Keywords:** {row['keywords']}")
            st.write(f"**Similarity Score:** {row['similarity_score']:.3f}")
            st.markdown("---")


# ---------------------------
# Main interface
# ---------------------------
st.title("📘 Research Paper Recommendation System")
st.write(
    "This application recommends research papers based on the topic entered by the user "
    "using Natural Language Processing techniques."
)

st.markdown("### Example Topics You Can Try")
st.markdown("""
- Deep Learning in Healthcare  
- Natural Language Processing  
- Cybersecurity Threat Detection  
- IoT in Agriculture  
- Sentiment Analysis  
- Recommendation Systems  
- Artificial Intelligence  
- Blockchain Security  
- Speech Recognition  
- Weather Forecasting  
""")

# Sidebar
st.sidebar.header("Search Configuration")
selected_topic = st.sidebar.selectbox("Choose a sample topic", SAMPLE_TOPICS)
top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Load data and model
try:
    df = load_dataset("papers.csv")
    vectorizer, tfidf_matrix = build_vector_model(df["processed_content"])
except FileNotFoundError:
    st.error("The file 'papers.csv' was not found. Please keep it in the same folder as app.py.")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Unexpected error: {error}")
    st.stop()

# Input section
user_input = st.text_input("Enter your research topic:", value=selected_topic)

col1, col2 = st.columns([1, 1])

with col1:
    search_clicked = st.button("Recommend Papers")

with col2:
    clear_clicked = st.button("Clear Results")

if clear_clicked:
    st.rerun()

if search_clicked:
    if not user_input.strip():
        st.warning("Please enter a valid research topic.")
    else:
        recommendations = get_recommendations(
            user_query=user_input,
            df=df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_n=top_n
        )
        display_recommendations(recommendations)

# Footer
st.markdown("---")
st.markdown(
    "<center><b>Research Paper Recommendation System using NLP</b></center>",
    unsafe_allow_html=True
)
