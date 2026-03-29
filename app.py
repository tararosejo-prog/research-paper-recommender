import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Page title
st.title("Research Paper Recommendation System")

# Intro text
st.write("This system recommends research papers based on the topic entered by the user using Natural Language Processing techniques.")

# Example topics shown on opening
st.markdown("### Example Topics You Can Try:")
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

# Sample topic dropdown
sample_topics = [
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

selected_topic = st.selectbox("Or select a sample topic:", sample_topics)

# User input
user_input = st.text_input("Enter your research topic:")

# Use selected topic if input box is empty
if not user_input:
    user_input = selected_topic

# Load dataset
df = pd.read_csv("papers.csv")
df.columns = df.columns.str.strip().str.lower()

# Combine all important columns into one
df["content"] = (
    df["title"].astype(str) + " " +
    df["domain"].astype(str) + " " +
    df["abstract"].astype(str) + " " +
    df["keywords"].astype(str)
)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess(text):
    text = str(text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
df["processed_content"] = df["content"].apply(preprocess)

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed_content"])

# Button to generate recommendations
if st.button("Recommend Papers"):
    processed_query = preprocess(user_input)
    query_vector = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:5]

    st.subheader("Top Recommended Papers:")
    for i in top_indices:
        st.write("**Title:**", df.iloc[i]["title"])
        st.write("**Domain:**", df.iloc[i]["domain"])
        st.write("**Abstract:**", df.iloc[i]["abstract"])
        st.write("**Keywords:**", df.iloc[i]["keywords"])
        st.write("**Similarity Score:**", round(float(similarity_scores[i]), 3))
        st.write("---")