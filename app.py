# app.py
import streamlit as st
import pdfplumber
import re
import os
from collections import Counter
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora, models

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="Godrej Annual Report — NLP Explorer",
                   page_icon="📄",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# NLTK Setup (fixed for Streamlit Cloud)
# ---------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

download_nltk_resources()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Helper functions
# ---------------------------
def preprocess_text(text, remove_stopwords=True, lemmatize=True, extra_stopwords=None):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    if remove_stopwords:
        sw = set(stop_words)
        if extra_stopwords:
            sw = sw.union(set(extra_stopwords))
        tokens = [t for t in tokens if t not in sw and len(t) > 1]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def extract_pdf_text(file_stream):
    pages = []
    with pdfplumber.open(file_stream) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            pages.append({'page': i + 1, 'text': text or ''})
    return pd.DataFrame(pages)

def section_segmentation(raw_text):
    """Detect section headers heuristically"""
    lines = [ln.strip() for ln in raw_text.split('\\n') if ln.strip()]
    header_idxs = []
    for i, ln in enumerate(lines):
        words = ln.split()
        if len(words) <= 8 and len(ln) > 3:
            if ln.isupper() or (sum(1 for w in words if w[0].isupper()) >= max(1, len(words)//2)):
                header_idxs.append(i)
    if not header_idxs:
        return None
    sections = []
    for idx_idx, h in enumerate(header_idxs):
        start = h
        end = header_idxs[idx_idx + 1] if idx_idx + 1 < len(header_idxs) else len(lines)
        header = lines[start]
        content = ' '.join(lines[start+1:end]).strip()
        sections.append({'section_heading': header, 'content': content})
    return sections

@st.cache_data
def compute_sentiments(sentences):
    rows = []
    for s in sentences:
        tb = TextBlob(s)
        rows.append({'sentence': s, 'polarity': tb.sentiment.polarity, 'subjectivity': tb.sentiment.subjectivity})
    return pd.DataFrame(rows)

def plot_wordcloud(freq_dict, width=900, height=450):
    wc = WordCloud(width=width, height=height, background_color='white', collocations=False)
    wc.generate_from_frequencies(freq_dict)
    fig = plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return fig

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Annual Report (PDF)", type=['pdf'])
min_df = st.sidebar.slider("min_df (for Vectorizer)", 1, 10, 2)
n_topics = st.sidebar.slider("Number of Topics", 2, 15, 10)
extra_stop = st.sidebar.text_input("Extra stopwords (comma-separated)",
                                   value="godrej,industries,ltd,limited,company,year,report")
extra_stopwords = [s.strip().lower() for s in extra_stop.split(",") if s.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("**This app performs:**")
st.sidebar.markdown("- PDF extraction & cleaning\n- Sentiment analysis\n- WordCloud\n- Topic Modeling (LDA)\n- CSV downloads")

# ---------------------------
# Main App Logic
# ---------------------------
st.title("📄 Godrej Industries Annual Report (FY 2024–25) — NLP Explorer")

if uploaded_file is None:
    st.info("📤 Please upload the Godrej Annual Report PDF using the sidebar.")
    st.stop()

# Extract pages
with st.spinner("Extracting text from PDF..."):
    df_pages = extract_pdf_text(uploaded_file)
st.success(f"Extracted {len(df_pages)} pages successfully!")

# Preview
with st.expander("Preview of first 2 pages"):
    for i in range(min(2, len(df_pages))):
        st.write(f"### Page {df_pages.loc[i,'page']}")
        st.write(df_pages.loc[i,'text'][:1500])

# Combine and clean
full_raw = " \n ".join(df_pages['text'].fillna('').tolist())
full_clean = preprocess_text(full_raw, extra_stopwords=extra_stopwords)

# Section segmentation
sections = section_segmentation(full_raw)
if sections and len(sections) >= 4:
    st.success(f"✅ Found {len(sections)} sections. Using sections for topic modeling.")
    df_sections = pd.DataFrame(sections)
    df_sections['clean'] = df_sections['content'].apply(lambda t: preprocess_text(t, extra_stopwords=extra_stopwords))
    documents = df_sections['clean'].tolist()
    use_sections = True
else:
    st.warning("⚠️ Section segmentation unreliable. Using pages instead.")
    df_pages['clean'] = df_pages['text'].apply(lambda t: preprocess_text(t, extra_stopwords=extra_stopwords))
    documents = df_pages['clean'].tolist()
    use_sections = False

# ---------------------------
# Sentiment Analysis
# ---------------------------
st.subheader("🧠 Sentence-level Sentiment Analysis")
sentences = sent_tokenize(full_raw)
if len(sentences) > 0:
    df_sentiments = compute_sentiments(sentences)
    col1, col2 = st.columns([3,1])
    with col1:
        fig = px.histogram(df_sentiments, x='polarity', nbins=40, title='Polarity Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Avg Polarity", f"{df_sentiments['polarity'].mean():.3f}")
        st.metric("Avg Subjectivity", f"{df_sentiments['subjectivity'].mean():.3f}")
else:
    st.info("No sentences found for sentiment analysis.")

# ---------------------------
# Word Frequency & WordCloud
# ---------------------------
st.subheader("🔤 Word Frequency & WordCloud")
word_tokens = word_tokenize(full_clean)
freq = Counter(word_tokens)
top_words = freq.most_common(100)
freq_df = pd.DataFrame(top_words, columns=['word','freq'])
col1, col2 = st.columns([1,2])
with col1:
    st.dataframe(freq_df.head(20))
with col2:
    fig_wc = plot_wordcloud(dict(top_words))
    st.pyplot(fig_wc)

# ---------------------------
# Topic Modeling (Gensim LDA)
# ---------------------------
st.subheader("📊 Topic Modeling (LDA via Gensim)")
st.markdown(f"Documents used: **{len(documents)}** — Topics: **{n_topics}**")

texts = [doc.split() for doc in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

with st.spinner("Training LDA Model..."):
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=10,
        random_state=42
    )
st.success("LDA Model Trained Successfully!")

# Show topics
for idx, topic in lda_model.print_topics(num_topics=n_topics, num_words=10):
    st.markdown(f"**Topic {idx+1}**: {topic}")

# Assign dominant topic
dominant_topics = []
for doc_bow in corpus:
    topic_probs = lda_model.get_document_topics(doc_bow)
    dominant = max(topic_probs, key=lambda x: x[1])[0] + 1 if topic_probs else 0
    dominant_topics.append(dominant)

# Add to dataframe
if use_sections:
    df_sections['dominant_topic'] = dominant_topics
    st.dataframe(df_sections[['section_heading','dominant_topic']].head(10))
    csv_data = df_sections.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sections with Topics (CSV)", csv_data, file_name="sections_with_topics.csv")
else:
    df_pages['dominant_topic'] = dominant_topics
    st.dataframe(df_pages[['page','dominant_topic']].head(10))
    csv_data = df_pages.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Pages with Topics (CSV)", csv_data, file_name="pages_with_topics.csv")

# Topic distribution chart
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
fig_bar = go.Figure([go.Bar(x=topic_counts.index, y=topic_counts.values)])
fig_bar.update_layout(title="Topic Distribution Across Documents",
                      xaxis_title="Topic", yaxis_title="Count")
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# Downloads
# ---------------------------
st.subheader("📁 Downloadable Outputs")
if 'df_sentiments' in locals():
    st.download_button("📥 Download Sentence Sentiments (CSV)",
                       df_sentiments.to_csv(index=False).encode('utf-8'),
                       file_name="sentence_sentiments.csv")
st.download_button("📥 Download Word Frequencies (CSV)",
                   freq_df.to_csv(index=False).encode('utf-8'),
                   file_name="word_freq.csv")

st.markdown("---")
st.caption("Developed by Lokesh — NLP Mini Project (Godrej Annual Report FY 2024–25)")
