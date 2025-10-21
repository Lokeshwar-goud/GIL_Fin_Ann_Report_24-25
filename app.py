
import streamlit as st
import pdfplumber
import re
import io
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
import lda
from tempfile import NamedTemporaryFile

# ---------------------------
# Initialization / setup
# ---------------------------
st.set_page_config(page_title="Godrej Annual Report — NLP Explorer",
                   page_icon="📄",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Ensure required NLTK corpora are downloaded (first run)
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab', quiet=True)


download_nltk_resources()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Utility functions
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
    """
    Heuristic: detect lines that are all-caps or Title Case and short -> treat as headers
    Returns list of {'section_heading', 'content'}
    """
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

@st.cache_data
def build_vectors(documents, max_df=0.9, min_df=2):
    cv = CountVectorizer(max_df=max_df, min_df=min_df, ngram_range=(1,1))
    dtm = cv.fit_transform(documents)
    tfidf = TfidfVectorizer(max_df=0.9, min_df=min_df)
    tfidf_mat = tfidf.fit_transform(documents)
    return cv, dtm, tfidf, tfidf_mat

@st.cache_data
def run_lda(dtm_array, n_topics=10, n_iter=3000, random_state=42):
    model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_state)
    model.fit(dtm_array)
    return model

def plot_wordcloud(freq_dict, width=900, height=450):
    wc = WordCloud(width=width, height=height, background_color='white', collocations=False)
    wc.generate_from_frequencies(freq_dict)
    fig = plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return fig

# ---------------------------
# Sidebar: Upload / options
# ---------------------------
st.sidebar.title("Controls")
st.sidebar.markdown("Upload your Annual Report PDF (FY 2024-25).")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])

use_sample = st.sidebar.checkbox("Use sample uploaded Godrej PDF (if available on server)", value=False)

min_df = st.sidebar.slider("min_df (CountVectorizer)", 1, 10, 2)
n_topics = st.sidebar.slider("Number of LDA topics", 2, 15, 10)
n_iter = st.sidebar.slider("LDA iterations", 500, 8000, 3000, step=500)

extra_stop = st.sidebar.text_input("Extra stopwords (comma-separated)", value="godrej,industries,ltd,limited,company,year")
extra_stopwords = [s.strip().lower() for s in extra_stop.split(",") if s.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("**App features:**")
st.sidebar.markdown("- Upload PDF or use sample\n- Section-based segmentation (heuristic)\n- Sentence-level sentiment (TextBlob)\n- Wordcloud + top frequent words\n- TF-IDF & Document-Term Matrix\n- LDA topic modelling (Gibbs sampling)\n- Download CSV outputs")

# ---------------------------
# Main UI
# ---------------------------
st.title("📄 Godrej Annual Report (FY 2024–25) — NLP Explorer")
st.markdown("Interactive Streamlit app to extract, analyze, and visualize text from an annual report. "
            "Use the sidebar to upload a PDF and tune modelling parameters.")

if uploaded_file is None and not use_sample:
    st.info("Upload the Godrej Annual Report PDF (.pdf) from the sidebar, or tick the sample option.")
    st.stop()

# Save uploaded to temp file (pdfplumber can open BytesIO directly; convert to file-like)
file_like = None
if uploaded_file:
    file_like = uploaded_file
else:
    # If you're running locally and want to use a pre-uploaded path, check standard locations
    # (This branch will rarely be used on Streamlit Cloud)
    sample_paths = ['/content/GIL_Annual_Report_2024-25.pdf', '/mnt/data/GIL_Annual_Report_2024-25.pdf']
    sample_path = None
    for p in sample_paths:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path:
        file_like = open(sample_path, 'rb')
    else:
        st.error("Sample file not found on server. Please upload PDF.")
        st.stop()

# ---------------------------
# Extract pages -> DataFrame
# ---------------------------
with st.spinner("Extracting text from PDF (this may take a while for large PDFs)..."):
    df_pages = extract_pdf_text(file_like)
st.success(f"Extracted {len(df_pages)} pages.")

# Show first pages preview
with st.expander("Preview: first 3 pages (raw text)"):
    for i in range(min(3, len(df_pages))):
        st.write(f"### Page {df_pages.loc[i,'page']}")
        st.write(df_pages.loc[i,'text'][:2000] + ("..." if len(df_pages.loc[i,'text']) > 2000 else ""))

# Build full raw text and clean
full_raw = " 
 ".join(df_pages['text'].fillna('').tolist())
full_clean = preprocess_text(full_raw, extra_stopwords=extra_stopwords)

# Section segmentation attempt
sections = section_segmentation(full_raw)
if sections and len(sections) >= 4:
    st.success(f"Section-based segmentation detected {len(sections)} sections — using sections as documents.")
    df_sections = pd.DataFrame(sections)
    df_sections['clean'] = df_sections['content'].apply(lambda t: preprocess_text(t, extra_stopwords=extra_stopwords))
    documents = df_sections['clean'].tolist()
    use_sections = True
else:
    st.warning("Section-based segmentation did not detect reliable headings. Falling back to using pages as documents.")
    df_pages['clean'] = df_pages['text'].fillna('').apply(lambda t: preprocess_text(t, extra_stopwords=extra_stopwords))
    documents = df_pages['clean'].tolist()
    use_sections = False

# Sentence-level sentiment
sentences = sent_tokenize(full_raw)
if len(sentences) > 0:
    df_sentiments = compute_sentiments(sentences)
    st.subheader("Sentence-level Sentiment (summary)")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.write("Polarity distribution (negative→positive)")
        fig = px.histogram(df_sentiments, x='polarity', nbins=40, labels={'polarity': 'Polarity'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Avg polarity", f"{df_sentiments['polarity'].mean():.3f}")
    with col3:
        st.metric("Avg subjectivity", f"{df_sentiments['subjectivity'].mean():.3f}")

    # Show most positive and most negative sentences
    top_pos = df_sentiments.sort_values('polarity', ascending=False).head(5)
    top_neg = df_sentiments.sort_values('polarity', ascending=True).head(5)
    with st.expander("Top positive sentences"):
        for idx, row in top_pos.iterrows():
            st.write(f"**Polarity {row['polarity']:.3f}** — {row['sentence']}")
    with st.expander("Top negative sentences"):
        for idx, row in top_neg.iterrows():
            st.write(f"**Polarity {row['polarity']:.3f}** — {row['sentence']}")
else:
    st.info("No sentences found for sentiment analysis.")

# Word tokens and word frequency
st.subheader("Word Frequency & WordCloud")
word_tokens = word_tokenize(full_clean)
freq = Counter(word_tokens)
top_n = 50
top_words = freq.most_common(top_n)
freq_df = pd.DataFrame(top_words, columns=['word','freq'])
col1, col2 = st.columns([2,3])
with col1:
    st.table(freq_df.head(15))
with col2:
    wc_fig = plot_wordcloud(dict(top_words))
    st.pyplot(wc_fig)

# ---------------------------
# Topic Modeling (using Gensim)
# ---------------------------
from gensim import corpora, models

st.subheader("Topic Modeling (LDA - Gensim)")
st.markdown(f"Documents used: **{len(documents)}** — Topics: **{n_topics}**")

# Tokenize each document
texts = [doc.split() for doc in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

with st.spinner("Training LDA model using Gensim..."):
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=10,
        random_state=42
    )

st.success(f"LDA model trained successfully with {n_topics} topics!")

# Display the top words for each topic
for idx, topic in lda_model.print_topics(num_topics=n_topics, num_words=10):
    st.markdown(f"**Topic {idx+1}**: {topic}")

# Assign dominant topic for each document
dominant_topics = []
for doc_bow in corpus:
    topic_probs = lda_model.get_document_topics(doc_bow)
    dominant = max(topic_probs, key=lambda x: x[1])[0] + 1 if topic_probs else 0
    dominant_topics.append(dominant)

# Attach topic assignments
if use_sections:
    df_sections['dominant_topic'] = dominant_topics
    st.dataframe(df_sections[['section_heading', 'dominant_topic']].head(10))
    csv_sections = df_sections.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download sections + topics (CSV)", csv_sections, file_name="sections_with_topics.csv")
else:
    df_pages['dominant_topic'] = dominant_topics
    st.dataframe(df_pages[['page', 'dominant_topic']].head(10))
    csv_pages = df_pages.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download pages + topics (CSV)", csv_pages, file_name="pages_with_topics.csv")

# Visualize topic distribution
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
fig = go.Figure(data=[go.Bar(x=topic_counts.index, y=topic_counts.values)])
fig.update_layout(title="Topic Distribution", xaxis_title="Topic", yaxis_title="Document Count")
st.plotly_chart(fig, use_container_width=True)
