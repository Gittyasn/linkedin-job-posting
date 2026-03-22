from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def get_ngrams(df, n=2, top_k=15):
    """Finds top N-grams (phrases) in descriptions."""
    if 'description' not in df.columns:
        return None
        
    text = df['description'].dropna().sample(frac=0.1, random_state=42).apply(clean_text)
    
    # Align stop words with CountVectorizer's internal tokenizer
    tmp_vec = CountVectorizer()
    tokenize = tmp_vec.build_tokenizer()
    
    stops = set()
    for s in list(STOPWORDS) + ['experience', 'work', 'team', 'skills', 'job', 'will', 'years', 'requirements']:
        stops.update(tokenize(s.lower()))
    
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(stops), max_features=1000)
    try:
        bag_of_words = vec.fit_transform(text)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq[:top_k], columns=['Phrase', 'Count'])
    except ValueError:
        return None

def generate_wordcloud(df, title_filter=None):
    """Generates wordcloud, optionally filtered by job title."""
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        return None

    if 'description' not in df.columns:
        return None
        
    # Filter by title if provided
    if title_filter:
        subset = df[df['clean_title'] == title_filter]
        if len(subset) == 0: return None
        text_data = subset['description'].dropna().str.cat(sep=' ')
    else:
        text_data = df['description'].dropna().sample(frac=0.1, random_state=42).str.cat(sep=' ')

    stopwords = set(STOPWORDS)
    stopwords.update(["job", "description", "requirements", "will", "experience", "work", "team", "looking", "role"])

    try:
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, max_words=100).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Top Skills {'for ' + title_filter if title_filter else '(Overall)'}")
        return fig
    except Exception:
        return None
