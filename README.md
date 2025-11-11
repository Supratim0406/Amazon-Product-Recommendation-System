# 🛒 Amazon Electronics Product Recommendation System

This project builds a **content-based recommendation system** using product data from Amazon’s Electronics category.  
It recommends similar products based on their textual information (product names and keywords) using **TF-IDF** and **Cosine Similarity**.

---

## 🚀 Features

- Cleaned and preprocessed raw product data  
- Text normalization (lowercasing, punctuation removal, stemming)  
- TF-IDF vectorization with n-grams (1 to 3 words)  
- Cosine similarity for measuring product similarity  
- Top 10 similar product recommendations for any given product  
- Model persistence with `pickle` for deployment

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| Programming Language | Python 3 |
| Data Handling | pandas, numpy |
| NLP | nltk |
| Machine Learning | scikit-learn |
| Model Storage | pickle |

---

## 📦 Dataset

- File: **`All Electronics.csv`**
- Rows: 9,600  
- Columns:  
  - `name` – Product title  
  - `image` – Product image URL  
  - `link` – Amazon product URL  
  - `ratings` – Average customer rating  
  - `no_of_ratings` – Number of ratings  
  - `discount_price` – Discounted price  
  - `actual_price` – Original price  
  - (plus two dropped metadata columns)

---

## 🧹 Data Preprocessing Steps

1. **Removed unnecessary columns:**  
   Dropped `main_category` and `sub_category`.

2. **Cleaned text:**  
   - Converted to lowercase  
   - Removed punctuation and special characters using regex  
   - Removed stopwords (`nltk.corpus.stopwords`)  
   - Applied stemming (`PorterStemmer`)

3. **Created `keywords` column:**  
   Used the cleaned product names as features for similarity computation.

---

## 🧮 TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,3),
    dtype=np.float32
)

## 🔢 Cosine Similarity

```
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)

```


tfidf_matrix = tfidf_vectorizer.fit_transform(amazon_df['keywords']).toarray()
