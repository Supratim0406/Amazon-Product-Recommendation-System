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
tfidf_matrix = tfidf_vectorizer.fit_transform(amazon_df['keywords']).toarray()

```

## 🔢 Cosine Similarity

```
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)

```
## 🧠 Product Recommendation Function

```
def product_recommender(query):
    matches = amazon_df[amazon_df['name'].str.lower() == query.lower()]
    if matches.empty:
        print(f"No product found matching '{query}'.")
        return

    product_index = matches.index[0]
    similarity_list = list(enumerate(similarity[product_index]))
    top_10 = sorted(similarity_list, key=lambda x: x[1], reverse=True)[1:11]

    print(f"\nTop recommendations for '{query}':\n")
    for idx, score in top_10:
        print(f"- {amazon_df.iloc[idx]['name']}")


```

## 🧠 How the Recommendation Works

TF-IDF Encoding – turns product names into numerical vectors

Cosine Similarity – measures how close two vectors are in meaning

Ranking – sorts products by similarity score

Recommendation – returns top N most similar products

## 💾 Saving Model Artifacts

```
import pickle

pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(amazon_df, open('amazon_dict.pkl', 'wb'))

```

## 🧩 Future Improvements

Add fuzzy string matching for partial search queries

Use product descriptions or user reviews for deeper similarity

Integrate into a Streamlit / Flask web app

Combine with collaborative filtering for hybrid recommendations
