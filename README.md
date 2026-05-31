# 🌍 Travel Destination Recommender System

A hybrid travel destination recommender system built as a final project for the Smart Information Systems and Engineering course. The system combines Collaborative Filtering, Content-Based Filtering, and NLP Sentiment Analysis to deliver personalized travel recommendations through a full-stack web application.

---

## 📌 Project Overview

Most travel platforms recommend destinations based on popularity alone. Our system goes further — it learns from user behavior, destination features, and the sentiment of thousands of real traveler reviews to deliver recommendations that are personal, relevant, and trustworthy.

The system recommends hotels, attractions, tours, and sightseeing experiences grouped by city — making it feel like a full travel planning assistant rather than a simple search engine.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│   User preferences · Filters · Map · Score breakdown   │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP requests
┌─────────────────────────▼───────────────────────────────┐
│                    FastAPI Backend                      │
│   /recommend · /search · /popular · /business/{id}     │
└────────┬──────────────┬───────────────┬─────────────────┘
         │              │               │
┌────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐
│ Collaborative │ │  Content   │ │ NLP Sentiment  │
│  Filtering   │ │   Based    │ │   Analysis     │
│    (ALS)     │ │ Filtering  │ │   (VADER)      │
│   CF Score   │ │  CB Score  │ │ Sentiment Score│
└────────┬──────┘ └─────┬──────┘ └─────┬──────────┘
         └──────────────┴───────────────┘
                         │
              ┌──────────▼──────────┐
              │   Hybrid Fusion     │
              │  CF×0.40 + CB×0.35  │
              │   + Sentiment×0.25  │
              └─────────────────────┘
```

---

## 🤖 Recommendation Models

### 1. Collaborative Filtering (ALS)
Finds users with similar travel behavior and recommends what they visited.

- **Algorithm:** Alternating Least Squares (ALS) via `implicit` library
- **Features:** User ID, Business ID, Star rating (as confidence weight)
- **Why ALS:** 99.97% matrix sparsity — ALS is designed for sparse implicit feedback
- **Cold start:** Popularity-based fallback for new users
- **Training data:** 49,831 interactions from 10,749 power users (3+ reviews)

### 2. Content-Based Filtering (Cosine Similarity)
Recommends destinations similar to places the user already liked.

- **Algorithm:** TF-IDF vectorization + Cosine Similarity
- **Features:** Categories (TF-IDF), star rating, review count (log-normalized), price range, WiFi availability
- **Feature matrix:** 5,131 businesses × 112 features
- **Catalog coverage:** 14.5% across 743 unique businesses per evaluation batch

### 3. NLP Sentiment Analysis (VADER)
Scores every business based on what travelers actually say in their reviews.

- **Algorithm:** VADER (Valence Aware Dictionary and Sentiment Reasoner)
- **Input:** 304,060 cleaned review texts
- **Output:** Compound sentiment score per business (0 to 1)
- **Accuracy:** 74.4% against star ratings, 0.690 Pearson correlation
- **Why VADER:** 95-word median review length, no training required, CPU-efficient

### 4. Hybrid Fusion Engine
Combines all three models into a single ranked recommendation list.

```
Hybrid Score = (0.40 × CF) + (0.35 × CB) + (0.25 × Sentiment)
```

- **Lift over CF alone:** +68.0%
- **Interpretability:** Each recommendation includes a score breakdown and plain-English explanation
- **Filters:** Price range, WiFi, Wheelchair accessibility, City

---

## 📊 Dataset

**Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)

| File | Description | Size |
|---|---|---|
| `business.json` | Business info, categories, attributes | ~120MB |
| `review.json` | User reviews with star ratings and text | ~5GB |
| `user.json` | User profiles, elite status, social graph | ~3GB |

**After filtering to tourism categories (Hotels, Attractions, Tours, Sightseeing, Resorts, Vacation Rentals):**

| Metric | Value |
|---|---|
| Businesses | 5,131 |
| Reviews | 329,797 |
| Users | 245,455 |
| Cities | 100+ across the US |
| Date range | 2005 – 2022 |
| Matrix sparsity | 99.97% |

---

## 🔑 Key EDA Findings

- **Bimodal rating distribution** — travelers give 1★ or 5★, rarely 3★ → VADER works well on polarized opinions
- **July is peak travel month** — clear seasonality in review volume
- **Unhappy users write 60% more** — 1★/2★ reviews average 160 words vs 100 for 5★
- **99.97% sparsity** → ALS chosen over SVD/KNN for CF
- **Boutique B&Bs score highest on sentiment** — consistently outperform large hotel chains
- **Emergent geographic clustering** — CF model learned city-based user segments without being told city information

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python, pandas, NumPy, scikit-learn |
| Collaborative Filtering | implicit (ALS) |
| Content-Based Filtering | scikit-learn TF-IDF + cosine similarity |
| NLP | vaderSentiment |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Map visualization | Folium + streamlit-folium |
| Charts | Plotly |
| Storage | Google Drive (models + data) |
| Notebooks | Google Colab |
| Version control | GitHub |

---

## 📁 Project Structure

```
travel-destination-recommender/
├── app/
│   ├── main.py                  ← FastAPI backend
│   └── streamlit_app.py         ← Streamlit UI
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Collaborative_Filtering.ipynb
│   ├── 05_Content_Based_Filtering.ipynb
│   ├── 06_NLP_Sentiment.ipynb
│   └── 07_Hybrid_Engine.ipynb
├── models/                      ← downloaded from Google Drive
│   ├── als_model.pkl
│   ├── cf_encodings.pkl
│   ├── cb_similarity_matrix.npy
│   ├── cb_encodings.pkl
│   ├── business_sentiment.parquet
│   ├── popularity_baseline.parquet
│   └── user_item_matrix.npz
├── data/                        ← downloaded from Google Drive
│   ├── filtered_businesses.parquet
│   └── processed/
│       ├── biz_index.parquet
│       ├── reviews_cf.parquet
│       └── reviews_nlp.parquet
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** `models/` and `data/` are excluded from Git due to file size. Download from the shared Google Drive folder (link shared with team members).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git
- Access to the shared Google Drive folder (contact team lead)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/travel-destination-recommender.git
cd travel-destination-recommender

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model Files
Download the `models/` and `data/` folders from the shared Google Drive and place them in the project root as shown in the structure above.

### Run the API

```bash
cd app
uvicorn main:app --reload --port 8000
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### Run the UI

```bash
streamlit run app/streamlit_app.py
```

UI will open at `http://localhost:8501`

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Server status and data stats |
| `/recommend?user_id=X` | GET | Hybrid recommendations for a user |
| `/popular` | GET | Most popular destinations |
| `/search?q=hotel` | GET | Search businesses by name |
| `/business/{id}` | GET | Full details for one business |
| `/cities` | GET | All cities with business counts |

**Recommendation filters:**

| Parameter | Type | Example |
|---|---|---|
| `price_filter` | string | `$`, `$$`, `$$$`, `$$$$` |
| `wifi_filter` | string | `free`, `paid`, `no` |
| `wheelchair_filter` | boolean | `true`, `false` |
| `city_filter` | string | `New Orleans` |
| `n` | integer | `10` (max 50) |

---

## 🖥️ UI Features

- **Personalized recommendations** with score breakdown per destination
- **Interactive map** with color-coded sentiment pins
- **Destination cards** showing name, city, stars, price, and category
- **Plain-English explanations** — why each place was recommended
- **Filters** — price range, WiFi, wheelchair accessibility, city
- **Cold start support** — popularity-based recommendations for new users
- **Search** — find any business by name

---

## 📈 Model Evaluation

| Model | Metric | Score |
|---|---|---|
| Collaborative Filtering | Qualitative geographic consistency | ✅ Strong |
| Content-Based Filtering | Catalog coverage | 14.5% |
| Content-Based Filtering | Intra-list diversity | 0.177 |
| NLP Sentiment | Accuracy vs star ratings | 74.4% |
| NLP Sentiment | Pearson correlation | 0.690 |
| Hybrid Engine | Lift over CF alone | +68.0% |
| Hybrid Engine | Evaluated users | 100 |

---

## 🎯 Design Decisions

| Decision | Rationale |
|---|---|
| ALS over SVD | 99.97% sparsity makes SVD unstable |
| VADER over BERT | 95-word median review length, no GPU required, interpretable |
| Price as filter not feature | Only 49% of businesses have price data |
| Remove 2020-2021 reviews | COVID distorted travel behavior patterns |
| Power users only for CF | Users with 3+ reviews provide reliable signal |
| Log-normalize review count | Prevents viral businesses from dominating CB scores |

---

## 👥 Team

| Role | Responsibility |
|---|---|
| Group Lead | Architecture, hybrid engine, API |
| Data Engineer | Data collection, preprocessing |
| CF Specialist | ALS model, user-item matrix |
| CB Specialist | TF-IDF, cosine similarity |
| NLP Specialist | VADER sentiment, aggregation |
| UI Developer | Streamlit frontend |

---

## 📚 References

- Yelp Open Dataset — https://www.yelp.com/dataset
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text
- Lim, K.H. et al. (2015). Tour Recommendation and Trip Planning Using Location-Based Social Media

---

## 📄 License

This project was developed for academic purposes as part of the Smart Information Systems and Engineering course.
