
# Sentiment Analysis GUI

A Python-based GUI application that predicts the sentiment of customer reviews as **Positive, Neutral, or Negative**, and displays a corresponding emoji. Built using **Tkinter**, **scikit-learn**, and **Pandas**.

---

## 📂 Project Structure

```
project_root/
│
├── data/
│   └── customer_reviews.csv          # Sample reviews dataset
│
├── model/
│   ├── sentiment_analysis_model.pkl  # Trained Naive Bayes model
│   ├── vectorizer.pkl                # TF-IDF Vectorizer
│   └── sentiment_analysis.ipynb      # Model training notebook
│
├── gui/
│   └── sentiment_gui.py              # Tkinter GUI script
│
├── assets/
│   ├── happy.png                     # Emoji for Positive sentiment
│   ├── neutral.png                   # Emoji for Neutral sentiment
│   └── angry.png                     # Emoji for Negative sentiment
│
├── requirements.txt                  # Required Python packages
└── README.md
```

---

## 🛠️ Requirements

- Python 3.8+
- Virtual environment recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

**Key libraries used:**

- `tkinter` – GUI
- `pandas` – Data handling
- `Pillow` – Image handling
- `scikit-learn` – Machine Learning
- `nltk` – Text preprocessing

---

## ⚡ Setup

1. Clone/download the project.
2. Activate your virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure your folder structure is intact:
- `data/customer_reviews.csv`
- `model/sentiment_analysis_model.pkl` and `vectorizer.pkl`
- `assets/` contains emoji images

---

## ▶️ Running the GUI

**Windows:**
```bash
python gui/sentiment_gui.py
```

**Mac/Linux:**
```bash
python3 gui/sentiment_gui.py
```

- The GUI window will open.
- It displays the first review and its corresponding emoji.
- Click **Next Review** to cycle through all reviews.

---

## 📝 Features

- Preprocesses text using **NLTK** (stopwords removal, lemmatization, POS tagging).
- Predicts **3 sentiment classes**: Positive, Neutral, Negative.
- Displays **emoji feedback**:
  - 😊 Positive
  - 😐 Neutral
  - 😡 Negative

---

## 📊 Professional Insights & Results

### 1. Dataset Overview
The dataset contains a total of **1,000 customer reviews** distributed across three sentiment categories:

| Sentiment | Count |
|-----------|-------|
| Positive  | 350   |
| Neutral   | 347   |
| Negative  | 303   |

**Observations:**
- Well-balanced dataset across three classes.
- Positive and Neutral reviews slightly outnumber Negative reviews.
- Balanced data ensures robust model learning without bias.

### 2. Model Performance
Evaluated on a **test set of 200 reviews**:

**Overall Accuracy:** 100%  

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 1.00      | 1.00   | 1.00     | 61      |
| Neutral   | 1.00      | 1.00   | 1.00     | 69      |
| Positive  | 1.00      | 1.00   | 1.00     | 70      |

**Macro Avg:** 1.00  | **Weighted Avg:** 1.00  

**Observations:**
- Perfect classification on the test set.
- No misclassifications; reliable predictions.

### 3. Insights from Sentiment Distribution
- Positive Reviews (350): Generally satisfied customers.
- Neutral Reviews (347): Mixed or indifferent feedback.
- Negative Reviews (303): Areas for improvement.

**Actionable Insight:** Focus on analyzing Negative reviews to address recurring issues.



### 4. Recommendations
1. Monitor sentiment trends over time.
2. Deploy in real-time for new customer feedback.
3. Analyze Negative reviews to identify actionable improvements.
4. Expand dataset for more diverse reviews to improve model robustness.

**Conclusion:** The sentiment analysis model provides accurate insights into customer opinions, enabling data-driven decisions and improved customer satisfaction.

---

## 📌 Notes

- Ensure NLTK data is installed before running the GUI:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```
- Check file paths in `sentiment_gui.py`.

---


