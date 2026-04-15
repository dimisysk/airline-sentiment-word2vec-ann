# US Airline Sentiment Classification (Word2Vec + ANN)

This project implements a complete NLP pipeline for sentiment classification on airline tweets using Word2Vec embeddings and a feed-forward neural network (ANN).

---

##  Overview

The goal is to classify tweets into three sentiment categories:
- Negative
- Neutral
- Positive

The pipeline includes:
- Text preprocessing
- Word2Vec embeddings
- Feature extraction
- ANN model training
- Evaluation with multiple metrics

---

## ⚙️ Project Structure

```
project-1/
│
├── data/
│   └── Tweets.csv
│
├── src/
│   ├── model.py        # ANN model
│   ├── utils.py        # All helper functions
│   └── main.py         # Main pipeline
│
├── requirements.txt
└── README.md
```



---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements.txt
```

Also install NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```


---

## ▶️ How to Run

Run the project from the `src` directory:

```bash
python main.py
```

---

## 📊 Results

- Accuracy: ~0.63–0.65  
- Weighted F1-score: ~0.65  

The model achieves balanced performance across all sentiment classes after applying class weighting.


---

## 📈 Visualizations

The following plots are generated during training:

- Training vs Validation Loss  
- Training vs Validation Accuracy  
- Confusion Matrix  

You can find them in the project directory:

- `training_validation_loss.pdf`
- `training_validation_accuracy.pdf`
- `confusion_matrix.pdf`

These plots provide insight into model performance, convergence behavior, and classification errors across sentiment classes.

---

## ⚠️ Limitations

- Word2Vec averaging ignores word order
- Neutral sentiment is harder to classify due to ambiguity