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


##  Dataset

The dataset used is the **US Airline Sentiment** dataset from Kaggle.

- ~14,000 tweets  
- Includes:
  - Tweet text  
  - Airline name  
  - Sentiment label (target variable)  

This dataset is commonly used for benchmarking sentiment classification models on short and noisy text.

---

## ️ Project Structure

```

project-1/
│
├── data/
│ └── Tweets.csv
│
├── src/
│ ├── model.py # ANN model definition
│ ├── utils.py # preprocessing & helper functions
│
├── main.py # main pipeline
│
├── requirements.txt
└── README.md


```




## 🔧 Installation


```bash
pip install -r requirements.txt
```

Also install NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```


---

##  How to Run

Run the project from the `src` directory:

```bash
python main.py
```

---

##  Embeddings

Word2Vec is used to convert text into dense vector representations.

**Model configuration:**
- `vector_size = 100`
- `window = 5`
- `min_count = 2`
- `architecture = CBOW`

Each tweet is represented as the average of its word embeddings.

---

## 🤖 Model

The classifier is a feed-forward neural network consisting of:

- Input layer (tweet embedding vector)
- Hidden layer 1: `128` units with ReLU activation
- Dropout layer: `p = 0.2`
- Hidden layer 2: `64` units with ReLU activation
- Dropout layer: `p = 0.2`
- Output layer: `3` units for sentiment classification

**Training setup:**
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `0.0001`
- Batch size: `16`
- Early stopping with `patience = 5`
- Class weighting applied to handle class imbalance 

---

## 📊 Results

- Accuracy: ~0.63–0.65  
- Weighted F1-score: ~0.65  

The model achieves reasonable performance given:

- the simplicity of the architecture  
- the short and noisy nature of tweets  
- the difficulty in distinguishing neutral sentiment  



##  Results

- Accuracy: ~0.63–0.65  
- Weighted F1-score: ~0.65  

The model achieves balanced performance across all sentiment classes after applying class weighting.


---

##  Visualizations

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

##  Limitations

- Word2Vec averaging ignores word order
- Neutral sentiment is harder to classify due to ambiguity