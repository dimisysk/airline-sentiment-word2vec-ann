import copy
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def inspect_dataset(df: pd.DataFrame, target_column: str) -> None:
    print("Shape of dataset:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nUnique values in '{target_column}':")
    print(df[target_column].unique())

    print(f"\nClass distribution for '{target_column}':")
    print(df[target_column].value_counts())


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[["text", "airline_sentiment"]].copy()


def clean_and_tokenize_text(text: str, stop_words: set) -> list:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp;", "and", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    tokens = re.findall(r"[a-zA-Z]+", text)
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


def preprocess_dataset(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    stop_words = set(stopwords.words("english"))
    custom_stopwords = {"flight", "flights", "plane", "airline", "get"}
    final_stopwords = stop_words.union(custom_stopwords)

    df = df.copy()
    df["cleaned_tokens"] = df[text_column].apply(
        lambda x: clean_and_tokenize_text(x, final_stopwords)
    )
    return df


def inspect_preprocessed_data(df: pd.DataFrame, text_column: str, top_n: int = 30) -> None:
    print("\nSample original and cleaned tweets:")
    print(df[[text_column, "cleaned_tokens"]].head())

    empty_tweets = (df["cleaned_tokens"].apply(len) == 0).sum()
    print(f"\nNumber of empty tweets after preprocessing: {empty_tweets}")

    all_tokens = [token for tokens in df["cleaned_tokens"] for token in tokens]
    most_common_words = Counter(all_tokens).most_common(top_n)

    print(f"\nTop {top_n} most frequent words:")
    for word, count in most_common_words:
        print(f"{word}: {count}")


def plot_top_words(df: pd.DataFrame, top_n: int = 30) -> None:
    all_tokens = [token for tokens in df["cleaned_tokens"] for token in tokens]
    word_counts = Counter(all_tokens).most_common(top_n)

    words = [word for word, _ in word_counts]
    counts = [count for _, count in word_counts]

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.title(f"Top {top_n} Most Frequent Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def train_word2vec(tokenized_sentences: list,
                   vector_size: int = 100,
                   window: int = 5,
                   min_count: int = 2,
                   workers: int = 4) -> Word2Vec:
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model


def inspect_similar_words(model: Word2Vec, test_words: list, top_n: int = 5) -> None:
    print("\nMost similar words:")

    for word in test_words:
        print(f"\nWord: '{word}'")

        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=top_n)
            for similar_word, score in similar_words:
                print(f"{similar_word}: {score:.4f}")
        else:
            print("Word not in vocabulary.")


def average_word_vectors(tokens: list, model: Word2Vec) -> np.ndarray:
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]

    if len(valid_vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(valid_vectors, axis=0)


def create_feature_matrix(tokenized_sentences: list, model: Word2Vec) -> np.ndarray:
    return np.array([average_word_vectors(tokens, model) for tokens in tokenized_sentences])


def encode_labels(y: pd.Series) -> np.ndarray:
    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    return y.map(label_map).values


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=200, patience=5, min_delta=0.0):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_acc += calculate_accuracy(outputs, labels)

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = running_train_acc / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                running_val_acc += calculate_accuracy(outputs, labels)

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = running_val_acc / len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def print_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nTest Set Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["negative", "neutral", "positive"]
    ))


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.pdf"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies,
                          loss_path="training_validation_loss.pdf",
                          acc_path="training_validation_accuracy.pdf"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path, format="pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path, format="pdf", bbox_inches="tight")
    plt.show()