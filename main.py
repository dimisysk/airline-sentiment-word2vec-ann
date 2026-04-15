import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model import ANNClassifier
from src.utils import (
    load_dataset,
    inspect_dataset,
    select_columns,
    preprocess_dataset,
    inspect_preprocessed_data,
    plot_top_words,
    train_word2vec,
    inspect_similar_words,
    create_feature_matrix,
    encode_labels,
    split_data,
    to_tensors,
    create_dataloaders,
    train_model,
    plot_training_history,
    evaluate_model,
    print_classification_metrics,
    plot_confusion_matrix,

)


def main():
    file_path = "data/Tweets.csv"
    target_column = "airline_sentiment"
    text_column = "text"

    df = load_dataset(file_path)
    inspect_dataset(df, target_column)

    df = select_columns(df)
    print("\nAfter column selection:")
    print(df.head())

    df = preprocess_dataset(df, text_column)
    inspect_preprocessed_data(df, text_column, top_n=30)
    plot_top_words(df, top_n=30)

    tokenized_sentences = df["cleaned_tokens"].tolist()

    word2vec_model = train_word2vec(
        tokenized_sentences=tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    print("\nWord2Vec model trained successfully.")
    print("Vocabulary size:", len(word2vec_model.wv))

    test_words = ["cancelled", "service", "help", "delayed"]
    inspect_similar_words(word2vec_model, test_words, top_n=5)

    X = create_feature_matrix(df["cleaned_tokens"].tolist(), word2vec_model)
    y = df["airline_sentiment"].copy()

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nFirst tweet vector (first 10 values):")
    print(X[0][:10])

    y_encoded = encode_labels(y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)

    X_train, X_val, X_test, y_train, y_val, y_test = to_tensors(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ANNClassifier(
        input_dim=X_train.shape[1],
        hidden_dim1=128,
        hidden_dim2=64,
        output_dim=3,
        dropout_p1=0.2,
        dropout_p2=0.2
    ).to(device)

    print(model)

    class_counts = np.bincount(y_train.cpu().numpy())
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        patience=5
    )

    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    y_true, y_pred = evaluate_model(model, test_loader, device)
    print_classification_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()