from sklearn.neural_network import MLPClassifier


def train_mlp(X, y, random_state=42):
    """Train a simple MLP classifier for tabular data."""
    clf = MLPClassifier(
        hidden_layer_sizes=(
            32,
            16),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=random_state)
    clf.fit(X, y)
    return clf
