import numpy as np
def relu(x):
    return np.maximum(0, x)
def relu_grad(x):
    return (x > 0).astype(float)
def tanh_grad(x):
    return 1 - np.tanh(x) ** 2
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class ManualNN:
    def __init__(self, input_size=12, learning_rate=0.01):
        self.lr = learning_rate

        self.W1 = np.random.randn(input_size, 250) * 0.01
        self.b1 = np.zeros((1, 250))

        self.W2 = np.random.randn(250, 100) * 0.01
        self.b2 = np.zeros((1, 100))

        self.W3 = np.random.randn(100, 1) * 0.01
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        """Forward pass through network"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.tanh(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)

        return self.a3

    def backward(self, X, y, y_pred, batch_size):
        dz3 = (y_pred - y) / batch_size
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * tanh_grad(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_grad(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=20, batch_size=32, validation_split=0.2):
        val_size = int(len(X) * validation_split)
        X_val, X_train_split = X[-val_size:], X[:-val_size]
        y_val, y_train_split = y[-val_size:], y[:-val_size]

        num_batches = len(X_train_split) // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train_split))
            X_shuffled = X_train_split[indices]
            y_shuffled = y_train_split[indices]

            epoch_loss = 0

            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                loss = binary_crossentropy(y_batch, y_pred)
                epoch_loss += loss

                self.backward(X_batch, y_batch, y_pred, len(X_batch))

            y_val_pred = self.forward(X_val)
            val_loss = binary_crossentropy(y_val, y_val_pred)
            val_acc = np.mean((y_val_pred > 0.5).astype(int) == y_val)

            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / num_batches}, val_loss: {val_loss}, val_acc: {val_acc}")

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = binary_crossentropy(y, y_pred)
        accuracy = np.mean((y_pred > 0.5).astype(int) == y)
        return loss, accuracy

    def predict(self, X):
        return self.forward(X)
