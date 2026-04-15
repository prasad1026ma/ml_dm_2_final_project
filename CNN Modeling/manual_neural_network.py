import numpy as np


def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)


def sigmoid(x):
    """Sigmoid: 1 / (1 + e^-x)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative of sigmoid: s(x) * (1 - s(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def binary_crossentropy(y_true, y_pred):
    """Binary crossentropy loss"""
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


class ManualCNN:
    def __init__(self, input_size, num_features):
        """
        Conv1 → Conv2 → Dense pipeline
        """
        self.input_size = input_size
        self.num_features = num_features

        # conv layer 1
        self.kernel1 = np.random.randn(3, num_features) * 0.01
        self.bias1 = 0.0
        self.conv1_size = input_size - 3 + 1
        self.pool1_size = self.conv1_size // 2

        # conv layer 2
        self.kernel2 = np.random.randn(3, 1) * 0.01
        self.bias2 = 0.0
        self.conv2_size = self.pool1_size - 3 + 1
        self.pool2_size = max(1, self.conv2_size // 2)

        # dense layer
        self.dense_w = np.random.randn(self.pool2_size, 1) * 0.01
        self.dense_b = 0.0

        self.lr = 0.01

    def forward(self, X):
        """Forward pass"""
        self.X = X

        # Conv1 + ReLU + MaxPool
        self.conv1 = self.convolve(X, self.kernel1, self.bias1)
        self.pool1 = self.maxpool(self.conv1, pool_size=2)

        # Conv2 + ReLU + MaxPool
        pool1_expanded = self.pool1.reshape(self.pool1.shape[0], -1, 1)
        self.conv2 = self.convolve(pool1_expanded, self.kernel2, self.bias2)
        self.pool2 = self.maxpool(self.conv2, pool_size=2)

        # Dense
        self.logits = self.pool2 @ self.dense_w + self.dense_b
        self.predictions = sigmoid(self.logits)

        return self.predictions

    def convolve(self, X, kernel, bias):
        batch_size = X.shape[0]
        conv_size = X.shape[1] - kernel.shape[0] + 1
        out = np.zeros((batch_size, conv_size))

        for b in range(batch_size):
            for t in range(conv_size):
                window = X[b, t:t + kernel.shape[0], :]
                out[b, t] = relu(np.sum(window * kernel) + bias)

        return out

    def maxpool(self, X, pool_size=2):

        batch_size = X.shape[0]
        pool_out_size = X.shape[1] // pool_size
        out = np.zeros((batch_size, pool_out_size))
        self.pool_mask = np.zeros_like(X)

        for b in range(batch_size):
            for p in range(pool_out_size):
                start = p * pool_size
                end = start + pool_size
                window = X[b, start:end]
                max_idx = np.argmax(window)
                out[b, p] = window[max_idx]
                self.pool_mask[b, start + max_idx] = 1

        return out

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        # output gradient
        d_logits = (self.predictions - y_true.reshape(-1, 1)) / batch_size

        # backprop through dense
        d_dense_w = self.pool2.T @ d_logits
        d_dense_b = np.sum(d_logits)

        self.dense_w -= self.lr * d_dense_w
        self.dense_b -= self.lr * d_dense_b

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = binary_crossentropy(y, y_pred)
        self.backward(y)
        return loss, y_pred.flatten()

    def predict(self, X):
        return self.forward(X).flatten()
