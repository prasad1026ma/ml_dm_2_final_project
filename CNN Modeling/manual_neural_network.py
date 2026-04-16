import numpy as np


def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def binary_crossentropy(y_true, y_pred):
    """Binary crossentropy loss"""
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=1.0):
    """
    Binary crossentropy with class weighting.

    pos_weight > 1 means we penalize false negatives more.
    This forces the model to learn class 1 even if it's rare.
    """
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(pos_weight * y_true * np.log(y_pred) +
             (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

class ManualCNN:
    """
    CNN for sequence classification

    Input shape: (batch_size, seq_length) where seq_length = timesteps * num_features
    Architecture:
    Conv -> ReLU -> MaxPool -> Dense -> Sigmoid
    """
    def __init__(self, input_size, num_filters=16, l2_lambda=0.0005):
        """
        Args:
            input_size: length of input sequence (timesteps * features)
            num_filters: number of convolutional filters
            l2_lambda: L2 regularization strength
        """
        self.input_size = input_size
        self.num_filters = num_filters
        self.l2_lambda = l2_lambda
        self.kernel_size = 3

        # Conv layer
        self.conv_kernel = np.random.randn(self.kernel_size, num_filters) * np.sqrt(2.0 / self.kernel_size)
        self.conv_bias = np.zeros(num_filters)
        # Layer size computation for outputs
        self.conv_len = input_size - self.kernel_size + 1
        self.pool_len = self.conv_len // 2
        self.dense_input_size = self.pool_len * num_filters
        # Dense layer paramters
        self.dense_w = np.random.randn(self.dense_input_size, 1) * np.sqrt(2.0 / self.dense_input_size)
        self.dense_b = 0.0
        # Learning rate
        self.lr = 0.01

    def forward(self, X):
        """
        Forward pass through the network

        :param X: input with shape (batch_size, input_size)
        :return:
            self.probs: predicted probabilities
        """
        batch_size = X.shape[0]
        self.X = X

        # Compute outputs for the convolutional layers based on a 3x3 kernel
        self.conv_out = np.zeros((batch_size, self.conv_len, self.num_filters))

        for b in range(batch_size):
            for i in range(self.conv_len):
                # Extract window of size kernel_size
                window = X[b, i:i + self.kernel_size]  # shape: (3,)

                # Apply all filters
                for f in range(self.num_filters):
                    self.conv_out[b, i, f] = np.dot(window, self.conv_kernel[:, f]) + self.conv_bias[f]

        # ReLU activation function
        self.relu_out = relu(self.conv_out)

        # Max Pooling with a stride of 2
        self.pool_out = np.zeros((batch_size, self.pool_len, self.num_filters))
        self.pool_mask = {}

        for b in range(batch_size):
            for f in range(self.num_filters):
                for p in range(self.pool_len):
                    start_idx = p * 2
                    end_idx = min(start_idx + 2, self.conv_len)

                    window = self.relu_out[b, start_idx:end_idx, f]
                    max_idx_local = np.argmax(window)
                    max_idx_global = start_idx + max_idx_local

                    self.pool_out[b, p, f] = window[max_idx_local]
                    self.pool_mask[(b, p, f)] = max_idx_global

        # Flatten Data
        self.flat_out = self.pool_out.reshape(batch_size, -1)

        # Dense Layer
        self.logits = self.flat_out @ self.dense_w + self.dense_b
        self.probs = sigmoid(self.logits)

        return self.probs

    def backward(self, y_true):
        """
        Backward pass - backpropagation through entire network
        """
        batch_size = y_true.shape[0]

        # Output Gradient
        dL_dlogits = (self.probs - y_true.reshape(-1, 1)) / batch_size

        # Dense layer gradidents
        dL_dW_dense = self.flat_out.T @ dL_dlogits
        dL_db_dense = np.sum(dL_dlogits, axis=0)

        # Add L2 regularization
        dL_dW_dense += self.l2_lambda * self.dense_w

        # Gradient flowing back to pool (Backprop into the pooled layer)
        dL_dflat = dL_dlogits @ self.dense_w.T
        dL_dpool = dL_dflat.reshape(self.pool_out.shape)

        # Maxpool Backprop
        dL_drelu = np.zeros_like(self.relu_out)
        for b in range(batch_size):
            for f in range(self.num_filters):
                for p in range(self.pool_len):
                    max_idx = self.pool_mask[(b, p, f)]
                    dL_drelu[b, max_idx, f] += dL_dpool[b, p, f]

        # ReLU backprop
        dL_dconv = dL_drelu * relu_derivative(self.conv_out)

        # Convolutional layer backprop
        dL_dW_conv = np.zeros_like(self.conv_kernel)
        dL_db_conv = np.zeros_like(self.conv_bias)

        for b in range(batch_size):
            for i in range(self.conv_len):
                window = self.X[b, i:i + self.kernel_size]
                for f in range(self.num_filters):
                    grad = dL_dconv[b, i, f]
                    dL_dW_conv[:, f] += window * grad
                    dL_db_conv[f] += grad

        # Add L2 regularization for convolutional backprop
        dL_dW_conv += self.l2_lambda * self.conv_kernel

        # Update the overall weights via Gradient Descent
        self.dense_w -= self.lr * dL_dW_dense
        self.dense_b -= self.lr * dL_db_dense
        self.conv_kernel -= self.lr * dL_dW_conv / batch_size
        self.conv_bias -= self.lr * dL_db_conv / batch_size

    def train_step(self, X, y):
        """Run one training step: forward + backward"""
        pred = self.forward(X)
        loss = binary_crossentropy(y, pred)
        self.backward(y)
        return loss, pred.flatten()

    def predict(self, X):
        """Return predictions (probabilities)"""
        return self.forward(X).flatten()