import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import os

# Dataset: Generate (a, b, c, d) → (r1, r2, r3) pairs
def generate_data(n_samples=10000):
    a = np.random.uniform(0.5, 2.0, size=n_samples)
    b = np.random.uniform(-5, 5, size=n_samples)
    c = np.random.uniform(-5, 5, size=n_samples)
    d = np.random.uniform(-5, 5, size=n_samples)

    coeffs = np.stack([a, b, c, d], axis=1)

    roots = []
    for i in range(n_samples):
        r = np.roots(coeffs[i])
        r = sorted(r, key=lambda x: (x.real, x.imag))  # consistent ordering
        roots.append([r[0].real, r[0].imag, r[1].real, r[1].imag, r[2].real, r[2].imag])
        
    roots = np.array(roots)
    return coeffs.astype(np.float32), roots.astype(np.float32)

# Deep ANN
class RootFinderANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # [r1_re, r1_im, r2_re, r2_im, r3_re, r3_im]
        )

    def forward(self, x):
        return self.net(x)

# Training
def train_model(model, X_train, y_train, epochs=200, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        inputs = torch.from_numpy(X_train)
        targets = torch.from_numpy(y_train)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return losses

# Evaluation
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test)
        targets = torch.from_numpy(y_test)
        preds = model(inputs).numpy()

    errors = np.linalg.norm(preds - y_test, axis=1)  # Euclidean error per sample
    print(f"\nMean root difference (L2 norm): {np.mean(errors):.4e}")
    print(f"Max root difference (L2 norm):  {np.max(errors):.4e}")
    return preds, errors

# Plot training loss
def save_loss_plot(losses, filename="roots_loss.png"):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Visualize closeness of predicted and actual roots
def plot_test_predictions(preds, trues, n=5, filename="roots_preds.png"):
    plt.figure(figsize=(15, 3 * n))
    for i in range(n):
        true_roots = trues[i].reshape(3, 2)
        pred_roots = preds[i].reshape(3, 2)

        plt.subplot(n, 1, i+1)
        plt.scatter(true_roots[:, 0], true_roots[:, 1], color='blue', label='True', marker='o')
        plt.scatter(pred_roots[:, 0], pred_roots[:, 1], color='red', label='Predicted', marker='x')

        # Draw lines connecting true and predicted roots
        for j in range(3):
            plt.plot(
                [true_roots[j, 0], pred_roots[j, 0]],
                [true_roots[j, 1], pred_roots[j, 1]],
                'k--', linewidth=0.8, alpha=0.5
            )

        plt.title(f"Sample {i+1} – Root Prediction vs Ground Truth")
        plt.xlabel("Real")
        plt.ylabel("Imag")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Main pipeline
if __name__ == "__main__":
    os.makedirs(".", exist_ok=True)

    # Generate data
    X, y = generate_data(10000)
    X_train, y_train = X[:8000], y[:8000]
    X_test, y_test = X[8000:], y[8000:]

    # Train model
    model = RootFinderANN()
    losses = train_model(model, X_train, y_train, epochs=200)

    # Save loss plot
    save_loss_plot(losses, filename="roots_loss.png")

    # Evaluate model
    preds, errors = evaluate(model, X_test, y_test)

    # Save predicted vs true root plots
    plot_test_predictions(preds, y_test, n=5, filename="roots_preds.png")

    # Print a few predictions
    for i in range(3):
        print(f"\nSample {i+1}")
        print(f"Coefficients (a,b,c,d): {X_test[i]}")
        for j in range(3):
            tr = y_test[i][2*j:2*j+2]
            pr = preds[i][2*j:2*j+2]
            print(f"  Root {j+1} True: {tr[0]:+.3f} {tr[1]:+.3f}j, Pred: {pr[0]:+.3f} {pr[1]:+.3f}j, Δ = {np.linalg.norm(tr - pr):.3e}")
