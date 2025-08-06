import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# --- Data ---
def generate_data(n_samples=20000):
    a = np.random.uniform(0.5, 2.0, size=n_samples)
    b = np.random.uniform(-5, 5, size=n_samples)
    c = np.random.uniform(-5, 5, size=n_samples)
    d = np.random.uniform(-5, 5, size=n_samples)

    coeffs = np.stack([a, b, c, d], axis=1)
    roots = []
    for i in range(n_samples):
        norm_coeffs = coeffs[i] / coeffs[i][0]  # make monic: a = 1
        r = np.roots(norm_coeffs)
        r = sorted(r, key=lambda x: (x.real, x.imag))  # fix order
        roots.append([r[0].real, r[0].imag, r[1].real, r[1].imag, r[2].real, r[2].imag])
    return coeffs.astype(np.float64), np.array(roots).astype(np.float64)

# --- Activation ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.act = Swish()
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        identity = x
        out = self.act(self.linear1(x))
        out = self.linear2(out)
        out += identity
        out = self.norm(out)
        return out

# --- Model ---
class DeepRootNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 128)
        self.body = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.output = nn.Linear(128, 6)

    def forward(self, x):
        x = Swish()(self.input(x))
        x = self.body(x)
        return self.output(x)

# --- Training ---
def train_model(model, X_train, y_train, epochs=500, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
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
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")
    return losses

# --- Eval ---
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test)
        targets = torch.from_numpy(y_test)
        preds = model(inputs).numpy()
    errors = np.linalg.norm(preds - y_test, axis=1)
    print(f"\nMean L2 Error: {np.mean(errors):.6e}")
    print(f"Max  L2 Error: {np.max(errors):.6e}")
    return preds

# --- Plotting ---
def save_loss_plot(losses, filename="complex_nn_roots_loss.png"):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_predictions(preds, trues, n=5, filename="complex_nn_roots_preds.png"):
    plt.figure(figsize=(15, 3 * n))
    for i in range(n):
        true_roots = trues[i].reshape(3, 2)
        pred_roots = preds[i].reshape(3, 2)

        plt.subplot(n, 1, i+1)
        plt.scatter(true_roots[:, 0], true_roots[:, 1], color='blue', label='True', marker='o')
        plt.scatter(pred_roots[:, 0], pred_roots[:, 1], color='red', label='Predicted', marker='x')

        for j in range(3):
            plt.plot(
                [true_roots[j, 0], pred_roots[j, 0]],
                [true_roots[j, 1], pred_roots[j, 1]],
                'k--', linewidth=0.8, alpha=0.5
            )

        plt.title(f"Sample {i+1} – Root Prediction vs Ground Truth")
        plt.xlabel("Real")
        plt.ylabel("Imag")
        plt.axis('equal')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main ---
if __name__ == "__main__":
    X, y = generate_data(20000)
    X_train, y_train = X[:16000], y[:16000]
    X_test, y_test = X[16000:], y[16000:]

    model = DeepRootNet()
    losses = train_model(model, X_train, y_train, epochs=5000, lr=5e-5)
    save_loss_plot(losses)

    preds = evaluate(model, X_test, y_test)
    plot_predictions(preds, y_test, n=5)

    # Print a few cases
    for i in range(3):
        print(f"\nSample {i+1}")
        print(f"Coefficients (a,b,c,d): {X_test[i]}")
        for j in range(3):
            t = y_test[i][2*j:2*j+2]
            p = preds[i][2*j:2*j+2]
            print(f"  Root {j+1} True: {t[0]:+.5f} {t[1]:+.5f}j, Pred: {p[0]:+.5f} {p[1]:+.5f}j, Δ = {np.linalg.norm(t - p):.2e}")
        # --- Export to ONNX ---
    onnx_filename = "DeepRootNet_cubic.onnx"
    sample_input = torch.randn(1, 4, dtype=torch.float64)
    # ONNX export requires float32 → cast temporarily
    model_float32 = model.float()
    sample_input = sample_input.float()
    torch.onnx.export(
        model_float32,
        sample_input,
        onnx_filename,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['poly_coeffs'],       # Name the input
        output_names=['roots_output'],     # Name the output
        dynamic_axes={
            'poly_coeffs': {0: 'batch_size'},
            'roots_output': {0: 'batch_size'}
        }
    )
    print(f"\n✅ Model exported to ONNX format: {onnx_filename}")
