import pandas as pd
import numpy as np
import torch
import random
from torch import nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Read data
X_df = pd.read_csv("LSTM_HRR_Train.csv", header=None)
Y_df = pd.read_csv("LSTM_fire_Train.csv", header=None)

# Data preprocessing
X_data = X_df.iloc[:, 1::2].values.T[:, :, np.newaxis]  # Shape: (100, 300, 1)
Y_data = Y_df.iloc[:, 1::2].values.T[:, :, np.newaxis]  # Shape: (100, 300, 1)

# Use the first 20 samples as validation set; remaining 80 for training/CV
X_train_test_data = X_data[20:]
Y_train_test_data = Y_data[20:]
X_val_data = X_data[:20]
Y_val_data = Y_data[:20]

# Convert to tensors
X_tensor = torch.tensor(X_train_test_data, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_train_test_data, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_data, dtype=torch.float32).to(device)
Y_val_tensor = torch.tensor(Y_val_data, dtype=torch.float32).to(device)

# Define stacked LSTM model
class StackedSeq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, fc1_size, fc2_size, output_size):
        super(StackedSeq2SeqLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)

    def forward(self, x):
        h1, _ = self.lstm1(x)
        h2, _ = self.lstm2(h1)
        h3 = torch.relu(self.fc1(h2))
        out = torch.relu(self.fc2(h3))
        out = self.fc3(out)
        return out

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"\nFold {fold + 1}")
    X_train_fold = X_tensor[train_idx]
    Y_train_fold = Y_tensor[train_idx]
    X_val_fold = X_tensor[val_idx]
    Y_val_fold = Y_tensor[val_idx]

    model = StackedSeq2SeqLSTM(1, 64, 64, 32, 32, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(3000):
        model.train()
        output = model(X_train_fold)
        loss = criterion(output, Y_train_fold)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_fold)
                val_loss = criterion(val_output, Y_val_fold).item()
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Fold Val Loss: {val_loss:.6f}")

    fold_losses.append(val_loss)

print(f"\n✅ K-Fold Mean Loss: {np.mean(fold_losses):.6f}, Std: {np.std(fold_losses):.6f}")

# Train final model on all training data for validation prediction
final_model = StackedSeq2SeqLSTM(1, 64, 64, 32, 32, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.002)

for epoch in range(3000):
    final_model.train()
    output = final_model(X_tensor)
    loss = criterion(output, Y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict on validation set
final_model.eval()
with torch.no_grad():
    val_predictions = final_model(X_val_tensor)
    val_loss = criterion(val_predictions, Y_val_tensor)
    print(f"Validation Loss (MSE): {val_loss.item():.6f}")
    val_predictions = val_predictions.cpu().numpy()
    val_truth = Y_val_tensor.cpu().numpy()

# Save predictions and ground truth
pd.DataFrame(val_predictions.squeeze()).to_csv("validation_predictions.csv", index=False)
pd.DataFrame(val_truth.squeeze()).to_csv("validation_truth.csv", index=False)
print("Validation predictions saved to 'validation_predictions.csv'")

# Plot predictions vs ground truth
plt.figure(figsize=(12, 8))
for i in range(20):
    plt.plot(val_truth[i].squeeze(), color='black', alpha=0.3, linewidth=2, linestyle='--')
    plt.plot(val_predictions[i].squeeze(), color='red', alpha=0.3, linewidth=2)

plt.xlabel("Time step")
plt.ylabel("Output")
plt.title("Prediction (red) vs Actual (black) for 20 validation samples")
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute normalized error and plot PDF
normalized_error = (val_predictions - val_truth) / np.max(val_truth, axis=1, keepdims=True)
normalized_error_flat = normalized_error.flatten()

plt.figure(figsize=(10, 6))
sns.kdeplot(normalized_error_flat, bw_adjust=1, fill=True, color="purple", alpha=0.5, linewidth=2)
plt.xlabel("Normalized Error")
plt.ylabel("Probability Density")
plt.title("PDF of Normalized Error on Validation Set")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save PDF data
kde = gaussian_kde(normalized_error_flat, bw_method=1)
x_vals = np.linspace(np.min(normalized_error_flat), np.max(normalized_error_flat), 1000)
pdf_vals = kde(x_vals)
pdf_df = pd.DataFrame({'Normalized_Error': x_vals, 'PDF_Value': pdf_vals})
pdf_df.to_csv("pdf_normalized_error_validation.csv", index=False)
print("PDF data saved to 'pdf_normalized_error_validation.csv'")

# Save model
torch.save(final_model.state_dict(), "stacked_lstm_model.pth")
print("Model saved to 'stacked_lstm_model.pth'")
