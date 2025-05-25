import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import argparse
import os

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

def compute_hrr_curve(A, t_alpha, Y, dt=1):
    t = []
    HRR = []
    t_grow = np.sqrt(Y) * t_alpha
    t_now = 0
    area_grow = 0

    while t_now <= t_grow:
        hrr_now = min((t_now / t_alpha) ** 2, Y)
        HRR.append(hrr_now)
        t.append(t_now)
        area_grow += hrr_now * dt
        t_now += dt

    area_target = 0.7 * A
    area_steady = 0
    while area_grow + area_steady < area_target:
        HRR.append(Y)
        t.append(t_now)
        area_steady += Y * dt
        t_now += dt

    area_decay = A - (area_grow + area_steady)
    t_decay = 2 * area_decay / Y
    t_end = t_now + t_decay

    while t_now < t_end:
        hrr_now = Y * (1 - (t_now - t_grow - (area_steady / Y)) / t_decay)
        HRR.append(max(0, hrr_now))
        t.append(t_now)
        t_now += dt

    while t_now <= 18000:
        HRR.append(0)
        t.append(t_now)
        t_now += dt

    return np.array(t), np.array(HRR)

def main():
    parser = argparse.ArgumentParser(description="Predict fire temperature curve from HRR input")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--floor', type=float, required=True, help='Floor area (m²)')
    parser.add_argument('--total', type=float, required=True, help='Total fire area (m²)')
    args = parser.parse_args()

    # Load input and model
    data = pd.read_csv(args.input, header=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(os.path.dirname(__file__), "stacked_lstm_model.pth")
    model = StackedSeq2SeqLSTM(1, 64, 64, 32, 32, 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    hrr_curves = []
    for _, row in data.iterrows():
        A = row[0] * args.floor
        t_alpha = row[1]
        Y = row[2] * args.total * 1.4
        t, HRR = compute_hrr_curve(A, t_alpha, Y)
        mask = (t % 60 == 0)
        hrr_curves.append(HRR[mask])

    X_data = np.array(hrr_curves)[:, :, np.newaxis]
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    # Save output with first row as time
    time_minutes = np.arange(predictions.shape[1])
    df = pd.DataFrame(predictions)
    df.columns = [str(t) for t in time_minutes]
    df.T.to_csv("Predicted_fire_curve.csv", header=False)

    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(df.shape[0]):
        plt.plot(time_minutes, predictions[i], alpha=0.6)
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature")
    plt.title("Predicted Fire Curves")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
