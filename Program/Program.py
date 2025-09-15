import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
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
    t_now = 0.0
    area_grow = 0.0

    # growth
    while t_now <= t_grow:
        hrr_now = min((t_now / t_alpha) ** 2, Y) if t_alpha > 0 else 0.0
        HRR.append(hrr_now)
        t.append(t_now)
        area_grow += hrr_now * dt
        t_now += dt

    # steady (70% area)
    area_target = 0.7 * A
    area_steady = 0.0
    while area_grow + area_steady < area_target:
        HRR.append(Y)
        t.append(t_now)
        area_steady += Y * dt
        t_now += dt

    # decay
    area_decay = A - (area_grow + area_steady)
    t_decay = 2 * area_decay / Y if Y > 0 else 0.0
    t_end = t_now + t_decay

    while t_now < t_end:
        frac = (t_now - (t_grow + area_steady / Y)) / t_decay if t_decay > 0 else 1.0
        hrr_now = Y * (1 - frac)
        HRR.append(max(0.0, hrr_now))
        t.append(t_now)
        t_now += dt

    # pad to 18000 s
    while t_now <= 18000:
        HRR.append(0.0)
        t.append(t_now)
        t_now += dt

    return np.array(t), np.array(HRR)


def main():
    input_path = input("Input file path (C1: Fire load density (MJ/m²); C2: Fire growth rate (s); C3: Opening factor; C4: Exposed surface area (m2)): ").strip()
    Compartment_length = float(input("Compartment length (m): ").strip())
    Compartment_width = float(input("Compartment width (m): ").strip())
    Compartment_height = float(input("Compartment height (m): ").strip())
    # Exposed_surface_area = float(input("Exposed surface area (m2): ").strip())

    floor_area = Compartment_length * Compartment_width
    total_enclosure_area = (
        floor_area * 2
        + Compartment_height * Compartment_width * 2
        + Compartment_height * Compartment_length * 2
    )


    data = pd.read_csv(input_path, header=None).astype(float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), "stacked_lstm_model.pth")
    model = StackedSeq2SeqLSTM(1, 64, 64, 32, 32, 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    hrr_curves = []

    beta = 0.7


    for idx, row in data.iterrows():
        C1 = float(row[0])  # Fire load density (MJ/m²)
        C2 = float(row[1])  # Fire growth rate (s)
        C3 = float(row[2])  # Opening factor
        C4 = float(row[3]) # Exposed surface area

        Et = C1 * floor_area/total_enclosure_area
        t0 = 0.009 * Et / C3
        dchar0 = 2 * beta * t0
        qchar0 = 28.315 + 3.773 * max(dchar0 - 10.0, 0.0)
        e0 = qchar0 * C4

        E = [e0]
        en = e0

        while True:
            t_iter = 0.009 * en / C3
            dchar = 2 * beta * t_iter
            qchar = 3.773 * dchar
            en = qchar * C4/total_enclosure_area

            if en <= (e0 / 10.0):
                break

            E.append(en)

        E_sum = sum(E)

        data.at[idx, 0] = C1 + E_sum / floor_area


    for _, row in data.iterrows():
        C1, C2, C3 = float(row[0]), float(row[1]), float(row[2])

        A = C1 * (4.182 / (total_enclosure_area / floor_area)) * 30.25
        t_alpha = C2 * total_enclosure_area / 126.5
        Y = C3 * 1.4 * 126.5

        t_arr, HRR = compute_hrr_curve(A, t_alpha, Y, dt=1)
        mask = (t_arr % 60 == 0)
        hrr_curves.append(HRR[mask])

    X_data = np.array(hrr_curves, dtype=np.float32)
    if X_data.ndim == 2:
        X_data = X_data[:, :, np.newaxis]
    elif X_data.ndim == 1:
        X_data = X_data[np.newaxis, :, np.newaxis]

    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
        if predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]
        if predictions.ndim == 3:
            predictions = predictions.squeeze(-1)

    time_minutes = np.arange(predictions.shape[1], dtype=int)
    df = pd.DataFrame(predictions)
    df.columns = [str(t) for t in time_minutes]
    df.T.to_csv("Predicted_fire_curve.csv", header=False)

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
