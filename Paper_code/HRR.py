import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_hrr_curve(A, t_alpha, Y, dt=1):
    """
    Compute the HRR curve with growth, steady, and decay phases.

    Parameters:
    - A: Total energy release (area under HRR curve)
    - t_alpha: Fire growth rate parameter
    - Y: Steady-state HRR value
    - dt: Time step for integration

    Returns:
    - t: Time array
    - HRR: HRR values corresponding to time
    """
    t = [0]  # Time array
    HRR = [0]  # HRR values

    # Growth Phase: HRR = (t/t_alpha)^2, until reaching Y
    t_grow = np.sqrt(Y) * t_alpha  # Time to reach HRR = Y
    t_now = 0
    area_grow = 0

    while t_now <= t_grow:
        hrr_now = (t_now / t_alpha) ** 2
        if hrr_now > Y:
            hrr_now = Y
        HRR.append(hrr_now)
        t.append(t_now)
        area_grow += hrr_now * dt
        t_now += dt

    # Steady Phase: Constant HRR = Y until reaching 70% of total area
    area_target = 0.7 * A
    area_steady = 0

    while area_grow + area_steady < area_target:
        HRR.append(Y)
        t.append(t_now)
        area_steady += Y * dt
        t_now += dt

    # Decay Phase: Linear drop to zero
    area_decay = A - (area_grow + area_steady)
    t_decay = 2 * area_decay / Y  # Linear decay duration
    t_end = t_now + t_decay

    while t_now <= t_end:
        hrr_now = Y * (1 - (t_now - t_grow - (area_steady / Y)) / t_decay)  # Linear decrease
        HRR.append(max(0, hrr_now))
        t.append(t_now)
        t_now += dt

    while 18000 >= t_now > t_end:
        hrr_now = 0
        HRR.append(hrr_now)
        t.append(t_now)
        t_now += dt


    return np.array(t), np.array(HRR)


# Load data from CSV file (assume columns: A, t_alpha, Y)
data = pd.read_csv("input.csv",header=None)  # Update with your actual file path
Floor_area=30.25
Total_area=126.5
# Dictionary to store HRR data
hrr_dict = {}

# Process each row
for idx, row in data.iterrows():
    A, t_alpha, Y = row[0]*Floor_area, row[1], row[2]*Total_area*1.4

    # Compute HRR curve
    t, HRR = compute_hrr_curve(A, t_alpha, Y)

    # Sample every 60th point
    mask = (t % 60 == 0)  # 选择 60s 的整数倍
    t_sampled = (t[mask] / 60).astype(int)  # 转换为分钟
    HRR_sampled = HRR[mask]

    # Store data
    hrr_dict[f"Time Case {idx + 1}"] = t_sampled
    hrr_dict[f"HRR Case {idx + 1}"] = HRR_sampled

    # Plot HRR curve
    plt.plot(t, HRR, label=f"Case {idx + 1}")

# Convert to DataFrame (align columns to the longest array)
df_output = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hrr_dict.items()]))

# Save to CSV
df_output.to_csv("HRR.csv", index=False)

# Customize plot
plt.xlabel("Time (s)")
plt.ylabel("HRR (kW)")
plt.title("Heat Release Rate (HRR) Curves")
# plt.legend()
plt.grid()
plt.show()