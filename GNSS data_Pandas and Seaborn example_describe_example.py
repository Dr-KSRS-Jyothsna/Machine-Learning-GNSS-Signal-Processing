"""
File: gnss_signal_complete_eda.py
Author: Dr. K. S. R. S. Jyothsna
Description:
Complete Exploratory Data Analysis (EDA) of GNSS signal data using
Pandas describe() and multiple Seaborn visualizations with
customized colors and styles.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# Global Seaborn style
# --------------------------------------------------
sns.set(style="whitegrid")

# --------------------------------------------------
# 1. Create Sample GNSS / Signal Dataset
# --------------------------------------------------
data = {
    "Satellite_ID": ["G01", "G02", "G03", "G04", "G05", "G06"],
    "SNR_dB": [45.2, 38.5, 42.0, 30.8, 47.1, 35.6],
    "Pseudorange_m": [20200000, 20185000, 20215000, 20150000, 20230000, 20190000],
    "Doppler_Hz": [-1200.5, -1150.3, -1180.0, -1305.7, -1100.2, -1250.4],
    "Elevation_deg": [60, 45, 55, 30, 70, 40]
}

df = pd.DataFrame(data)

print("GNSS Signal Dataset:")
print(df)
print("-" * 90)

# --------------------------------------------------
# 2. Statistical Summary using describe()
# --------------------------------------------------
print("Statistical Summary using describe():")
print(df.describe())
print("-" * 90)

# --------------------------------------------------
# 3. Box Plot – SNR Distribution (Green)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(y="SNR_dB", data=df, color="lightgreen")
plt.title("SNR Distribution (Box Plot)")
plt.ylabel("SNR (dB)")
plt.show()

# --------------------------------------------------
# 4. Histogram – Elevation Angle Distribution (Purple)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["Elevation_deg"],
             bins=6,
             kde=True,
             color="purple")
plt.title("Elevation Angle Distribution")
plt.xlabel("Elevation Angle (degrees)")
plt.show()

# --------------------------------------------------
# 5. Scatter Plot – Elevation vs SNR (Red)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Elevation_deg",
                y="SNR_dB",
                data=df,
                color="red",
                s=120)
plt.title("SNR vs Elevation Angle")
plt.xlabel("Elevation (degrees)")
plt.ylabel("SNR (dB)")
plt.show()

# --------------------------------------------------
# 6. Scatter Plot – Doppler vs Pseudorange (Orange)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Doppler_Hz",
                y="Pseudorange_m",
                data=df,
                color="orange",
                s=120)
plt.title("Doppler vs Pseudorange")
plt.xlabel("Doppler (Hz)")
plt.ylabel("Pseudorange (m)")
plt.show()

# --------------------------------------------------
# 7. Scatter Plot – Satellite-wise Coloring (Hue)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Elevation_deg",
                y="SNR_dB",
                hue="Satellite_ID",
                palette="Set2",
                data=df,
                s=120)
plt.title("SNR vs Elevation (Satellite-wise)")
plt.xlabel("Elevation (degrees)")
plt.ylabel("SNR (dB)")
plt.show()

# --------------------------------------------------
# 8. Correlation Heatmap (Coolwarm)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(
    df[["SNR_dB", "Pseudorange_m", "Doppler_Hz", "Elevation_deg"]].corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap of GNSS Parameters")
plt.show()

print("GNSS EDA completed with statistical summary and multiple visualizations.")
