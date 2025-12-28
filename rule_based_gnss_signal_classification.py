"""
Rule-Based GNSS Signal Labeling (Ocean Environment)
--------------------------------------------------

This module performs rule-based classification of GNSS signal observations
into LOS, Multipath (MP), and NLOS categories using domain-informed thresholds.

The generated labels can be directly used for:
    - Supervised machine learning training
    - Baseline performance comparison
    - Data quality assessment in GNSS navigation systems
"""

import pandas as pd
import matplotlib.pyplot as plt


def label_gnss_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign LOS, MP, or NLOS labels to GNSS signal observations.

    Classification logic is derived from physical signal behavior
    observed in oceanic GNSS environments.

    Parameters
    ----------
    data : pd.DataFrame
        Input GNSS dataset containing:
        - signal_strength (C/N0)
        - elevation_angle (degrees)
        - pseudorange_rate (m/s)
        - range_acceleration (m/s^2)

    Returns
    -------
    pd.DataFrame
        Dataset with an added 'label' column.
    """

    # Initialize all samples as NLOS (worst-case assumption)
    data['label'] = 'NLOS'

    # -------------------------------
    # Line-of-Sight (LOS) conditions
    # -------------------------------
    los_mask = (
        (data['signal_strength'] > 45) &
        (data['elevation_angle'] >= 30) &
        (data['pseudorange_rate'].between(-500, 500)) &
        (data['range_acceleration'].between(-1.5, 1.5))
    )

    data.loc[los_mask, 'label'] = 'LOS'

    # --------------------------------
    # Multipath (MP) conditions
    # --------------------------------
    mp_mask = (
        (data['signal_strength'].between(26, 45)) &
        (data['elevation_angle'].between(10, 30)) &
        (data['pseudorange_rate'].abs() > 500) &
        (data['range_acceleration'].abs() > 4)
    )

    data.loc[mp_mask, 'label'] = 'MP'

    return data


def summarize_labels(data: pd.DataFrame) -> None:
    """
    Print and visualize the distribution of GNSS signal labels.
    """

    label_counts = data['label'].value_counts()

    print("\nGNSS Signal Label Distribution:")
    for label in ['LOS', 'MP', 'NLOS']:
        print(f"{label}: {label_counts.get(label, 0)} samples")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.bar(label_counts.index, label_counts.values)
    plt.title("LOS / MP / NLOS Distribution (Ocean GNSS Data)", fontsize=14)
    plt.xlabel("Signal Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # -------------------------------------------------
    # Load GNSS ocean dataset
    # -------------------------------------------------
    INPUT_FILE = "PRN9_INPUTS_6_S1_Ocean.csv"
    OUTPUT_FILE = "gnss_labeled_ocean_data.csv"

    gnss_data = pd.read_csv(INPUT_FILE)

    # -------------------------------------------------
    # Apply rule-based labeling
    # -------------------------------------------------
    labeled_data = label_gnss_signals(gnss_data)

    # -------------------------------------------------
    # Save labeled dataset for ML training
    # -------------------------------------------------
    labeled_data.to_csv(OUTPUT_FILE, index=False)

    # -------------------------------------------------
    # Summary and visualization
    # -------------------------------------------------
    summarize_labels(labeled_data)
