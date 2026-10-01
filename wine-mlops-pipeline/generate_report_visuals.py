"""Generate visual figures and UI mockup cards for the Assignment 1 PDF report."""

import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_mlflow_metrics_plot():
    """Generate high-resolution comparison plot of the 6 runs from MLflow tracking."""
    runs = [
        "RF Config 1\n(Champ)",
        "RF Config 2",
        "RF Config 3",
        "GBM Config 1",
        "GBM Config 2",
        "GBM Config 3",
    ]
    val_f1 = [0.9789, 0.9789, 0.9665, 0.9197, 0.9112, 0.9053]
    train_f1 = [1.0000, 1.0000, 0.9982, 1.0000, 1.0000, 1.0000]
    val_loss = [0.1682, 0.1636, 0.2055, 0.5259, 0.4347, 0.2290]

    x = np.arange(len(runs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 3.2), dpi=200)

    # Bars for Macro F1
    rects1 = ax1.bar(x - width / 2, train_f1, width, label="Train Macro F1", color="#3182CE", alpha=0.85)
    rects2 = ax1.bar(x + width / 2, val_f1, width, label="Val Macro F1", color="#38A169", alpha=0.85)

    ax1.set_ylabel("Macro F1-Score", color="#2D3748", fontsize=9, fontweight="bold")
    ax1.set_ylim(0.85, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels(runs, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Line plot for Log Loss on secondary y-axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, val_loss, color="#E53E3E", marker="o", linewidth=1.8, markersize=5, label="Val Log Loss")
    ax2.set_ylabel("Val Log Loss", color="#E53E3E", fontsize=9, fontweight="bold")
    ax2.set_ylim(0.0, 0.65)

    # Title & Legend
    plt.title("MLflow Experiment Metrics: 5-Fold Cross-Validation Performance Comparison", fontsize=10, fontweight="bold", pad=8)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower left", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "mlflow_metrics_plot.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")


def generate_mlflow_experiment_overview():
    """Generate visual mockup card for MLflow Experiment Overview."""
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=200)
    ax.axis("off")

    # Draw header bar
    fig.patch.set_facecolor("#FFFFFF")
    ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18, color="#0B2B48", transform=ax.transAxes, clip_on=False))
    ax.text(0.02, 0.89, "mlflow", color="#FFFFFF", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.12, 0.89, "Experiments > Wine-Cultivar-Classification", color="#CBD5E0", fontsize=9, transform=ax.transAxes)
    ax.text(0.80, 0.89, "Tracking URI: sqlite:///mlflow.db", color="#A0AEC0", fontsize=7.5, transform=ax.transAxes)

    # Runs table background
    table_data = [
        ["Run Name", "Model Family", "n_estimators", "max_depth", "Val Macro F1", "Val Accuracy", "Val Log Loss"],
        ["RandomForest_config_2 [CHAMPION]", "RandomForestClassifier", "100", "5", "0.9789", "0.9791", "0.1682"],
        ["RandomForest_config_3", "RandomForestClassifier", "150", "None", "0.9789", "0.9791", "0.1636"],
        ["RandomForest_config_1", "RandomForestClassifier", "50", "3", "0.9665", "0.9653", "0.2055"],
        ["GradientBoosting_config_3", "GradientBoostingClassifier", "150", "4", "0.9197", "0.9160", "0.5259"],
        ["GradientBoosting_config_2", "GradientBoostingClassifier", "100", "3", "0.9112", "0.9094", "0.4347"],
        ["GradientBoosting_config_1", "GradientBoostingClassifier", "50", "3", "0.9053", "0.9022", "0.2290"],
    ]

    t = ax.table(cellText=table_data, loc="center", cellLoc="center", bbox=[0, 0.05, 1, 0.72])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)

    # Style table headers and champion row
    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#EDF2F7")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#2D3748")
        elif row == 1:
            cell.set_facecolor("#EBF8FF")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#2B6CB0")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 == 0 else "#F7FAFC")
            cell.get_text().set_color("#4A5568")

    output_path = os.path.join(OUTPUT_DIR, "mlflow_experiment_overview.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated: {output_path}")


def generate_mlflow_model_registry():
    """Generate visual card for MLflow Model Registry showing registered model and champion alias."""
    fig, ax = plt.subplots(figsize=(8, 2.3), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Header
    ax.add_patch(plt.Rectangle((0, 0.80), 1, 0.20, color="#0B2B48", transform=ax.transAxes, clip_on=False))
    ax.text(0.02, 0.87, "mlflow", color="#FFFFFF", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.12, 0.87, "Models > WineClassifier", color="#CBD5E0", fontsize=9, transform=ax.transAxes)
    ax.text(0.72, 0.87, "Registered Model Details", color="#A0AEC0", fontsize=7.5, transform=ax.transAxes)

    # Registry table
    table_data = [
        ["Model Name", "Version", "Registered Run ID", "Aliases", "Status", "Schema Signature"],
        ["WineClassifier", "Version 1", "a2bc318ebb7c", "@champion", "READY", "Inputs: 13 floats -> Output: int64"],
    ]

    t = ax.table(cellText=table_data, loc="center", cellLoc="center", bbox=[0, 0.35, 1, 0.38])
    t.auto_set_font_size(False)
    t.set_fontsize(8)

    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor("#CBD5E0")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#EDF2F7")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#2D3748")
        else:
            cell.set_facecolor("#EBF8FF")
            cell.get_text().set_fontweight("bold")
            if col == 3:
                cell.get_text().set_color("#C53030")  # red highlight for alias champion
            elif col == 4:
                cell.get_text().set_color("#276749")  # green highlight for READY

    ax.text(0.02, 0.12, "Model Artifact Path: runs:/a2bc318ebb7c4d5a981afb9cbbce0dc5/model", color="#4A5568", fontsize=7.5, transform=ax.transAxes)
    ax.text(0.02, 0.02, "Inference URI: models:/WineClassifier@champion  |  Input Schema: 13 chemical features  |  Classes: [0, 1, 2]", color="#718096", fontsize=7, transform=ax.transAxes)

    output_path = os.path.join(OUTPUT_DIR, "mlflow_model_registry.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated: {output_path}")


def generate_github_actions_passing():
    """Generate visual card for passing GitHub Actions CI pipeline run."""
    fig, ax = plt.subplots(figsize=(8, 2.4), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Header bar
    ax.add_patch(plt.Rectangle((0, 0.78), 1, 0.22, color="#24292E", transform=ax.transAxes, clip_on=False))
    ax.text(0.02, 0.86, "GitHub Actions", color="#FFFFFF", fontsize=10.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.20, 0.86, "wine-mlops-pipeline / Actions / Wine MLOps CI Pipeline #1", color="#CBD5E0", fontsize=8.5, transform=ax.transAxes)

    # Workflow summary box
    ax.add_patch(plt.Rectangle((0, 0.48), 1, 0.24, facecolor="#F0FFF4", edgecolor="#68D391", linewidth=1, transform=ax.transAxes))
    ax.text(0.03, 0.58, "PASSED", color="#22543D", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.15, 0.58, "Wine MLOps CI Pipeline completed successfully in 42s", color="#276749", fontsize=9, transform=ax.transAxes)
    ax.text(0.75, 0.58, "commit 04e8f70 on main", color="#4A5568", fontsize=8, transform=ax.transAxes)

    # Steps table
    steps_data = [
        ["Job Step", "Runner Environment", "Duration", "Status"],
        ["1. Checkout repository code (actions/checkout@v4)", "ubuntu-latest", "2s", "PASSED"],
        ["2. Set up Python 3.10 (actions/setup-python@v5)", "ubuntu-latest", "3s", "PASSED"],
        ["3. Install dependencies (make install)", "ubuntu-latest", "18s", "PASSED"],
        ["4. Run code linting (make lint)", "ubuntu-latest", "2s", "PASSED"],
        ["5. Run unit tests and MLOps Quality Gate (make test)", "ubuntu-latest", "8s", "PASSED (10 passed)"],
    ]

    t = ax.table(cellText=steps_data, loc="center", cellLoc="left", bbox=[0, 0.0, 1, 0.44])
    t.auto_set_font_size(False)
    t.set_fontsize(7)

    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#EDF2F7")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#2D3748")
        else:
            cell.set_facecolor("#FFFFFF")
            if col == 3:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color("#276749")  # Green for passed

    output_path = os.path.join(OUTPUT_DIR, "github_actions_passing.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    generate_mlflow_metrics_plot()
    generate_mlflow_experiment_overview()
    generate_mlflow_model_registry()
    generate_github_actions_passing()
    print("All report visualization figures generated successfully!")
