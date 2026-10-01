"""EDA on training partition only; holdout is kept untouched for final evaluation."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from heart.data import NUMERIC, ROOT
from heart.model import SEED


def run():
    df = pd.read_csv(ROOT / "data/processed/heart.csv")
    train, _ = train_test_split(df, test_size=0.2, stratify=df.target, random_state=SEED)
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=["#167d9a", "#ef8056"], font_scale=1.05)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(data=train, x=col, hue="target", element="step", bins=16, ax=ax)
    axes.flat[-1].axis("off")
    fig.suptitle("Clinical distributions | training partition only (n=242)", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        train[NUMERIC + ["target"]].corr(method="spearman"),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Spearman associations | continuous features and target")
    fig.tight_layout()
    fig.savefig(out / "correlation.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=train, x="target", hue="target", legend=False, ax=axes[0])
    axes[0].set(title="Training class balance")
    axes[0].set_xticks([0, 1], ["Absent (0)", "Present (1)"])
    missing = train.isna().sum()
    missing = missing[missing > 0]
    missing.plot.bar(ax=axes[1], color="#167d9a")
    axes[1].set(title="Missing values before fold-local imputation", ylabel="Rows", ylim=(0, 6))
    fig.tight_layout()
    fig.savefig(out / "balance_missing.png", dpi=180)
    plt.close(fig)
    train.describe().to_csv(ROOT / "reports/eda_summary.csv")
    (ROOT / "reports/eda.json").write_text(
        json.dumps(
            {
                "rows": len(train),
                "classes": train.target.value_counts().to_dict(),
                "missing": train.isna().sum().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
