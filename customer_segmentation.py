# ============================================================
#  PROJECT 2 — Customer Segmentation Analysis
#  Tools: Pandas, Scikit-learn (KMeans), Matplotlib, Plotly
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# STEP 1 — LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")

print("=" * 50)
print("STEP 1: DATA LOADED")
print("=" * 50)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn names: {list(df.columns)}")

# ------------------------------------------------------------
# STEP 2 — DATA CLEANING & PREPROCESSING
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2: DATA CLEANING")
print("=" * 50)

# Check missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Rename columns for easier use
df.columns = ["CustomerID", "Gender", "Age", "Income", "SpendingScore"]

# Check duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")

# Encode Gender to numbers (Male=0, Female=1)
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

print(f"\nGender distribution:\n{df['Gender'].value_counts()}")
print(f"\nAge range     : {df['Age'].min()} - {df['Age'].max()}")
print(f"Income range  : {df['Income'].min()}k - {df['Income'].max()}k")
print(f"Spending range: {df['SpendingScore'].min()} - {df['SpendingScore'].max()}")
print("\nData cleaning complete!")

# ------------------------------------------------------------
# STEP 3 — EXPLORE THE DATA
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3: DATA EXPLORATION")
print("=" * 50)
print(df[["Age", "Income", "SpendingScore"]].describe().round(2))

# ------------------------------------------------------------
# STEP 4 — FIND BEST K USING ELBOW METHOD
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 4: FINDING BEST NUMBER OF CLUSTERS (ELBOW METHOD)")
print("=" * 50)

X = df[["Income", "SpendingScore"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

print("Inertia values for K=1 to 10:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.2f}")
print("Best K = 5 (elbow point)")

# ------------------------------------------------------------
# STEP 5 — APPLY K-MEANS WITH K=5
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5: APPLYING K-MEANS CLUSTERING (K=5)")
print("=" * 50)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Get cluster centers (inverse transform back to original scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Label clusters based on Income & SpendingScore
cluster_labels = {}
for i, (income, spending) in enumerate(centers):
    if income >= 70 and spending >= 60:
        cluster_labels[i] = "High Income, High Spender 💎"
    elif income >= 70 and spending < 40:
        cluster_labels[i] = "High Income, Low Spender 💰"
    elif income < 40 and spending >= 60:
        cluster_labels[i] = "Low Income, High Spender 🛍️"
    elif income < 40 and spending < 40:
        cluster_labels[i] = "Low Income, Low Spender 💤"
    else:
        cluster_labels[i] = "Average Customer 👥"

df["Cluster_Label"] = df["Cluster"].map(cluster_labels)

print("\nCluster Summary:")
summary = df.groupby("Cluster_Label")[["Age", "Income", "SpendingScore"]].mean().round(1)
print(summary)

print("\nCustomers per Cluster:")
print(df["Cluster_Label"].value_counts())

# ------------------------------------------------------------
# STEP 6 — MATPLOTLIB CHARTS (Static)
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 6: CREATING CHARTS ...")
print("=" * 50)

COLORS = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
CLUSTER_COLORS = {label: COLORS[i] for i, label in cluster_labels.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Customer Segmentation Analysis Dashboard",
             fontsize=16, fontweight="bold")

# --- Chart 1: Elbow Curve ---
ax1 = axes[0, 0]
ax1.plot(list(K_range), inertias, marker="o", color="#2196F3",
         linewidth=2.5, markersize=8)
ax1.axvline(x=5, color="#E91E63", linestyle="--", linewidth=2, label="Best K=5")
ax1.set_title("Elbow Method — Finding Best K", fontweight="bold")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (WCSS)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Chart 2: Main Cluster Scatter Plot ---
ax2 = axes[0, 1]
for cluster_id, label in cluster_labels.items():
    mask = df["Cluster"] == cluster_id
    ax2.scatter(
        df.loc[mask, "Income"],
        df.loc[mask, "SpendingScore"],
        c=COLORS[cluster_id],
        label=label.split(" ", 2)[-1][:20],
        s=80, alpha=0.8, edgecolors="white", linewidth=0.5
    )
# Plot cluster centers
ax2.scatter(centers[:, 0], centers[:, 1],
            c="black", marker="X", s=200, zorder=5, label="Centers")
ax2.set_title("Customer Clusters (Income vs Spending)", fontweight="bold")
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.legend(fontsize=7, loc="upper left")
ax2.grid(True, alpha=0.3)

# --- Chart 3: Gender Distribution per Cluster ---
ax3 = axes[1, 0]
cluster_gender = df.groupby(["Cluster_Label", "Gender"]).size().unstack(fill_value=0)
short_labels = [l.split(",")[0].strip() for l in cluster_gender.index]
x = np.arange(len(short_labels))
w = 0.35
ax3.bar(x - w/2, cluster_gender.get("Female", 0),
        width=w, label="Female", color="#E91E63", alpha=0.85)
ax3.bar(x + w/2, cluster_gender.get("Male", 0),
        width=w, label="Male", color="#2196F3", alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
ax3.set_title("Gender Distribution per Cluster", fontweight="bold")
ax3.set_ylabel("Number of Customers")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

# --- Chart 4: Avg Age, Income, Spending per Cluster ---
ax4 = axes[1, 1]
metrics = df.groupby("Cluster")[["Age", "Income", "SpendingScore"]].mean()
x = np.arange(len(metrics))
w = 0.25
ax4.bar(x - w,   metrics["Age"],           width=w, label="Avg Age",    color="#9C27B0", alpha=0.85)
ax4.bar(x,       metrics["Income"],         width=w, label="Avg Income", color="#2196F3", alpha=0.85)
ax4.bar(x + w,   metrics["SpendingScore"],  width=w, label="Avg Score",  color="#4CAF50", alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels([f"C{i}" for i in range(5)])
ax4.set_title("Avg Age / Income / Spending per Cluster", fontweight="bold")
ax4.set_ylabel("Value")
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("segmentation_dashboard_static.png", dpi=150, bbox_inches="tight")
plt.show()
print("Static chart saved: segmentation_dashboard_static.png")

# ------------------------------------------------------------
# STEP 7 — PLOTLY INTERACTIVE DASHBOARD
# ------------------------------------------------------------
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Customer Clusters (Income vs Spending)",
        "Elbow Curve — Best K",
        "Cluster Size Distribution",
        "Avg Income & Spending per Cluster"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "pie"},     {"type": "bar"}]
    ],
    vertical_spacing=0.18,
    horizontal_spacing=0.12
)

# Plot 1 — Scatter: Clusters
for cluster_id, label in cluster_labels.items():
    mask = df["Cluster"] == cluster_id
    fig2.add_trace(
        go.Scatter(
            x=df.loc[mask, "Income"],
            y=df.loc[mask, "SpendingScore"],
            mode="markers",
            name=label,
            marker=dict(color=COLORS[cluster_id], size=9,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Income: $%{x}k<br>"
                "Spending Score: %{y}<extra></extra>"
            )
        ),
        row=1, col=1
    )
# Cluster centers
fig2.add_trace(
    go.Scatter(
        x=centers[:, 0], y=centers[:, 1],
        mode="markers",
        name="Centers",
        marker=dict(symbol="x", color="black", size=14, line=dict(width=2)),
        hovertemplate="Center<br>Income: $%{x:.0f}k<br>Spending: %{y:.0f}<extra></extra>"
    ),
    row=1, col=1
)

# Plot 2 — Elbow Curve
fig2.add_trace(
    go.Scatter(
        x=list(K_range), y=inertias,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#2196F3", width=2.5),
        marker=dict(size=8),
        hovertemplate="K=%{x}<br>Inertia: %{y:.1f}<extra></extra>"
    ),
    row=1, col=2
)
fig2.add_vline(x=5, line_dash="dash", line_color="#E91E63",
               annotation_text="Best K=5", row=1, col=2)

# Plot 3 — Pie: Cluster sizes
cluster_counts = df["Cluster_Label"].value_counts()
fig2.add_trace(
    go.Pie(
        labels=cluster_counts.index,
        values=cluster_counts.values,
        hole=0.4,
        marker_colors=COLORS,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
    ),
    row=2, col=1
)

# Plot 4 — Grouped Bar: Income & Spending per Cluster
avg = df.groupby("Cluster")[["Income", "SpendingScore"]].mean().round(1).reset_index()
fig2.add_trace(
    go.Bar(x=avg["Cluster"].astype(str), y=avg["Income"],
           name="Avg Income", marker_color="#2196F3",
           hovertemplate="Cluster %{x}<br>Avg Income: $%{y:.1f}k<extra></extra>"),
    row=2, col=2
)
fig2.add_trace(
    go.Bar(x=avg["Cluster"].astype(str), y=avg["SpendingScore"],
           name="Avg Spending Score", marker_color="#4CAF50",
           hovertemplate="Cluster %{x}<br>Avg Score: %{y:.1f}<extra></extra>"),
    row=2, col=2
)

fig2.update_layout(
    title=dict(text="🛍️ Customer Segmentation Dashboard", font=dict(size=20), x=0.5),
    height=780,
    barmode="group",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11),
    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5)
)

fig2.write_html("segmentation_dashboard_interactive.html")
fig2.show()
print("Interactive dashboard saved: segmentation_dashboard_interactive.html")

# ------------------------------------------------------------
# STEP 8 — BUSINESS INSIGHTS
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 8: BUSINESS INSIGHTS SUMMARY")
print("=" * 50)

for label, group in df.groupby("Cluster_Label"):
    count  = len(group)
    avg_income   = group["Income"].mean()
    avg_spending = group["SpendingScore"].mean()
    avg_age      = group["Age"].mean()
    print(f"\n{label}")
    print(f"  Customers : {count}")
    print(f"  Avg Age   : {avg_age:.1f} yrs")
    print(f"  Avg Income: ${avg_income:.1f}k")
    print(f"  Avg Score : {avg_spending:.1f}/100")

print("\n" + "=" * 50)
print("MARKETING RECOMMENDATIONS")
print("=" * 50)
print("💎 High Income, High Spender → VIP loyalty programs, premium products")
print("💰 High Income, Low Spender  → Targeted offers, convince them to spend more")
print("🛍️  Low Income, High Spender  → Budget deals, discounts, EMI options")
print("💤 Low Income, Low Spender   → Low priority, basic engagement only")
print("👥 Average Customer          → General promotions, standard campaigns")

print("\nFiles created:")
print("  segmentation_dashboard_static.png")
print("  segmentation_dashboard_interactive.html")
print("\nProject 2 Complete! 🎉")