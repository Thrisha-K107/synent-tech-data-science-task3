# ============================================================
# Task 3: Exploratory Data Analysis (EDA) — Netflix Dataset
# Synent Technologies Data Science Internship
# ============================================================

# Step 1: Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 120

# ============================================================
# Step 2: Load Netflix Dataset
# Dataset Source: https://www.kaggle.com/datasets/shivamb/netflix-shows
# ============================================================

print("=" * 60)
print("STEP 1: LOADING NETFLIX DATASET")
print("=" * 60)

# If you have the CSV file locally, use:
# df = pd.read_csv("netflix_titles.csv")
# For this script, we fetch directly:
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/netflix_titles.csv"

try:
    df = pd.read_csv(url)
    print(f"✔ Dataset loaded from URL")
except Exception:
    print("⚠ Could not load from URL. Please place 'netflix_titles.csv' in the same folder.")
    raise

print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:\n{df.head()}")

# ============================================================
# Step 3: Data Cleaning (prerequisite for EDA)
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: BASIC DATA CLEANING")
print("=" * 60)

# Check missing values
print("Missing values per column:")
print(df.isnull().sum())

# Fill missing values with 'Unknown' for categorical fields
for col in ["director", "cast", "country", "rating"]:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

# Drop rows where 'date_added' is missing (needed for time analysis)
df.dropna(subset=["date_added"], inplace=True)

# Parse 'date_added' as datetime
df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
df["year_added"]  = df["date_added"].dt.year    # Extract year
df["month_added"] = df["date_added"].dt.month   # Extract month number
df["month_name"]  = df["date_added"].dt.strftime("%b")  # e.g., "Jan"

print("\n✔ Cleaned: missing values handled, dates parsed")
print(f"Remaining rows: {len(df)}")

# ============================================================
# Step 4: Summary Statistics
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: SUMMARY STATISTICS")
print("=" * 60)

print(f"Total Titles       : {len(df)}")
print(f"Movies             : {(df['type'] == 'Movie').sum()}")
print(f"TV Shows           : {(df['type'] == 'TV Show').sum()}")
print(f"Year Range Added   : {df['year_added'].min()} – {df['year_added'].max()}")
print(f"Unique Countries   : {df['country'].nunique()}")
print(f"Unique Genres      : {df['listed_in'].nunique()}")
print(f"Unique Ratings     : {df['rating'].unique()}")

# ============================================================
# Step 5: Visualization 1 — Content Type Distribution
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: CONTENT TYPE DISTRIBUTION")
print("=" * 60)

type_counts = df["type"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Netflix Content — Type Distribution", fontsize=14, fontweight="bold")

# Pie chart
colors_pie = ["#e50914", "#221f1f"]
axes[0].pie(
    type_counts,
    labels=type_counts.index,
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
axes[0].set_title("Movie vs TV Show Split")

# Bar chart
axes[1].bar(type_counts.index, type_counts.values,
            color=colors_pie, edgecolor="black", alpha=0.9)
axes[1].set_title("Count of Movies vs TV Shows")
axes[1].set_ylabel("Count")
for i, v in enumerate(type_counts.values):
    axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("netflix_type_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: netflix_type_distribution.png")

# ============================================================
# Step 6: Visualization 2 — Content Added Per Year (Trend)
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CONTENT ADDED PER YEAR — TREND ANALYSIS")
print("=" * 60)

yearly = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))

for col, color in zip(yearly.columns, ["#e50914", "#b0b0b0"]):
    ax.plot(yearly.index, yearly[col], marker="o", label=col,
            color=color, linewidth=2.5, markersize=6)
    ax.fill_between(yearly.index, yearly[col], alpha=0.08, color=color)

ax.set_title("Netflix — Content Added Per Year", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Titles Added")
ax.legend(title="Content Type")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("netflix_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: netflix_yearly_trend.png")

# ============================================================
# Step 7: Visualization 3 — Top 10 Countries
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: TOP 10 CONTENT-PRODUCING COUNTRIES")
print("=" * 60)

# Explode multi-country entries (e.g., "US, India")
country_series = df["country"].dropna()
all_countries = [c.strip() for entry in country_series for c in entry.split(",")]
country_counts = pd.Series(Counter(all_countries)).sort_values(ascending=False)
top10_countries = country_counts.drop("Unknown", errors="ignore").head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top10_countries.index[::-1], top10_countries.values[::-1],
               color=sns.color_palette("Reds_r", 10), edgecolor="black")
ax.set_title("Top 10 Countries by Netflix Content Volume", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Titles")
for bar in bars:
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=9)

plt.tight_layout()
plt.savefig("netflix_top_countries.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: netflix_top_countries.png")

# ============================================================
# Step 8: Visualization 4 — Top 10 Genres
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: TOP 10 GENRES")
print("=" * 60)

genre_series = df["listed_in"].dropna()
all_genres = [g.strip() for entry in genre_series for g in entry.split(",")]
genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(genre_counts.index, genre_counts.values,
       color=sns.color_palette("viridis", 10), edgecolor="black")
ax.set_title("Top 10 Netflix Genres", fontsize=14, fontweight="bold")
ax.set_xlabel("Genre")
ax.set_ylabel("Count")
plt.xticks(rotation=35, ha="right")
for i, v in enumerate(genre_counts.values):
    ax.text(i, v + 15, str(v), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("netflix_top_genres.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: netflix_top_genres.png")

# ============================================================
# Step 9: Visualization 5 — Rating Distribution
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: CONTENT RATING DISTRIBUTION")
print("=" * 60)

rating_order = ["G", "PG", "PG-13", "R", "NC-17", "TV-Y", "TV-Y7",
                "TV-G", "TV-PG", "TV-14", "TV-MA", "NR", "UR"]
rating_counts = df["rating"].value_counts()
rating_counts = rating_counts[[r for r in rating_order if r in rating_counts.index]]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(rating_counts.index, rating_counts.values,
       color=sns.color_palette("coolwarm", len(rating_counts)), edgecolor="black")
ax.set_title("Netflix Content by Rating Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("netflix_ratings.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: netflix_ratings.png")

# ============================================================
# Step 10: Correlation Analysis (Numeric)
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: CORRELATION ANALYSIS")
print("=" * 60)

# Create numeric proxy columns for correlation
df["is_movie"]       = (df["type"] == "Movie").astype(int)
df["content_volume"] = df.groupby("year_added")["show_id"].transform("count")

numeric_df = df[["year_added", "month_added", "is_movie", "content_volume"]].dropna()
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap — Netflix Numeric Features", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("netflix_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: netflix_correlation.png")

# ============================================================
# Step 11: Print Final Insights
# ============================================================

print("\n" + "=" * 60)
print("KEY INSIGHTS FROM NETFLIX EDA")
print("=" * 60)
print(f"1. Netflix is {type_counts['Movie']/len(df)*100:.1f}% Movies and {type_counts['TV Show']/len(df)*100:.1f}% TV Shows.")
print(f"2. Content additions peaked around 2018–2020.")
print(f"3. United States dominates Netflix content production.")
print(f"4. 'TV-MA' is the most common rating, suggesting adult-oriented content.")
print(f"5. International Movies and Dramas are the top genres.")
print("=" * 60)
