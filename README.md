# synent-tech-data-science-task3
My first Internship with Task 3
# 🔍 Task 3: Exploratory Data Analysis (EDA) — Netflix Dataset
### Synent Technologies — Data Science Internship

---

## 📌 Problem Statement

With thousands of titles on Netflix, understanding content distribution, production trends, country contributions, and audience ratings provides strategic insight into how the platform has grown. This project conducts a full EDA on the Netflix dataset to surface these patterns using statistical methods and visualizations.

---

## 📂 Dataset Details

| Property | Details |
|---|---|
| **Name** | Netflix Movies and TV Shows |
| **Source** | [Kaggle — Shivam Bansal](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| **Rows** | ~8,800 |
| **Columns** | 12 |
| **Key Columns** | type, title, director, country, date_added, listed_in, rating |

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — core chart rendering
- `seaborn` — heatmaps and styled charts
- `collections.Counter` — efficient frequency counting

---

## 🔍 Step-by-Step Code Explanation

### Step 1 — Load Dataset
```python
df = pd.read_csv(url)
```
> Reads the Netflix CSV from a public URL. In production, replace `url` with your local file path.

---

### Step 2 — Basic Cleaning (Pre-EDA)
```python
for col in ["director", "cast", "country", "rating"]:
    df[col].fillna("Unknown", inplace=True)
df.dropna(subset=["date_added"], inplace=True)
df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
df["year_added"] = df["date_added"].dt.year
```
> Fills categorical nulls with `"Unknown"` to preserve row count. Drops rows missing `date_added` since we need it for trend analysis. `pd.to_datetime()` converts the string dates into proper datetime objects so we can extract `.dt.year` and `.dt.month`.

---

### Step 3 — Summary Statistics
```python
print(f"Movies: {(df['type'] == 'Movie').sum()}")
print(f"TV Shows: {(df['type'] == 'TV Show').sum()}")
```
> Boolean series `df['type'] == 'Movie'` creates a True/False series; `.sum()` counts the `True` values (Python treats `True` as `1`).

---

### Step 4 — Content Type Pie + Bar Chart
```python
type_counts = df["type"].value_counts()
axes[0].pie(type_counts, autopct="%1.1f%%", ...)
axes[1].bar(type_counts.index, type_counts.values, ...)
```
> `value_counts()` returns how often each unique value appears. `autopct="%1.1f%%"` formats the pie slice label to 1 decimal place percentage.

---

### Step 5 — Yearly Trend Line Chart
```python
yearly = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
ax.plot(yearly.index, yearly[col], ...)
ax.fill_between(yearly.index, yearly[col], alpha=0.08, ...)
```
> `groupby` on two columns then `.size()` counts combinations. `.unstack()` pivots the inner group level (`type`) into separate columns. `fill_between()` creates the semi-transparent area under the line for visual impact.

---

### Step 6 — Top 10 Countries
```python
all_countries = [c.strip() for entry in country_series for c in entry.split(",")]
country_counts = pd.Series(Counter(all_countries)).sort_values(ascending=False)
```
> Many rows list multiple countries (e.g., "US, India"). We split each entry on commas and flatten into one big list. `Counter` counts occurrences efficiently. `.drop("Unknown")` excludes our placeholder.

---

### Step 7 — Top 10 Genres
```python
all_genres = [g.strip() for entry in genre_series for g in entry.split(",")]
genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False).head(10)
```
> Same pattern as countries — genres are comma-separated. This list comprehension flattens all genres across all titles.

---

### Step 8 — Rating Distribution
```python
rating_counts = df["rating"].value_counts()
```
> Simple frequency count of each rating category. We use a pre-defined `rating_order` list to arrange them from child-friendly to adult content for logical axis ordering.

---

### Step 9 — Correlation Heatmap
```python
df["is_movie"] = (df["type"] == "Movie").astype(int)
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ...)
```
> We encode `type` as a binary integer (`is_movie`). `.corr()` computes the Pearson correlation coefficient between all numeric columns. `annot=True` writes the value inside each cell. Values near `1` or `-1` show strong relationships.

---

## 📊 Key Insights

| # | Insight |
|---|---|
| 1 | Netflix library is ~70% Movies and ~30% TV Shows |
| 2 | Content additions grew rapidly from 2015 to 2019 |
| 3 | USA dominates production, followed by India and UK |
| 4 | TV-MA is the most common content rating |
| 5 | "International Movies" and "Dramas" are top genres |

---

## 📁 Output Files

| File | Description |
|---|---|
| `netflix_type_distribution.png` | Pie and bar chart of Movie vs TV Show |
| `netflix_yearly_trend.png` | Line chart of content added per year |
| `netflix_top_countries.png` | Top 10 content-producing countries |
| `netflix_top_genres.png` | Top 10 genres by title count |
| `netflix_ratings.png` | Rating category distribution |
| `netflix_correlation.png` | Correlation heatmap of numeric features |

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn

python task3_eda_netflix.py
```

> **Note:** Download the Netflix dataset from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) and place it as `netflix_titles.csv` in the same directory if the URL fails.

---

## 🔗 Repository

`synent-task3-netflixeda-<yourname>`

---

*Synent Technologies Data Science Internship — Task 3*
