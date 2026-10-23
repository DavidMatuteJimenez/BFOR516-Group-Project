
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
#testing to commit
# Load the dataset
FILE = Path(__file__).with_name("US_Election_dataset_v1.csv")

VOTE_COLUMNS = [
    ("2020 Democrat vote raw", "2020 Democrat vote %"),
    ("2020 Republican vote raw", "2020 Republican vote %"),
    ("2020 other vote raw", "2020 other vote %"),
]


def clean_data(file_path):
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    old_rows, old_cols = df.shape

    # Remove the unnecessary index column
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()

    # Check for duplicate counties
    df["county"] = df["county"].str.strip()
    df["state"] = df["state"].str.strip()

    duplicates = df.assign(
        county_key=df["county"].str.lower(),
        state_key=df["state"].str.lower()
    ).duplicated(subset=["county_key", "state_key"]).sum()

    # Clean numeric columns
    for col in df.columns:
        if col in ("county", "state"):
            continue

        values = df[col].str.strip()
        values = values.str.replace(",", "", regex=False)
        values = values.str.replace("%", "", regex=False)
        values = values.replace(
            ["", "-", "NA", "N/A", "nan", "NaN"], pd.NA
        )

        df[col] = pd.to_numeric(values, errors="coerce")

    # Recalculate voting percentages
    vote_cols = [raw for raw, pct in VOTE_COLUMNS]
    pct_cols = [pct for raw, pct in VOTE_COLUMNS]

    previous_pcts = df[pct_cols].copy()

    total_votes = df[vote_cols].sum(
        axis=1, min_count=len(vote_cols)
    )

    if total_votes.isna().any() or (total_votes <= 0).any():
        raise ValueError("Missing or invalid vote counts were found.")

    for raw, pct in VOTE_COLUMNS:
        df[pct] = (df[raw] / total_votes * 100).round(2)

    # Check which percentages changed
    changes = (previous_pcts - df[pct_cols]).abs() > 0.1
    changed_rows = changes.any(axis=1)

    # Check counties with more votes than population
    high_vote_counts = total_votes > df["Total Population"]

    # Print cleaning results
    print("\nDataset Cleaning Results")
    print(f"Original: {old_rows:,} rows, {old_cols} columns")
    print(f"Cleaned: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Duplicate counties: {duplicates}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "None")

    print("\nVoting percentage corrections:")
    if not changed_rows.any():
        print("None")

    for i in df.index[changed_rows]:
        for pct in pct_cols:
            if changes.at[i, pct]:
                print(
                    f"{df.at[i, 'county']}, {df.at[i, 'state']}: "
                    f"{previous_pcts.at[i, pct]} -> {df.at[i, pct]}"
                )

    print("\nCounties with more votes than population:")
    if not high_vote_counts.any():
        print("None")

    for i in df.index[high_vote_counts]:
        print(f"{df.at[i, 'county']}, {df.at[i, 'state']}")

    print("\nFirst 5 rows (selected columns):")
    preview = [
        "county",
        "state",
        "Median income (dollars)",
        "2020 Republican vote %"
    ]
    print(df[preview].head().to_string(index=False))

    return df


def main():
    df = clean_data(FILE)

    # PCA, K-Means, and Logistic Regression will go here later.

    # After filtering of data by David, we are using K-Means from here

    # Check data types, unique values and missing values
    print(pd.DataFrame({'dtype': df.dtypes.astype(str), 'unique': df.nunique(), 'missing': df.isna().sum()}))

    # Target: 1 = Republican won, 0 = Democrat won
    df['Winner'] = (df['2020 Republican vote %'] > df['2020 Democrat vote %']).astype(int)

    # Select features (drop text columns and vote columns)
    drop_cols = ['county', 'state',
                 '2020 Democrat vote raw', '2020 Democrat vote %',
                 '2020 Republican vote raw', '2020 Republican vote %',
                 '2020 other vote raw', '2020 other vote %', 'Winner']
    features = [col for col in df.columns if col not in drop_cols]

    df_clean = df.dropna(subset=features + ['Winner']).copy()
    print("\nRows before:", df.shape[0], " Rows after:", df_clean.shape[0])
    print("Features:", len(features))

    # Check skewness here to see if we need a log transformation or not
    for col in features:
        print(f"{col:<75} skewness = {df_clean[col].skew():.2f}")

    # Log transform skewed columns
    skewed_cols = [col for col in features if df_clean[col].skew() > 1 and df_clean[col].min() >= 0]
    print("Skewed columns:", len(skewed_cols))
    print(skewed_cols)

    for col in skewed_cols:
        df_clean[col + '_log'] = np.log1p(df_clean[col])

    features_log = [col + '_log' if col in skewed_cols else col for col in features]

    # Scale the features
    X = df_clean[features_log]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(X_scaled.shape)

    # After log transformation, we check the silhouette score for each K value to get our K value
    inertia = []
    sil_scores = []
    k_range = range(2, 10)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertia, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax2.plot(k_range, sil_scores, marker='o', color='orange')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    plt.savefig("kmeans_elbow_silhouette.png", dpi=150)
    plt.show()

    for k, s in zip(k_range, sil_scores):
        print(f"k={k}: silhouette = {s:.3f}")

    # After getting the highest silhouette score, we use that K value in our final K-Means
    optimal_k = k_range[np.argmax(sil_scores)]
    print("Best k:", optimal_k)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)
    print(df_clean['Cluster'].value_counts())

    # Compare clusters with the vote
    cluster_analysis = df_clean.groupby('Cluster').agg({
        'Winner': ['mean', 'count'],
        'Bachelors Degree': 'mean',
        'Median income (dollars)': 'mean',
        'Density per square km': 'mean',
        'NH-White percentage': 'mean'
    })
    cluster_analysis['DemocratPropensity'] = 1 - cluster_analysis[('Winner', 'mean')]
    print(cluster_analysis)

    # Plot clusters
    plt.figure()
    sns.scatterplot(data=df_clean, x='Bachelors Degree', y='NH-White percentage', hue='Cluster', palette='muted')
    plt.title(f"K-Means Clusters (k={optimal_k})")
    plt.savefig("kmeans_clusters.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
