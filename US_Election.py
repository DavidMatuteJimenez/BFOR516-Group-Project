
from pathlib import Path
import pandas as pd

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

    # Remove the row with missing median income
    removed_rows = df["Median income (dollars)"].isna().sum()
    df = df.dropna(subset=["Median income (dollars)"]).copy()

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
    print(f"Rows removed (missing median income): {removed_rows}")

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


if __name__ == "__main__":
    main()
