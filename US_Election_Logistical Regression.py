
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from US_Election import FILE, clean_data


def main():
    # Load and clean the dataset
    df = clean_data(FILE)

    # Create the target variable
    # 0 = Republican, 1 = Democrat
    dem_votes = "2020 Democrat vote raw"
    rep_votes = "2020 Republican vote raw"

    # Remove tied counties if any exist
    df = df[df[dem_votes] != df[rep_votes]].copy()

    y = (df[dem_votes] > df[rep_votes]).astype(int)

    # Select socioeconomic and demographic features
    # Exclude county, state, and election results
    feature_cols = [
        col for col in df.columns
        if col not in ("county", "state")
        and not col.startswith("2020 ")
    ]

    X = df[feature_cols].copy()

    if X.isna().any().any():
        raise ValueError("Missing values remain in the input features.")

    print("\nLogistic Regression")
    print(f"Total counties: {len(df)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Republican counties: {(y == 0).sum()}")
    print(f"Democratic counties: {(y == 1).sum()}")

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining counties: {len(X_train)}")
    print(f"Testing counties: {len(X_test)}")

    # Build Logistic Regression model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nModel Evaluation")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\nConfusion Matrix")
    print("                     Pred Rep   Pred Dem")
    print(f"Actual Republican    {cm[0, 0]:<10} {cm[0, 1]}")
    print(f"Actual Democrat      {cm[1, 0]:<10} {cm[1, 1]}")

    # Examine model coefficients
    coefficients = pd.Series(
        model.named_steps["logistic_regression"].coef_[0],
        index=feature_cols
    ).sort_values()

    print("\nTop 5 Features Associated with Republican Wins")
    print(coefficients.head(5).round(4).to_string())

    print("\nTop 5 Features Associated with Democratic Wins")
    print(
        coefficients.tail(5)
        .sort_values(ascending=False)
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
