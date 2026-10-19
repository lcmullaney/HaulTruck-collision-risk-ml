
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
#edit file path
DATA_FILE = r"XXXXXXx\observed_risk_dataset.csv"
OUTPUT_DIR = r"XXXXXXX\out"

df = pd.read_csv(DATA_FILE)

features = [
    "speed_A_ms",
    "speed_B_ms",
    "loaded_A",
    "loaded_B",
    "gvw_A_t",
    "gvw_B_t",
    "segment_length_m",
    "grade",
    "speed_limit_kmh"
]

X = df[features]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic_Standard": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000
        ))
    ]),
    "Logistic_Balanced": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),
    "Gaussian_Naive_Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ])
}

labels = ["Safe", "Warning", "Critical"]
results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    report = classification_report(
        y_test, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": report["macro avg"]["f1-score"],
        "Critical Precision": report["Critical"]["precision"],
        "Critical Recall": report["Critical"]["recall"],
        "Critical F1": report["Critical"]["f1-score"]
    })

    print(f"\n {name} ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    print(classification_report(
        y_test, y_pred,
        labels=labels,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test, y_pred,
        labels=labels
    )

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(name.replace("_", " "))
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{OUTPUT_DIR}/observed_model_comparison.csv",
    index=False
)

print("\n FINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

print("\nSaved model comparison and confusion matrices.")