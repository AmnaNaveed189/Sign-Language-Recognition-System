import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MIN_SAMPLES_PER_CLASS = 10
RANDOM_STATE = 42

# Rebuild the exact filtered test split used during training.
df = pd.read_csv("asl_data/landmarks.csv")
class_counts = df["label"].value_counts()
df = df[df["label"].isin(class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index)]

X = df.drop("label", axis=1).values
y = df["label"].values

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

y_encoded = le.transform(y)

_, X_temp, _, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)
_, X_test, _, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

with open("models/asl_classifier.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
y_pred_lbl = le.inverse_transform(y_pred)
y_true_lbl = le.inverse_transform(y_test)

os.makedirs("results", exist_ok=True)

# ============================================================
# Confusion Matrix
# ============================================================
cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=le.classes_)

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    linewidths=0.5,
)
plt.title("ASL Confusion Matrix\n(dark diagonal = good, off-diagonal = errors)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

# ============================================================
# Per-class accuracy bar chart
# ============================================================
report = classification_report(y_true_lbl, y_pred_lbl, output_dict=True)
classes_acc = {k: v["f1-score"] for k, v in report.items() if k in le.classes_}

plt.figure(figsize=(14, 5))
plt.bar(
    classes_acc.keys(),
    classes_acc.values(),
    color=["#1D9E75" if v >= 0.9 else "#D85A30" for v in classes_acc.values()],
)
plt.axhline(y=0.9, color="gray", linestyle="--", label="90% threshold")
plt.title("F1 Score per ASL letter\n(green=good >=90%, red=needs improvement)")
plt.xlabel("Letter")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("results/per_class_accuracy.png", dpi=150)
plt.close()

weak = [k for k, v in classes_acc.items() if v < 0.9]
print(f"\nLetters below 90% F1 score: {weak}")
print("These letters need more training data or data augmentation")
