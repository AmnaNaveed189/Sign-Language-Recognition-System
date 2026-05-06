import pandas as pd
import numpy as np
import pickle, json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

MIN_SAMPLES_PER_CLASS = 10

# ============================================================
# Load the CSV we created in Step 1
# ============================================================
print("Loading landmarks CSV...")
df = pd.read_csv('asl_data/landmarks.csv')
print(f"Total samples: {len(df)}")
print(f"Classes: {df['label'].nunique()}")
print(f"Samples per class:\n{df['label'].value_counts()}")

class_counts = df['label'].value_counts()
rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
if not rare_classes.empty:
    print(
        f"\nDropping classes with fewer than {MIN_SAMPLES_PER_CLASS} samples "
        "because stratified train/val/test splitting would fail:"
    )
    print(rare_classes)
    df = df[df['label'].isin(class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index)]

print(f"\nFiltered samples: {len(df)}")
print(f"Filtered classes: {df['label'].nunique()}")

# Separate features (63 numbers) and labels (letters)
X = df.drop('label', axis=1).values  # shape: (N, 63)
y = df['label'].values                # shape: (N,)

# Encode labels as numbers: A=0, B=1, C=2 ...
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClass encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# ============================================================
# Split: 80% train, 10% val, 10% test
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}")

# ============================================================
# Model 1: Random Forest (fast, good baseline)
# ============================================================
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,    # 200 trees
    max_depth=30,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,           # use all CPU cores
    verbose=1
)
rf.fit(X_train, y_train)


rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
print(f"Random Forest val accuracy: {rf_val_acc:.2%}")

# ============================================================
# Model 2: MLP Neural Network (usually more accurate)
# ============================================================
print("\nTraining MLP Neural Network...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)
mlp.fit(X_train, y_train)


mlp_val_acc = accuracy_score(y_val, mlp.predict(X_val))
print(f"MLP val accuracy: {mlp_val_acc:.2%}")

# ============================================================
# Choose best model
# ============================================================
best_model = mlp if mlp_val_acc > rf_val_acc else rf
best_name  = "MLP" if mlp_val_acc > rf_val_acc else "RandomForest"
print(f"\nBest model: {best_name}")

# Final evaluation on test set (never seen during training)
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Final TEST accuracy: {test_acc:.2%}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred,
    target_names=le.classes_))


# ============================================================
# Save everything
# ============================================================
os.makedirs('models', exist_ok=True)

with open('models/asl_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('models/class_names.json', 'w') as f:
    json.dump(list(le.classes_), f)

print("\nModel saved to: models/asl_classifier.pkl")
print("Label encoder saved to: models/label_encoder.pkl")
