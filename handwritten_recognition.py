
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Create output folder
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# 1. Load dataset
print("Loading handwritten digits dataset...")

digits = load_digits()
X = digits.data
y = digits.target

print("Total images:", len(X))
print("Image size: 8 x 8 pixels")

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))

# 3. Train model
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Predict test images
y_pred = model.predict(X_test)

# 5. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save trained model
joblib.dump(model, output_folder / "handwritten_model.pkl")
print("\nTrained model saved in outputs/handwritten_model.pkl")

# 7. Display sample predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(
        X_test[i].reshape(8, 8),
        cmap="gray_r",
        interpolation="nearest"
    )

    ax.set_title(
        f"Actual: {y_test[i]}\nPredicted: {y_pred[i]}"
    )
    ax.axis("off")

fig.suptitle("Handwritten Digit Recognition - Sample Predictions")
fig.tight_layout()

fig.savefig(output_folder / "sample_predictions.png", dpi=150)
print("Sample predictions saved in outputs/sample_predictions.png")

plt.show()
plt.close(fig)

# 8. Create confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=ax,
    cmap="Blues",
    colorbar=False
)

ax.set_title("Handwritten Digit Recognition - Confusion Matrix")
fig.tight_layout()

fig.savefig(output_folder / "confusion_matrix.png", dpi=150)
print("Confusion matrix saved in outputs/confusion_matrix.png")

plt.show()
plt.close(fig)

# 9. Show one example prediction
sample_index = 0
sample = X_test[sample_index].reshape(1, -1)

prediction = model.predict(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")
print("Actual digit:", y_test[sample_index])
print("Predicted digit:", prediction)

print("\nProject completed successfully!")
