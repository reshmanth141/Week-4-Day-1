"""Basic scikit-learn classification workflow: fit, predict, and evaluate."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
	# Load data and hold out a stratified test set for evaluation.
	features, labels = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		features, labels, test_size=0.2, random_state=42, stratify=labels
	)

	# Build and fit the model using only the training data.
	model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
	model.fit(X_train, y_train)

	# Predict on unseen data and report evaluation metrics.
	predictions = model.predict(X_test)
	print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
	print(classification_report(y_test, predictions, target_names=load_iris().target_names))


if __name__ == "__main__":
	main()
