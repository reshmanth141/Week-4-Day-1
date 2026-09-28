"""Identify independent (feature) and dependent (target) variables."""

import argparse
from pathlib import Path

import pandas as pd


def identify_variables(dataframe: pd.DataFrame, target: str | None = None):
	"""Return the feature columns and target column from a DataFrame.

	When ``target`` is omitted, the final column is used as the target.
	"""
	if dataframe.empty or len(dataframe.columns) == 0:
		raise ValueError("The dataset must contain at least one column.")

	target = target or dataframe.columns[-1]
	if target not in dataframe.columns:
		raise ValueError(f"Target column {target!r} was not found.")

	features = [column for column in dataframe.columns if column != target]
	return features, target


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Identify independent and dependent variables in a CSV dataset."
	)
	parser.add_argument("csv_file", type=Path, help="Path to the CSV dataset")
	parser.add_argument(
		"--target",
		help="Dependent-variable column name (defaults to the last column)",
	)
	args = parser.parse_args()

	dataframe = pd.read_csv(args.csv_file)
	features, target = identify_variables(dataframe, args.target)

	print("Independent variables (features):")
	print(", ".join(features) if features else "None")
	print(f"Dependent variable (target): {target}")


if __name__ == "__main__":
	main()
