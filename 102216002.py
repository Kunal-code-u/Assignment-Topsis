import sys
import pandas as pd
import numpy as np

def validate_inputs(input_file, weights, impacts, result_file):
    try:
        # Read the input file
        data = pd.read_csv(input_file)

        # Check if the file has at least three columns
        if data.shape[1] < 3:
            raise ValueError("Input file must have at least three columns.")

        # Check if all columns from 2nd to last contain numeric values
        if not np.all(data.iloc[:, 1:].applymap(np.isreal).all()):
            raise ValueError("All columns from 2nd to last must contain numeric values.")

        # Check if weights and impacts have the same length as the number of criteria columns
        num_criteria = data.shape[1] - 1
        weights = [float(w) for w in weights.split(',')]
        impacts = impacts.split(',')

        if len(weights) != num_criteria or len(impacts) != num_criteria:
            raise ValueError("Number of weights and impacts must match the number of criteria columns.")

        # Check if impacts are valid
        if not all(i in ['+', '-'] for i in impacts):
            raise ValueError("Impacts must be either '+' or '-'.")

        return data, weights, impacts

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

def topsis(data, weights, impacts):
    # Normalize the decision matrix
    criteria = data.iloc[:, 1:]
    norm_matrix = criteria / np.sqrt((criteria**2).sum())

    # Apply weights
    weighted_matrix = norm_matrix * weights

    # Determine ideal best and ideal worst
    ideal_best = []
    ideal_worst = []

    for i, impact in enumerate(impacts):
        if impact == '+':
            ideal_best.append(weighted_matrix.iloc[:, i].max())
            ideal_worst.append(weighted_matrix.iloc[:, i].min())
        else:
            ideal_best.append(weighted_matrix.iloc[:, i].min())
            ideal_worst.append(weighted_matrix.iloc[:, i].max())

    # Calculate distances to ideal best and worst
    dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

    # Calculate Topsis score
    topsis_score = dist_worst / (dist_best + dist_worst)

    # Rank scores
    ranks = topsis_score.rank(ascending=False).astype(int)

    data['Topsis Score'] = topsis_score
    data['Rank'] = ranks

    return data

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]

    data, weights, impacts = validate_inputs(input_file, weights, impacts, result_file)
    result = topsis(data, weights, impacts)

    result.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
