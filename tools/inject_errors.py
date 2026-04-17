import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inject errors into the dataset.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file (vectorized dataset).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file with injected errors.')
    parser.add_argument('--error_type', type=str, required=True, choices=['random', 'system'], help='Type of error to inject: random or system errors.')
    parser.add_argument('--percent', type=float, required=True, help='Percentage of the data in which errors are injected.')
    args = parser.parse_args()
    return args

# Inject random errors (whole-row injection)
def inject_random_error(df, percent, target_column):
    """
    Inject random errors into the specified fraction of rows by replacing
    numeric feature values with 3x the column maximum, excluding the label column.
    """
    # Number of rows to corrupt based on the percentage
    num_samples = int(len(df) * percent / 100)

    # Randomly select rows
    random_indices = np.random.choice(df.index, size=num_samples, replace=False)

    # For every numeric feature column (excluding the label), replace the value with 3x the column max
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_column:
            continue  # skip the label column
        max_value = df[col].max()  # column maximum
        df.loc[random_indices, col] = max_value * 3  # replace in the chosen rows

    return df

# Inject systematic errors (whole-row injection)
def inject_system_error(df, percent, target_column):
    """
    Inject systematic errors into the specified fraction of rows by adjusting
    feature values based on model weights, excluding the label column.
    """
    # Split features and labels
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train an SGD classifier
    sgd = SGDClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sgd.fit(X_train, y_train)

    # Retrieve feature weights and pick the top-3 most important features
    feature_weights = np.abs(sgd.coef_)[0]
    top_3_indices = feature_weights.argsort()[-3:][::-1]
    top_3_features = X.columns[top_3_indices]

    # Sort rows by the three highest-weight features
    df_sorted = df.sort_values(by=top_3_features.tolist(), ascending=False)

    # Take the top x% of rows
    num_samples = int(len(df) * percent / 100)
    top_samples = df_sorted.head(num_samples)

    # For the selected rows, replace each of the three highest-weight features with its global mean
    for feature in top_3_features:
        mean_value = df[feature].mean()  # global mean of the feature
        df.loc[top_samples.index, feature] = mean_value  # replace in the chosen rows

    return df

# Main entry point
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load the dataset
    df = pd.read_csv(args.input)

    # Treat the last column as the label column
    target_column = df.columns[-1]

    # Perform error injection
    if args.error_type == 'random':
        print(f'Injecting random errors into {args.percent}% of the dataset...')
        df_with_errors = inject_random_error(df, args.percent, target_column)
    elif args.error_type == 'system':
        print(f'Injecting system errors into {args.percent}% of the dataset...')
        df_with_errors = inject_system_error(df, args.percent, target_column)

    # Save the result to the output file
    df_with_errors.to_csv(args.output, index=False)
    print(f'Injected errors saved to {args.output}')

if __name__ == '__main__':
    main()
