import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
from scipy.sparse import hstack

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Adult dataset vectorization and label conversion.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file (cleaned Adult dataset).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file (vectorized dataset).')
    args = parser.parse_args()
    return args

# Main entry point
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load the Adult dataset
    file_path = args.input
    df = pd.read_csv(file_path)

    # Map the 'income' column: '<=50K' -> 0, '>50K' -> 1
    df['income'] = df['income'].apply(lambda x: 0.0 if x == '<=50K' else 1.0)

    # Split features and labels
    X = df.drop(columns=['income'])  # features
    y = df['income']  # labels

    # Define numeric and categorical features
    numeric_features = ['age', 'fnlwgt', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    # Transformer for numeric features
    numeric_transformer = StandardScaler()

    # Apply TF-IDF vectorization to categorical features
    tfidf_vectorizers = {}
    tfidf_features = []

    for col in categorical_features:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_feature = vectorizer.fit_transform(df[col].astype(str))
        tfidf_vectorizers[col] = vectorizer
        tfidf_features.append(tfidf_feature)

    # Concatenate TF-IDF features across all categorical columns
    X_tfidf = hstack(tfidf_features)

    # Standardize numeric features
    X_numeric = numeric_transformer.fit_transform(df[numeric_features])

    # Concatenate numeric and TF-IDF features
    X_final = hstack([X_numeric, X_tfidf])

    # Convert the feature matrix to a DataFrame
    numeric_columns = numeric_features
    tfidf_columns = [f"{col}_tfidf_{i}" for col in categorical_features for i in range(tfidf_vectorizers[col].idf_.shape[0])]
    all_columns = numeric_columns + tfidf_columns

    X_transformed_df = pd.DataFrame(X_final.toarray(), columns=all_columns)

    # Attach the label column
    X_transformed_df['income'] = y.reset_index(drop=True)

    # Save the vectorized data to a CSV file
    output_file_path = args.output
    X_transformed_df.to_csv(output_file_path, index=False)

    # Report the output path
    print(f"Vectorized data saved to: {output_file_path}")

if __name__ == '__main__':
    main()
