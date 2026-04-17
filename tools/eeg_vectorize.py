import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import argparse

# 1. Load the ARFF file
def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df

# 2. Extract EEG features and process the label column
def extract_features_and_labels(df):
    # Use the first 14 columns (EEG data channels) directly as features
    eeg_features = df.iloc[:, :14]  # first 14 columns are EEG signal data

    # Convert the eyeDetection label column to integers
    eeg_labels = df['eyeDetection'].apply(lambda x: int(x.decode('utf-8')))

    return eeg_features, eeg_labels

# 3. Standardize the features
def standardize_features(eeg_features):
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(eeg_features)
    return pd.DataFrame(standardized_features, columns=eeg_features.columns)

# 4. Save to a CSV file
def save_to_csv(features, labels, output_path):
    standardized_data = features.copy()
    standardized_data['eyeDetection'] = labels  # attach the label column
    standardized_data.to_csv(output_path, index=False)
    print(f"Standardized data saved to: {output_path}")

# Main entry point: load ARFF data, standardize, and save
def main(input_file, output_file):
    # Load ARFF data
    df = load_arff_data(input_file)

    # Extract features and labels
    eeg_features, eeg_labels = extract_features_and_labels(df)

    # Standardize features
    standardized_features = standardize_features(eeg_features)

    # Save to CSV
    save_to_csv(standardized_features, eeg_labels, output_file)

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EEG ARFF file, standardize features, and save as CSV.')

    # Command-line arguments for input and output files
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input ARFF file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function
    main(args.input_file, args.output_file)

