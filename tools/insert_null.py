
import pandas as pd
import random


def inject_missing_values(csv_file, output_file, attributes_error_ratio=None, missing_value_in_ori_data='empty',missing_value_representation='empty'):
    """
    Null-value error-injection routine that normalizes any existing null values
    to a unified representation before injecting new ones.

    Args:
        csv_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        attributes_error_ratio (dict): Mapping from attribute name to error
            ratio (percentage).
        missing_value_in_ori_data (str): Representation of null values in the
            original data (default: "empty").
        missing_value_representation (str): Target representation for null
            values (default: "empty").

    Output:
        CSV file with the injected errors.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Iterate over each column and convert floats that are integer-valued to int
    for col in df.columns:
        # Check whether the column dtype is float
        if pd.api.types.is_float_dtype(df[col]):
            # Use apply() to convert integer-valued floats to int
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) and x == int(x) else x)

        # Finally, cast every column to string
        df[col] = df[col].astype(str)
    # Preprocessing: normalize existing nulls (NaN or empty string) to missing_value_representation
    df = df.fillna(missing_value_representation)
    df.replace('', missing_value_representation, inplace=True)
    df.replace('nan', missing_value_representation, inplace=True)
    df.replace('_nan_', missing_value_representation, inplace=True)
    df.replace('null', missing_value_representation, inplace=True)
    df.replace('__NULL__', missing_value_representation, inplace=True)
    df.replace(missing_value_in_ori_data, missing_value_representation, inplace=True)
    if attributes_error_ratio is None:
        print("No error ratio specified; only normalizing existing nulls in the original dataset, no errors added")
    else:
    # Iterate over each attribute and inject null values
        for attribute, error_ratio in attributes_error_ratio.items():
            if attribute in df.columns:
                num_rows = len(df)
                num_errors = int(num_rows * error_ratio / 100)
                error_indices = random.sample(range(num_rows), num_errors)

                # Replace values in the selected rows with the null representation
                df.loc[error_indices, attribute] = missing_value_representation

    # Save the CSV file with the injected errors
    df.to_csv(output_file, index=False)
    print(f"File with injected errors saved to: {output_file}")
if __name__ == "__main__":
    # Usage example
    # Attribute list
    attributes = [
        "journal_issn",
        "journal_title",
        "jounral_abbreviation",
    ]

    # Inject 2% error ratio for each attribute
    attributes_error_ratio = {attribute: 0 for attribute in attributes}

    # inject_missing_values(
    #     csv_file='../Data/4_rayyan/dirty_rayyan.csv',
    #     output_file='../Data/4_rayyan/dirty_rayyan.csv',
    #     attributes_error_ratio=attributes_error_ratio,
    #     missing_value_in_ori_data='empty',
    #     missing_value_representation='empty'
    # )
    # If the clean data contains null values, remember to normalize them to 'empty' as well
    # inject_missing_values(
    #     csv_file='../Data/4_rayyan/dirty.csv',
    #     output_file='../Data/4_rayyan/dirty.csv',
    #     attributes_error_ratio=attributes_error_ratio,
    #     missing_value_in_ori_data='NULL',
    #     missing_value_representation='empty'
    # )
    # If the clean data contains null values, remember to normalize them to 'empty' as well
    inject_missing_values(
        csv_file='../Data/3_beers/clean_index.csv',
        output_file='../Data/3_beers/clean_index.csv',
        attributes_error_ratio=None,
        missing_value_in_ori_data='NaN',
        missing_value_representation='empty'
    )