import numpy as np
import pandas as pd
def normalize_value(value):
    """
    Normalize a value to a string representation, dropping trailing zeros and the decimal point for integers.

    :param value: value to normalize
    :return: normalized string
    """
    try:
        # Try to cast the value to float; if it is an integer value, emit it without a decimal point.
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))  # drop the decimal point and trailing zeros
        else:
            return str(float_value)
    except ValueError:
        # If the value cannot be cast to float, return the raw string representation.
        return str(value)
def count_inconsistent_entries(dirty_df, clean_df, index_column):
    """
    Count the number of rows that differ between the dirty and clean datasets.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: column used to align the two DataFrames
    :return: number of inconsistent entries
    """
    # Align dirty and clean data on the same index.
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # Set of indices that differ.
    inconsistent_entry_indices = set()

    # Iterate over all columns and find cells that differ between dirty and clean.
    for column in dirty_df.columns:
        # Find the indices of cells whose values differ in this column.
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # Add the mismatched indices to the set.
        inconsistent_entry_indices.update(mismatched_indices)

    # Return the number of inconsistent entries.
    return len(inconsistent_entry_indices)


def generate_change_report(dirty_df, clean_df, index_column,output_file_name):
    """
    Compare cells between the dirty and clean datasets and emit a change.CSV report.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: column used to align the two DataFrames
    :return: number of inconsistent cells; also writes change.CSV
    """
    # Align dirty and clean data on the same index.
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)
    # List that accumulates change entries.
    changes = []

    # Iterate over all columns and find cells that differ between dirty and clean.
    for column in dirty_df.columns:
        # Find the indices of cells whose values differ in this column.
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        for idx in mismatched_indices:
            changes.append({
                'index': idx,
                'attribute': column,
                'dirty_value': dirty_df.at[idx, column],
                'clean_value': clean_df.at[idx, column]
            })

    # Store the change entries in a DataFrame.
    change_df = pd.DataFrame(changes)

    # Save the result as CSV.
    # change_df.to_csv(r"./change.CSV", index=False)
    # print("Differing cells saved to change.CSV")
    change_df.to_csv(output_file_name, index=False)
    print(f"Differing cells saved to {output_file_name}")
    # Return the total number of differing cells.
    return len(change_df)


def replace_with_empty_if_different(dirty_df, clean_df, index_column):
    """
    Compare dirty and clean cells; replace any differing dirty cell with 'empty'.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: column used to align the two DataFrames
    :return: the modified dirty DataFrame
    """
    # Align dirty and clean data on the same index.
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # Iterate over all columns and find cells that differ between dirty and clean.
    for column in dirty_df.columns:
        # Find the indices of cells whose values differ in this column.
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # Replace the differing dirty cells with 'empty'.
        for idx in mismatched_indices:
            dirty_df.at[idx, column] = 'empty'

    # Restore the original index column.
    dirty_df = dirty_df.reset_index()
    # Save the result as CSV.
    dirty_df.to_csv(r"./dirty_df.csv", index=False)
    return dirty_df
def replace_half_with_clean_value(dirty_df, clean_df, index_column):
    """
    Compare dirty and clean cells; randomly replace half of the differing dirty cells with the clean value, leaving the other half untouched.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: column used to align the two DataFrames
    :return: the modified dirty DataFrame
    """
    # Align dirty and clean data on the same index.
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # Iterate over all columns and find cells that differ between dirty and clean.
    for column in dirty_df.columns:
        # Find the indices of cells whose values differ in this column.
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # If any cells differ, randomly replace half of them.
        if len(mismatched_indices) > 0:
            # Randomly pick half of the differing indices.
            num_to_replace = len(mismatched_indices) // 2
            indices_to_replace = np.random.choice(mismatched_indices, num_to_replace, replace=False)

            # Replace the selected cells with the clean value.
            for idx in indices_to_replace:
                dirty_df.at[idx, column] = clean_df.at[idx, column]

    # Restore the original index column.
    dirty_df = dirty_df.reset_index()
    # Save the result as CSV.
    dirty_df.to_csv(r"./dirty_df.csv", index=False)
    return dirty_df
# Example usage. Do not modify the code above.
if __name__ == '__main__':
    # 1. Ensure dirty_df loads the dirty data and clean_df loads the clean data.
    dirty_df = pd.read_csv('../Data/adult/dirty_index.csv')
    clean_df = pd.read_csv('../Data/adult/clean_index.csv')

    index_col = 'index'  # your index column name

    # --- Count inconsistent entries (rows) ---
    inconsistent_entries_count = count_inconsistent_entries(dirty_df, clean_df, index_col)
    print(f'{inconsistent_entries_count} entries differ between the dirty and clean datasets.')

    # --- Generate the report and count inconsistent cells ---
    inconsistent_cells = generate_change_report(dirty_df, clean_df, index_col, "./change.CSV")
    print(f'{inconsistent_cells} cells differ between the dirty and clean datasets.')

    # --- Compute the error rate ---
    # Total row count.
    total_rows = len(dirty_df)

    # Number of compared columns (total columns minus the index column).
    # Note: the functions above set_index before comparing, so the denominator should exclude the index column.
    total_columns = len(dirty_df.columns) - 1

    # 1. Entry/row error rate.
    entry_error_rate = inconsistent_entries_count / total_rows if total_rows > 0 else 0

    # 2. Cell error rate.
    total_cells = total_rows * total_columns
    cell_error_rate = inconsistent_cells / total_cells if total_cells > 0 else 0

    print("-" * 30)
    print(f"Total rows: {total_rows}")
    print(f"Compared columns: {total_columns}")
    print(f"Total compared cells: {total_cells}")
    print("-" * 30)
    print(f"Entry Error Rate: {entry_error_rate:.2%}")
    print(f"Cell Error Rate: {cell_error_rate:.2%}")
