import numpy as np
import pandas as pd
def normalize_value(value):
    """
    Normalize a value to a canonical string form, dropping the decimal point
    and trailing zeros when possible.
    :param value: the value to normalize
    :return: the normalized string
    """
    try:
        # Try to convert the value to float, then to int, then to string
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))  # drop the decimal point and trailing zeros
        else:
            return str(float_value)
    except ValueError:
        # Fall back to the string form of the original value
        return str(value)
def count_inconsistent_entries(dirty_df, clean_df, index_column):
    """
    Count the number of inconsistent entries (rows) between dirty and clean data.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: name of the index column used for alignment
    :return: number of inconsistent entries
    """
    # Align dirty and clean data on the same index
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # Collect inconsistent entry indices
    inconsistent_entry_indices = set()

    # Iterate over all columns and locate mismatched cells between dirty and clean
    for column in dirty_df.columns:
        # Find cells in the current column where dirty and clean values differ
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # Add the mismatched indices to the set
        inconsistent_entry_indices.update(mismatched_indices)

    # Return the count of inconsistent entries
    return len(inconsistent_entry_indices)


def generate_change_report(dirty_df, clean_df, index_column,output_file_name):
    """
    Compare cell-level changes between dirty and clean data and emit change.CSV.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: name of the index column used for alignment
    :return: number of inconsistent cells; also writes change.CSV to disk
    """
    # Align dirty and clean data on the same index
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)
    # Accumulator for change records
    changes = []

    # Iterate over all columns and locate mismatched cells between dirty and clean
    for column in dirty_df.columns:
        # Find cells in the current column where dirty and clean values differ
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        for idx in mismatched_indices:
            changes.append({
                'index': idx,
                'attribute': column,
                'dirty_value': dirty_df.at[idx, column],
                'clean_value': clean_df.at[idx, column]
            })

    # Materialize the change records as a DataFrame
    change_df = pd.DataFrame(changes)

    # Save the result to a CSV file
    # change_df.to_csv(r"./change.CSV", index=False)
    # print("Cells with differences saved to change.CSV")
    change_df.to_csv(output_file_name, index=False)
    print(f"Cells with differences saved to {output_file_name}")
    # Return the total number of inconsistent cells
    return len(change_df)


def replace_with_empty_if_different(dirty_df, clean_df, index_column):
    """
    Compare cells between dirty and clean data; whenever they differ, replace
    the dirty value with 'empty'.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: name of the index column used for alignment
    :return: the processed dirty DataFrame
    """
    # Align dirty and clean data on the same index
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # Iterate over all columns and locate mismatched cells between dirty and clean
    for column in dirty_df.columns:
        # Find cells in the current column where dirty and clean values differ
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # Replace the mismatched dirty values with 'empty'
        for idx in mismatched_indices:
            dirty_df.at[idx, column] = 'empty'

    # Restore the original index column
    dirty_df = dirty_df.reset_index()
    # Save the result to a CSV file
    dirty_df.to_csv(r"./dirty_df.csv", index=False)
    return dirty_df
def replace_half_with_clean_value(dirty_df, clean_df, index_column):
    """
    Compare cells between dirty and clean data; randomly replace half of the
    inconsistent cells with clean values and leave the rest unchanged.

    :param dirty_df: dirty DataFrame
    :param clean_df: clean DataFrame
    :param index_column: name of the index column used for alignment
    :return: the processed dirty DataFrame
    """
    # Align dirty and clean data on the same index
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # Iterate over all columns and locate mismatched cells between dirty and clean
    for column in dirty_df.columns:
        # Find cells in the current column where dirty and clean values differ
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # If mismatches exist, randomly replace half of them
        if len(mismatched_indices) > 0:
            # Randomly sample half of the mismatched indices
            num_to_replace = len(mismatched_indices) // 2
            indices_to_replace = np.random.choice(mismatched_indices, num_to_replace, replace=False)

            # Replace the selected dirty values with clean ones
            for idx in indices_to_replace:
                dirty_df.at[idx, column] = clean_df.at[idx, column]

    # Restore the original index column
    dirty_df = dirty_df.reset_index()
    # Save the result to a CSV file
    dirty_df.to_csv(r"./dirty_df.csv", index=False)
    return dirty_df
# Usage example; do not modify the code above
if __name__ == '__main__':
    dirty_df = pd.read_csv('../Data/5_tax/subset_directly_dirty_index_10k.csv')
    clean_df = pd.read_csv('../Data/5_tax/subset_directly_clean_index_10k.csv')
    # replace_half_with_clean_value(dirty_df, clean_df, 'id')
    inconsistent_entries_count = count_inconsistent_entries(dirty_df, clean_df, 'index')
    print(f'Dirty and clean data have {inconsistent_entries_count} inconsistent entries.')

    inconsistent_cells = generate_change_report(dirty_df, clean_df, 'index',"./change.CSV")
    print(f'Dirty and clean data have {inconsistent_cells} inconsistent cells.')

