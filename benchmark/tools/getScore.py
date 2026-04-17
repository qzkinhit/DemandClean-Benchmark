import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, jaccard_score
import numpy as np


def calculate_all_metrics(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', calculate_precision_recall=True,
                          calculate_edr=True, calculate_hybrid=True, calculate_r_edr=True, mse_attributes=[], relax=True,
                          save_debug_files=True):
    """
    Unified entry point that computes several metrics: repair precision and recall, EDR, hybrid distance, and the record-based R-EDR.

    :param clean: clean DataFrame
    :param dirty: dirty DataFrame
    :param cleaned: cleaned DataFrame
    :param attributes: attributes to score over
    :param output_path: directory for saving results
    :param task_name: task name
    :param calculate_precision_recall: whether to compute repair precision and recall
    :param calculate_edr: whether to compute error detection rate (EDR)
    :param calculate_hybrid: whether to compute hybrid distance
    :param calculate_r_edr: whether to compute record-based error detection rate (R-EDR)
    :param relax: case-insensitive comparison (some baselines such as HoloClean lowercase the output)
    :param save_debug_files: whether to save diff CSVs to a debug subdirectory (default True)
    :return: dict of computed metrics
    """
    results = {}

    # Compute precision and recall.
    if calculate_precision_recall:
        try:
            accuracy, recall = calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name,
                                                             index_attribute=index_attribute, relax=relax,
                                                             save_debug_files=save_debug_files)
            results['accuracy'] = accuracy       # Legacy name; this is precision = TP/(TP+FP).
            results['precision'] = accuracy      # Explicit precision field.
            results['recall'] = recall
            f1_score = calF1(accuracy, recall)
            results['f1_score'] = f1_score
            print(f"Repair precision: {accuracy}, Repair recall: {recall}, F1: {f1_score}")
            print("=" * 40)
        except Exception as e:
            print(f"Precision/recall computation failed: {e}")

    # Compute EDR.
    if calculate_edr:
        try:
            edr = get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute=index_attribute, relax=relax)
            results['edr'] = edr
            print(f"Error detection rate (EDR): {edr}")
            print("=" * 40)
        except Exception as e:
            print(f"EDR computation failed: {e}")

    # Compute hybrid distance.
    if calculate_hybrid:
        try:
            hybrid_distance = get_hybrid_distance(clean, cleaned, attributes, output_path, task_name,
                                                  index_attribute=index_attribute, mse_attributes=mse_attributes, relax=relax)
            results['hybrid_distance'] = hybrid_distance
            print(f"Hybrid distance: {hybrid_distance}")
            print("=" * 40)
        except Exception as e:
            print(f"Hybrid distance computation failed: {e}")

    # Compute record-based R-EDR.
    if calculate_r_edr:
        try:
            r_edr = get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute=index_attribute, relax=relax)
            results['r_edr'] = r_edr
            print(f"Record-based error detection rate (R-EDR): {r_edr}")
            print("=" * 40)
        except Exception as e:
            print(f"R-EDR computation failed: {e}")

    return results

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


def default_distance_func(value1, value2):
    """
    Default distance function: 1 if the two values differ, 0 otherwise.
    """
    return (value1 != value2).sum()

def record_based_distance_func(row1, row2):
    """
    Record-level distance function: returns 1 if any value in the row differs, else 0.
    """
    for val1, val2 in zip(row1, row2):
        if val1 != val2:
            return 1  # any value mismatch -> immediately return 1
    return 0  # all values match -> return 0
def calF1(precision, recall):
    """
    Compute the F1 score.

    :param precision: precision
    :param recall: recall
    :return: F1 score
    """
    return 2 * precision * recall / (precision + recall + 1e-10)


def calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', relax=False,
                                  save_debug_files=True):
    """
    Compute repair precision and recall over a set of attributes. Writes the result to file and emits diff CSVs.

    :param save_debug_files: whether to save diff CSVs to a debug subdirectory (default True)
    """
    import os
    import sys
    import pandas as pd

    os.makedirs(output_path, exist_ok=True)

    # Output paths: evaluation text is always written to the root directory; diff CSVs honor save_debug_files.
    out_path = os.path.join(output_path, f"{task_name}_evaluation.txt")
    if save_debug_files:
        debug_dir = os.path.join(output_path, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        diff_dir = debug_dir
    else:
        diff_dir = output_path
    clean_dirty_diff_path = os.path.join(diff_dir, f"{task_name}_clean_vs_dirty.csv")
    dirty_cleaned_diff_path = os.path.join(diff_dir, f"{task_name}_dirty_vs_cleaned.csv")
    clean_cleaned_diff_path = os.path.join(diff_dir, f"{task_name}_clean_vs_cleaned.csv")
    repair_errors_path = os.path.join(diff_dir, f"{task_name}_repair_errors.csv")
    unrepaired_path = os.path.join(diff_dir, f"{task_name}_unrepaired.csv")

    # Back up the original stdout.
    original_stdout = sys.stdout

    # Set the specified attribute as the index.
    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    # If case-insensitive matching is requested, lowercase all values.
    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Redirect stdout to the output file; restore in finally.
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            total_true_positives = 0
            total_false_positives = 0
            total_true_negatives = 0

            clean_dirty_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Clean Value', 'Dirty Value'])
            dirty_cleaned_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value', 'Cleaned Value'])
            clean_cleaned_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Clean Value', 'Cleaned Value'])
            repair_errors = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value', 'Cleaned Value'])
            unrepaired = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value'])

            for attribute in attributes:
                clean_values = clean[attribute].apply(normalize_value)
                dirty_values = dirty[attribute].apply(normalize_value)
                cleaned_values = cleaned[attribute].apply(normalize_value)

                common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
                clean_values = clean_values.loc[common_indices]
                dirty_values = dirty_values.loc[common_indices]
                cleaned_values = cleaned_values.loc[common_indices]

                true_positives = ((cleaned_values == clean_values) & (dirty_values != cleaned_values)).sum()
                false_positives = ((cleaned_values != clean_values) & (dirty_values != cleaned_values)).sum()
                true_negatives = (dirty_values != clean_values).sum()

                mismatched_indices = dirty_values[dirty_values != clean_values].index
                clean_dirty_diff = pd.concat([clean_dirty_diff, pd.DataFrame({
                    'Attribute': attribute, 'Index': mismatched_indices,
                    'Clean Value': clean_values.loc[mismatched_indices],
                    'Dirty Value': dirty_values.loc[mismatched_indices]
                })], ignore_index=True)

                cleaned_indices = cleaned_values[cleaned_values != dirty_values].index
                dirty_cleaned_diff = pd.concat([dirty_cleaned_diff, pd.DataFrame({
                    'Attribute': attribute, 'Index': cleaned_indices,
                    'Dirty Value': dirty_values.loc[cleaned_indices],
                    'Cleaned Value': cleaned_values.loc[cleaned_indices]
                })], ignore_index=True)

                clean_cleaned_indices = cleaned_values[cleaned_values != clean_values].index
                clean_cleaned_diff = pd.concat([clean_cleaned_diff, pd.DataFrame({
                    'Attribute': attribute, 'Index': clean_cleaned_indices,
                    'Clean Value': clean_values.loc[clean_cleaned_indices],
                    'Cleaned Value': cleaned_values.loc[clean_cleaned_indices]
                })], ignore_index=True)

                repair_error_indices = cleaned_values[
                    (cleaned_values != clean_values) & (dirty_values != cleaned_values)].index
                repair_errors = pd.concat([repair_errors, pd.DataFrame({
                    'Attribute': attribute, 'Index': repair_error_indices,
                    'Clean Value': clean_values.loc[repair_error_indices],
                    'Dirty Value': dirty_values.loc[repair_error_indices],
                    'Cleaned Value': cleaned_values.loc[repair_error_indices]
                })], ignore_index=True)

                unrepaired_indices = cleaned_values[(cleaned_values == dirty_values) & (dirty_values != clean_values)].index
                unrepaired = pd.concat([unrepaired, pd.DataFrame({
                    'Attribute': attribute, 'Index': unrepaired_indices,
                    'Clean Value': clean_values.loc[unrepaired_indices],
                    'Dirty Value': dirty_values.loc[unrepaired_indices]
                })], ignore_index=True)

                total_true_positives += true_positives
                total_false_positives += false_positives
                total_true_negatives += true_negatives
                print("Attribute:", attribute, "Correct repairs:", true_positives, "Incorrect repairs:", false_positives,
                      "Cells needing repair:", true_negatives)
                print("=" * 40)

            accuracy = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
            recall = total_true_positives / total_true_negatives if total_true_negatives > 0 else 0

            print(f"Repair precision: {accuracy}")
            print(f"Repair recall: {recall}")
    finally:
        sys.stdout = original_stdout

    # Save diff data to CSV files.
    clean_dirty_diff.to_csv(clean_dirty_diff_path, index=False)
    dirty_cleaned_diff.to_csv(dirty_cleaned_diff_path, index=False)
    clean_cleaned_diff.to_csv(clean_cleaned_diff_path, index=False)
    repair_errors.to_csv(repair_errors_path, index=False)
    unrepaired.to_csv(unrepaired_path, index=False)

    if save_debug_files:
        print(f"Diff files saved to: {diff_dir}")
    else:
        print(f"Diff files saved to:\n{clean_dirty_diff_path}\n{dirty_cleaned_diff_path}\n{clean_cleaned_diff_path}")
    print(f"Incorrect repair file saved to: {repair_errors_path}")
    print(f"Unrepaired-but-should-be-repaired file saved to: {unrepaired_path}")

    return accuracy, recall


def get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', distance_func=default_distance_func, relax=False):
    """
    Compute the error detection rate (EDR) over a set of attributes and write the result to file.
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_edr_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            total_distance_dirty_to_clean = 0
            total_distance_repaired_to_clean = 0

            for attribute in attributes:
                clean_values = clean[attribute].apply(normalize_value)
                dirty_values = dirty[attribute].apply(normalize_value)
                cleaned_values = cleaned[attribute].apply(normalize_value)

                common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
                clean_values = clean_values.loc[common_indices]
                dirty_values = dirty_values.loc[common_indices]
                cleaned_values = cleaned_values.loc[common_indices]

                distance_dirty_to_clean = distance_func(dirty_values, clean_values)
                distance_repaired_to_clean = distance_func(cleaned_values, clean_values)

                total_distance_dirty_to_clean += distance_dirty_to_clean
                total_distance_repaired_to_clean += distance_repaired_to_clean

                print(f"Attribute: {attribute}")
                print(f"Distance (Dirty to Clean): {distance_dirty_to_clean}")
                print(f"Distance (Repaired to Clean): {distance_repaired_to_clean}")
                print("=" * 40)

            if total_distance_dirty_to_clean == 0:
                edr = 0
            else:
                edr = (total_distance_dirty_to_clean - total_distance_repaired_to_clean) / total_distance_dirty_to_clean

            print(f"Total dirty-to-clean distance: {total_distance_dirty_to_clean}")
            print(f"Total repaired-to-clean distance: {total_distance_repaired_to_clean}")
            print(f"Error detection rate (EDR): {edr}")
    finally:
        sys.stdout = original_stdout

    print(f"EDR result saved to: {out_path}")
    return edr

def get_hybrid_distance(clean, cleaned, attributes, output_path, task_name, index_attribute='index', mse_attributes=[], w1=0.5, w2=0.5, relax=False):
    """
    Compute the hybrid distance (MSE + Jaccard) and write the result to file.
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_hybrid_distance_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            total_mse = 0
            total_jaccard = 0
            attribute_count = 0

            for attribute in attributes:
                clean_values = clean[attribute].apply(normalize_value).replace('empty', np.nan).dropna()
                cleaned_values = cleaned[attribute].apply(normalize_value).replace('empty', np.nan).dropna()

                if attribute in mse_attributes and not clean_values.empty and not cleaned_values.empty:
                    try:
                        clean_float = clean_values.astype(float)
                        cleaned_float = cleaned_values.astype(float)
                        # Min-Max normalize against the clean value range so MSE is comparable across columns.
                        col_min = clean_float.min()
                        col_max = clean_float.max()
                        col_range = col_max - col_min
                        if col_range > 1e-10:
                            clean_norm = (clean_float - col_min) / col_range
                            cleaned_norm = (cleaned_float - col_min) / col_range
                        else:
                            # Zero range (constant column); use the raw values.
                            clean_norm = clean_float
                            cleaned_norm = cleaned_float
                        mse = mean_squared_error(clean_norm, cleaned_norm)
                    except ValueError:
                        print(f"Check whether attribute {attribute} is numeric!")
                        mse = np.nan
                else:
                    mse = np.nan

                if not clean_values.empty and not cleaned_values.empty:
                    try:
                        common_indices = clean_values.index.intersection(cleaned_values.index)
                        jaccard = 1 - jaccard_score(
                            clean_values.loc[common_indices],
                            cleaned_values.loc[common_indices],
                            average='macro'
                        )
                    except ValueError:
                        print(f"Cannot compute Jaccard distance: {attribute} is not categorical")
                        jaccard = np.nan
                else:
                    jaccard = np.nan

                if not np.isnan(mse):
                    total_mse += mse
                if not np.isnan(jaccard):
                    total_jaccard += jaccard

                if not np.isnan(mse) or not np.isnan(jaccard):
                    attribute_count += 1

                print(f"Attribute: {attribute}, MSE: {mse}, Jaccard: {jaccard}")

            if attribute_count == 0:
                hybrid_distance = None
            else:
                avg_mse = total_mse / attribute_count if attribute_count > 0 else 0
                avg_jaccard = total_jaccard / attribute_count if attribute_count > 0 else 0
                hybrid_distance = w1 * avg_mse + w2 * avg_jaccard
                print(f"Weighted hybrid distance: {hybrid_distance}")
    finally:
        sys.stdout = original_stdout

    print(f"Hybrid distance result saved to: {out_path}")
    return hybrid_distance

def get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute='index', relax=False):
    """
    Compute the record-based error detection rate (R-EDR). Writes per-record distances and the final R-EDR to file.
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_record_based_edr_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    total_distance_dirty_to_clean = 0
    total_distance_repaired_to_clean = 0

    # Three-way index intersection: compare only on rows that all three datasets share (needed when cleaned drops rows).
    common_indices = clean.index.intersection(dirty.index).intersection(cleaned.index)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            for idx in common_indices:
                clean_row = clean.loc[idx].apply(normalize_value)
                dirty_row = dirty.loc[idx].apply(normalize_value)
                cleaned_row = cleaned.loc[idx].apply(normalize_value)

                distance_dirty_to_clean = record_based_distance_func(dirty_row, clean_row)
                distance_repaired_to_clean = record_based_distance_func(cleaned_row, clean_row)

                total_distance_dirty_to_clean += distance_dirty_to_clean
                total_distance_repaired_to_clean += distance_repaired_to_clean

                print(f"Record {idx}")
                print(f"Distance (Dirty to Clean): {distance_dirty_to_clean}")
                print(f"Distance (Repaired to Clean): {distance_repaired_to_clean}")
                print("=" * 40)

            if total_distance_dirty_to_clean == 0:
                r_edr = 0
            else:
                r_edr = (total_distance_dirty_to_clean - total_distance_repaired_to_clean) / total_distance_dirty_to_clean

            print(f"Total dirty-to-clean distance: {total_distance_dirty_to_clean}")
            print(f"Total repaired-to-clean distance: {total_distance_repaired_to_clean}")
            print(f"Record-based error detection rate (R-EDR): {r_edr}")
    finally:
        sys.stdout = original_stdout

    print(f"R-EDR result saved to: {out_path}")
    return r_edr

def calculate_all_metrics_TEST():
    data = {
        'index1': [1, 2, 3, 4, 5],
        'Attribute1': [1, 2, 3, 4, 5],
        'Attribute2': ['A', 'B', 'C', 'D', 'E'],
        'Attribute3': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    clean_df = pd.DataFrame(data)
    dirty_data = {
        'index1': [1, 2, 3, 4, 5],
        'Attribute1': [1, 9, 3, 4, 5],
        'Attribute2': ['A', 'B', 'X', 'D', 'E'],
        'Attribute3': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    dirty_df = pd.DataFrame(dirty_data)
    cleaned_data = {
        'index1': [1, 2, 3, 4, 5],
        'Attribute1': [1, 9, 3, 4, 5],
        'Attribute2': ['A', 'X', 'C', 'D', 'E'],
        'Attribute3': [1.1, 2.2, 3.3, 4.4, 5.7]
    }
    cleaned_df = pd.DataFrame(cleaned_data)
    attributes = ['Attribute1', 'Attribute2', 'Attribute3']
    output_path = './temp_test_output'
    task_name = 'test_task'
    results = calculate_all_metrics(clean_df, dirty_df, cleaned_df, attributes, output_path, task_name, index_attribute='index1', mse_attributes=['Attribute3'])
    print("Test results:")
    print(f"Accuracy: {results.get('accuracy')}")
    print(f"Recall: {results.get('recall')}")
    print(f"F1 Score: {results.get('f1_score')}")
    print(f"EDR: {results.get('edr')}")
    print(f"Hybrid Distance: {results.get('hybrid_distance')}")
    print(f"R-EDR: {results.get('r_edr')}")
    print("Test passed.")

if __name__ == "__main__":
    clean_path = '../Data/1_hospitals/clean_index.csv'
    dirty_path = '../Data/1_hospitals/dirty_index.csv'
    cleaned_path = '../results/holoclean/1_hospital_ori/1_hospital_ori_repaired.csv'
    output_path = './'
    task_name = '11111'
    clean=pd.read_csv(clean_path)
    dirty=pd.read_csv(dirty_path)
    cleaned=pd.read_csv(cleaned_path)
    attributes = clean.columns.tolist()
    calculate_all_metrics(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index',
                              calculate_precision_recall=True,
                              calculate_edr=True, calculate_hybrid=True, calculate_r_edr=True)
