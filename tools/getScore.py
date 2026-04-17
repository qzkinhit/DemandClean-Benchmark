import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, jaccard_score
import numpy as np


def calculate_all_metrics(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', calculate_precision_recall=True,
                          calculate_edr=True, calculate_hybrid=True, calculate_r_edr=True, mse_attributes=[], relax=True):
    """
    Compute multiple metrics in one call: repair precision/recall, EDR, hybrid distance,
    and record-based R-EDR.

    :param clean: clean-data DataFrame
    :param dirty: dirty-data DataFrame
    :param cleaned: cleaned-data DataFrame
    :param attributes: attributes to evaluate
    :param output_path: directory to save results
    :param task_name: task name
    :param calculate_precision_recall: whether to compute repair precision/recall
    :param calculate_edr: whether to compute error reduction rate (EDR)
    :param calculate_hybrid: whether to compute the hybrid-distance metric
    :param calculate_r_edr: whether to compute record-based error reduction rate (R-EDR)
    :param relax: ignore case when comparing (some baselines, e.g. HoloClean, force lowercase output)
    :return: all computed metrics
    """
    results = {}

    # Precision / recall
    if calculate_precision_recall:
        try:
            accuracy, recall = calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name,
                                                             index_attribute=index_attribute, relax=relax)
            results['accuracy'] = accuracy       # legacy: this is actually precision = TP/(TP+FP)
            results['precision'] = accuracy      # explicit precision
            results['recall'] = recall
            f1_score = calF1(accuracy, recall)
            results['f1_score'] = f1_score
            print(f"Repair precision: {accuracy}, repair recall: {recall}, F1: {f1_score}")
            print("=" * 40)
        except Exception as e:
            print(f"Precision/recall failed: {e}")

    # EDR
    if calculate_edr:
        try:
            edr = get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute=index_attribute, relax=relax)
            results['edr'] = edr
            print(f"Error reduction rate (EDR): {edr}")
            print("=" * 40)
        except Exception as e:
            print(f"EDR failed: {e}")

    # Hybrid distance
    if calculate_hybrid:
        try:
            hybrid_distance = get_hybrid_distance(clean, cleaned, attributes, output_path, task_name,
                                                  index_attribute=index_attribute, mse_attributes=mse_attributes, relax=relax)
            results['hybrid_distance'] = hybrid_distance
            print(f"Hybrid distance: {hybrid_distance}")
            print("=" * 40)
        except Exception as e:
            print(f"Hybrid distance failed: {e}")

    # Record-based R-EDR
    if calculate_r_edr:
        try:
            r_edr = get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute=index_attribute, relax=relax)
            results['r_edr'] = r_edr
            print(f"Record-based error reduction rate (R-EDR): {r_edr}")
            print("=" * 40)
        except Exception as e:
            print(f"R-EDR failed: {e}")

    return results

def normalize_value(value):
    """
    Normalize a value into its string form, stripping trailing zeros after the decimal point.

    :param value: value to normalize
    :return: normalized string
    """
    try:
        # Parse as float, coerce to int when integral, then stringify
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))  # drop decimal zeros
        else:
            return str(float_value)
    except ValueError:
        # Non-numeric values fall back to their string form
        return str(value)


def default_distance_func(value1, value2):
    """
    Default distance: 1 if the values differ, 0 if they match.
    """
    return (value1 != value2).sum()

def record_based_distance_func(row1, row2):
    """
    Record-based distance: 1 if any column differs between the two rows, 0 otherwise.
    """
    for val1, val2 in zip(row1, row2):
        if val1 != val2:
            return 1  # differ on any column -> 1
    return 0  # all columns match -> 0
def calF1(precision, recall):
    """
    Compute F1.

    :param precision: precision
    :param recall: recall
    :return: F1
    """
    return 2 * precision * recall / (precision + recall + 1e-10)


def calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', relax=False):
    """
    Compute repair precision and recall over the given attributes, writing results to files
    and producing diff CSVs.
    """
    import os
    import sys
    import pandas as pd

    os.makedirs(output_path, exist_ok=True)

    # Output file paths
    out_path = os.path.join(output_path, f"{task_name}_evaluation.txt")
    clean_dirty_diff_path = os.path.join(output_path, f"{task_name}_clean_vs_dirty.csv")
    dirty_cleaned_diff_path = os.path.join(output_path, f"{task_name}_dirty_vs_cleaned.csv")
    clean_cleaned_diff_path = os.path.join(output_path, f"{task_name}_clean_vs_cleaned.csv")
    repair_errors_path = os.path.join(output_path, f"{task_name}_repair_errors.csv")
    unrepaired_path = os.path.join(output_path, f"{task_name}_unrepaired.csv")

    # Back up the original stdout
    original_stdout = sys.stdout

    # Index by the chosen attribute
    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    # Case-insensitive comparison when relax is set
    if relax:
        clean = clean.map(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.map(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.map(lambda x: x.lower() if isinstance(x, str) else x)

    # Redirect stdout to the output file (try/finally ensures stdout is restored)
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
                print("Attribute:", attribute, "correctly repaired:", true_positives, "incorrectly repaired:", false_positives,
                      "should be repaired:", true_negatives)
                print("=" * 40)

            accuracy = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
            recall = total_true_positives / total_true_negatives if total_true_negatives > 0 else 0

            print(f"Repair precision: {accuracy}")
            print(f"Repair recall: {recall}")
    finally:
        sys.stdout = original_stdout

    # Save diff data as CSV
    clean_dirty_diff.to_csv(clean_dirty_diff_path, index=False)
    dirty_cleaned_diff.to_csv(dirty_cleaned_diff_path, index=False)
    clean_cleaned_diff.to_csv(clean_cleaned_diff_path, index=False)
    repair_errors.to_csv(repair_errors_path, index=False)
    unrepaired.to_csv(unrepaired_path, index=False)

    print(f"Diff files saved:\n{clean_dirty_diff_path}\n{dirty_cleaned_diff_path}\n{clean_cleaned_diff_path}")
    print(f"Incorrectly repaired file saved: {repair_errors_path}")
    print(f"Unrepaired-but-should-repair file saved: {unrepaired_path}")

    return accuracy, recall


def get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', distance_func=default_distance_func, relax=False):
    """
    Compute the error reduction rate (EDR) over the given attributes and write results to a file.
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_edr_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.map(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.map(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.map(lambda x: x.lower() if isinstance(x, str) else x)

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

            print(f"Total distance dirty-to-clean: {total_distance_dirty_to_clean}")
            print(f"Total distance repaired-to-clean: {total_distance_repaired_to_clean}")
            print(f"Error reduction rate (EDR): {edr}")
    finally:
        sys.stdout = original_stdout

    print(f"EDR results saved: {out_path}")
    return edr

def get_hybrid_distance(clean, cleaned, attributes, output_path, task_name, index_attribute='index', mse_attributes=[], w1=0.5, w2=0.5, relax=False):
    """
    Compute the hybrid-distance metric (MSE + Jaccard) and write results to a file.
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_hybrid_distance_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.map(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.map(lambda x: x.lower() if isinstance(x, str) else x)

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
                        # Min-max normalize against the clean range so that
                        # columns with different scales produce comparable MSEs.
                        col_min = clean_float.min()
                        col_max = clean_float.max()
                        col_range = col_max - col_min
                        if col_range > 1e-10:
                            clean_norm = (clean_float - col_min) / col_range
                            cleaned_norm = (cleaned_float - col_min) / col_range
                        else:
                            # Zero range (constant column): use raw values
                            clean_norm = clean_float
                            cleaned_norm = cleaned_float
                        mse = mean_squared_error(clean_norm, cleaned_norm)
                    except ValueError:
                        print(f"Check whether attribute {attribute} is numeric.")
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

    print(f"Hybrid distance results saved: {out_path}")
    return hybrid_distance

def get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute='index', relax=False):
    """
    Compute the record-based error reduction rate (R-EDR) and write per-record
    distances plus the final R-EDR to a file.
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_record_based_edr_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.map(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.map(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.map(lambda x: x.lower() if isinstance(x, str) else x)

    total_distance_dirty_to_clean = 0
    total_distance_repaired_to_clean = 0

    # Three-way index intersection: if cleaned dropped some rows, only compute on common rows
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

            print(f"Total distance dirty-to-clean: {total_distance_dirty_to_clean}")
            print(f"Total distance repaired-to-clean: {total_distance_repaired_to_clean}")
            print(f"Record-based error reduction rate (R-EDR): {r_edr}")
    finally:
        sys.stdout = original_stdout

    print(f"R-EDR results saved: {out_path}")
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
