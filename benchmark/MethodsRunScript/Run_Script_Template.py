import os
import sys
import argparse
import time

import pandas as pd

# Add the parent of the current script directory to the Python path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from Cleaner.Holoclean.Holoclean import Holoclean
from util.getScore import calculate_accuracy_and_recall, calculate_all_metrics
from util.insert_null import inject_missing_values


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(description='Run Holoclean data cleaning script.')

    # CLI arguments with the original default paths preserved.
    parser.add_argument('--dirty_path', type=str,
                        default='../../Data/1_hospitals/dirty.csv',
                        help='Path to the input dirty CSV file.')
    parser.add_argument('--rule_path', type=str, default='../../Data/1_hospitals/dc_rules-validate-fd-Holoclean.txt',
                        help='Path to the input rule file.')
    parser.add_argument('--clean_path', type=str, default='../../Data/1_hospitals/clean_index.csv',
                        help='Path to the input clean CSV file.')
    parser.add_argument('--task_name', type=str, default='hospital_test',
                        help='Task name for the cleaning process.')
    parser.add_argument('--output_path', type=str, default='../../results/Holoclean/',
                        help='Path to save the output results.')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='index_attribute of data')
    parser.add_argument('--mse_attributes', type=str, nargs='*', default=[],
                        help='List of attributes to calculate MSE, separated by space. Example: --mse_attributes Attribute1 Attribute3')
    # Parse the arguments.
    args = parser.parse_args()
    mse_attributes = args.mse_attributes
    stra_path = os.path.join(args.output_path, f"{args.task_name}")
    index_attribute = args.index_attribute
    # Ensure the output directory exists.
    if not os.path.exists(stra_path):
        os.makedirs(stra_path)
    # Run the cleaning step and collect the repaired result.
    # Normalize nulls in the data to a uniform "empty" representation.
    inject_missing_values(
        csv_file=args.clean_path,
        output_file=args.clean_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )
    inject_missing_values(
        csv_file=args.dirty_path,
        output_file=args.dirty_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )
    # Record the start time.
    start_time = time.time()

    print(f"Running Holoclean with dirty file: {args.dirty_path}")

    # Focal point: invoke the cleaning procedure with the required arguments and capture the cleaned output.
    # res_df= Holoclean(
    #     args.dirty_path, args.rule_path, args.clean_path,XXX
    # )
    # Save the repaired data.
    res_path = os.path.join(stra_path, f"{args.task_name}_repaired.csv")
    # Key: ensure the cleaned data is written to res_path.
    #res_df.to_csv(res_path, index=False)
    # print("===============================================")

    # Record the end time and compute elapsed time.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Results saved to {res_path}")
    print(f"Holoclean finished in {elapsed_time} seconds.")


    print("Starting evaluation:")
    # Load the clean data, dirty data, and repaired data.
    inject_missing_values(
        csv_file=res_path,
        output_file=res_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )
    clean_data = pd.read_csv(args.clean_path)
    dirty_data = pd.read_csv(args.dirty_path)
    cleaned_data = pd.read_csv(res_path)

    # Attributes defined by the rules.
    attributes = clean_data.columns.tolist()
    # Call the metric function.
    results = calculate_all_metrics(clean_data, dirty_data, cleaned_data, attributes, stra_path, args.task_name,
                                    index_attribute=index_attribute, mse_attributes=mse_attributes)
    # Output file path.
    results_path = os.path.join(stra_path, f"{args.task_name}_total_evaluation.txt")
    # Back up the original stdout.
    original_stdout = sys.stdout
    # Redirect stdout to the output file.
    with open(results_path, 'w', encoding='utf-8') as f:
        sys.stdout = f  # redirect sys.stdout to the file
        # Print results to the file.
        print("Test results:")
        print(f"Accuracy: {results.get('accuracy')}")
        print(f"Recall: {results.get('recall')}")
        print(f"F1 Score: {results.get('f1_score')}")
        print(f"EDR: {results.get('edr')}")
        print(f"Hybrid Distance: {results.get('hybrid_distance')}")
        print(f"R-EDR: {results.get('r_edr')}")
        print(f"Time: {elapsed_time}")
        print(f"speed: {100*float(elapsed_time)/clean_data.shape[0]} seconds/100num")
    # Restore stdout.
    sys.stdout = original_stdout
    # Also print to the terminal.
    print("Test results:")
    print(f"Accuracy: {results.get('accuracy')}")
    print(f"Recall: {results.get('recall')}")
    print(f"F1 Score: {results.get('f1_score')}")
    print(f"EDR: {results.get('edr')}")
    print(f"Hybrid Distance: {results.get('hybrid_distance')}")
    print(f"R-EDR: {results.get('r_edr')}")
    print(f"time(s): {elapsed_time}")
    print(f"speed: {100 * float(elapsed_time) / clean_data.shape[0]} seconds/100num")
    print("Evaluation complete. Detailed logs: " + str(stra_path))



if __name__ == "__main__":
    main()
