# Transpose the clean-data CSV into HoloClean's expected format.
# Wraps the transformation as a function that takes explicit input/output paths.
import csv


def transform_csv_file(input_csv: str, output_csv: str):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='',
                                                                     encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write the output header.
        writer.writerow(['tid', 'attribute', 'correct_val'])

        tid = 0
        # Iterate over the CSV rows.
        for row in reader:
            for key, value in row.items():
                # if value.lower() == 'empty':  # map 'empty' to an empty string
                #     value = ''
                writer.writerow([tid, key, value])
            tid += 1

    print(f"Transformation complete; result saved to {output_csv}")

if __name__ == '__main__':
    # Simulated test.
    test_input_csv = r'../Data/4_rayyan/clean_rayyan.csv'
    test_output_csv = r'../Data/4_rayyan/rayyan_clean_holoclean.csv'

    # Run the transformation.
    transform_csv_file(test_input_csv, test_output_csv)
