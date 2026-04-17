# Adapt the clean data format for HoloClean (transposed layout)
# Encapsulate the transformation as a reusable function over input/output files
import csv


def transform_csv_file(input_csv: str, output_csv: str):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='',
                                                                     encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write the header row of the output file
        writer.writerow(['tid', 'attribute', 'correct_val'])

        tid = 0
        # Iterate over each row of the input CSV
        for row in reader:
            for key, value in row.items():
                # if value.lower() == 'empty':  # replace 'empty' with an empty value
                #     value = ''
                writer.writerow([tid, key, value])
            tid += 1

    print(f"Conversion complete. Result saved to {output_csv}")

if __name__ == '__main__':
    # Demonstration / smoke test
    test_input_csv = r'../Data/4_rayyan/clean_rayyan.csv'
    test_output_csv = r'../Data/4_rayyan/rayyan_clean_holoclean.csv'

    # Run the function to process the CSV file
    transform_csv_file(test_input_csv, test_output_csv)
