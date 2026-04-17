# Run Script Template

## Horizon
**Paper**: [Horizon: Scalable Dependency-Driven Data Cleaning](https://www.vldb.org/pvldb/vol14/p25)
### Overview
This script wraps the **Horizon** dependency-driven cleaner. Horizon cleans data by detecting functional-dependency (FD) violations. The script loads the specified input data, applies the constraint rules (FDs), and emits a cleaned CSV.

## Command 1:
```bash
python run_horizon.py --input data/input.csv --rule_text data/rules.txt --output data/output.csv
```
### Arguments
- `--input`: **required**. Path to the input CSV to clean.
- `--rule_text`: **required**. Path to the file containing FD rules.
- `--output`: **required**. Path where the cleaned CSV will be saved.

### What the command does
1. **Load the input data**:
   - Supplied via `--input`, e.g. `data/input.csv`.
   - Must be a **CSV** containing the data to clean.
2. **Load the FD rules**:
   - Supplied via `--rule_text`, e.g. `data/rules.txt`.
   - Contains FD rules such as `A => B` (attribute A functionally determines attribute B).

3. **Emit the cleaned result**:
   - Saved to the path supplied by `--output`, e.g. `data/output.csv`.
   - Output is also **CSV**.



### Example
```bash
python run_horizon.py --input data/hospital_test.csv --rule_text data/hospital_rules.txt --output data/hospital_cleaned.csv
```

## Command 2:
```bash
python xx
```
### Arguments
xx
### What the command does
xx
