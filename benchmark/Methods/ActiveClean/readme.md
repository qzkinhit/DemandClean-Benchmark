## ActiveClean Overview

ActiveClean performs gradient-descent-based data cleaning: it incrementally cleans data and updates the downstream model parameters in each iteration, eventually producing the model corresponding to the fully cleaned data.

## File Structure

### 1. `activeclean.py`
- **`activeclean`**: Main entry function.
    - **Inputs**:
      - `dirty_data`: The dirty data.
      - `clean_data`: The clean data.
      - `test_data`: The test data.
      - `index_tuple`: Index array.
      - `batchsize`: Sample size per iteration.
      - `total`: Total number of records.
    - **Helper inputs**:
      - `globali`: The element set.
      - `imap`: The index set to look up.
    - **Output**:
      - `clf`: Indices in `globali` that correspond to entries in `imap`.
- **`error_classifier`**: Trains an error-detection classifier using the sampled labeled data.
    - **Inputs**:
      - `total_labels`: Sampled labels.
      - `full_data`: All data.
    - **Output**:
      - `clf`: Classifier trained on the sampled labels.
- **`ec_filter`**: Uses the classifier to decide whether uncleaned tuples are dirty or clean.
    - **Inputs**:
      - `dirtyex`: Indices of dirty data.
      - `full_data`: All data.
      - `clf`: Classifier for labeling dirty rows.
      - `t`: Confidence threshold; tuples above this threshold are considered clean.
    - **Output**:
      - Subset of `dirtyex` whose classification confidence is below the threshold `t`.

## Pipeline

1. **Sampling**:
   - Call `random` to sample from the remaining uncleaned data.

2. **Cleaning the sample**:
   - Read the clean version of the sampled rows directly from the dataset.

3. **Training the classifier**:
   - Call `error_classifier` to train a classifier that distinguishes clean from dirty rows.

4. **Using the classifier to prune the cleaning workload**:
   - Call `ec_filter` on the remaining uncleaned data. Rows confidently classified as clean are excluded from subsequent sampling rounds.

5. **Updating the model**:
   - Call `partial_fit` to update the model on the currently cleaned rows.
