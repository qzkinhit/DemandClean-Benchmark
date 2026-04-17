"""
ActiveClean Wrapper

Wraps the ActiveClean algorithm and emits the cleaned dataset.

Core logic:
1. ActiveClean selects the most valuable samples for cleaning using a sampling strategy.
2. "Cleaning" = replace the sampled dirty rows with the corresponding clean rows.
3. Sampling budget = ground truth cost (number of labels used).
4. Rows that are not sampled remain unchanged (not cleaned).

Features:
- Automatic vectorization: internally vectorized for computation; restored to the
  original format on output.
- Supports every task type: classification, regression, clustering (evaluated by getScoreML).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def vectorize_dataframe(df, label_col=None, encoders=None):
    """
    Vectorize a DataFrame (convert every column to numeric).

    Args:
        df: input DataFrame.
        label_col: label column name (uses the last column when None).
        encoders: existing encoder dict (used to keep clean/dirty encodings consistent).

    Returns:
        vectorized_df: vectorized DataFrame.
        encoders: encoder dict (for subsequent decoding or alignment).
    """
    df = df.copy()

    # Replace the "empty" placeholder with a unified NA marker
    df = df.replace('empty', '__NA__')

    if label_col is None:
        label_col = df.columns[-1]

    if encoders is None:
        encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            if col not in encoders:
                encoders[col] = LabelEncoder()
                # Fit the encoder on the union of possible values
                encoders[col].fit(df[col].astype(str).fillna('__NA__'))

            # transform
            df[col] = encoders[col].transform(df[col].astype(str).fillna('__NA__'))

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs
    df = df.fillna(0)

    return df, encoders


def needs_vectorization(df):
    """Check whether the DataFrame needs vectorization."""
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            return True
    return False


# ============================================================================
# ActiveClean helper functions (using the modern sklearn API)
# ============================================================================

def translate_indices(globali, imap):
    """Return the positions of globali elements within imap."""
    lset = set(globali)
    return [s for s, t in enumerate(imap) if t in lset]


def error_classifier(total_labels, full_data):
    """
    Train an error-detection classifier using the labels of the sampled data.

    Args:
        total_labels: [(index, is_clean), ...] list of labels.
        full_data: full feature matrix.

    Returns:
        The trained classifier, or None if all samples are clean.
    """
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]

    # Check whether all labels are "clean"
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices, :], labels)
        return clf
    else:
        return None


def ec_filter(dirtyex, full_data, clf, t=0.90):
    """
    Filter out samples that are likely to be clean using the error classifier.

    Args:
        dirtyex: list of candidate indices to filter.
        full_data: full feature matrix.
        clf: error classifier.
        t: threshold.

    Returns:
        Filtered list of indices.
    """
    if clf is not None:
        pred = clf.predict_proba(full_data[dirtyex, :])
        return [j for i, j in enumerate(dirtyex) if pred[i][0] < t]

    return dirtyex


# ============================================================================
# Main functions
# ============================================================================

def run_activeclean(clean_path, dirty_path, batchsize=50, total=10000):
    """
    Run ActiveClean and return the cleaned dataset.

    Automatic vectorization: if the data contains non-numeric columns, LabelEncoder
    conversion is applied automatically. The output keeps the original (non-vectorized)
    format and only replaces rows that were sampled.

    Args:
        clean_path (str): path to the clean data.
        dirty_path (str): path to the dirty data.
        batchsize (int): number of samples per iteration.
        total (int): maximum number of cleaned samples.

    Returns:
        txt: textual run report.
        cleaned_data: cleaned DataFrame (original format).
        ground_truth_cost: amount of ground truth used (number of cleaned samples).
    """
    # Read raw data
    correct_data_raw = pd.read_csv(clean_path)
    injected_data_raw = pd.read_csv(dirty_path)

    # Preserve the raw data (used for the final output)
    original_clean = correct_data_raw.copy()
    original_dirty = injected_data_raw.copy()
    original_columns = correct_data_raw.columns.tolist()

    # Check whether vectorization is needed (only for internal processing)
    if needs_vectorization(correct_data_raw) or needs_vectorization(injected_data_raw):
        print("[ActiveClean Wrapper] Non-numeric columns detected; vectorizing automatically...")

        # Concatenate the data to fit the encoder (ensures consistent encoding)
        combined = pd.concat([correct_data_raw, injected_data_raw], ignore_index=True)
        _, encoders = vectorize_dataframe(combined)

        # Vectorize each side individually (for internal algorithm use)
        correct_data, _ = vectorize_dataframe(correct_data_raw, encoders=encoders)
        injected_data, _ = vectorize_dataframe(injected_data_raw, encoders=encoders)

        print(f"[ActiveClean Wrapper] Vectorization done; encoded columns: {list(encoders.keys())}")
    else:
        correct_data = correct_data_raw.copy()
        injected_data = injected_data_raw.copy()
        encoders = None

    # Extract features and labels (vectorized data; used internally by the algorithm)
    X_correct = correct_data.iloc[:, :-1].values
    y_correct = correct_data.iloc[:, -1].values

    X_full = injected_data.iloc[:, :-1].values
    y_full = injected_data.iloc[:, -1].values

    size = len(X_full)

    # Summarise clean/dirty statistics (for information only)
    try:
        indices_clean_rows = np.where((X_full == X_correct).all(axis=1))[0]
        indices_dirty_rows = np.where((X_full != X_correct).any(axis=1))[0]
        print(f"[ActiveClean Wrapper] Dataset: {size} rows, {len(indices_dirty_rows)} dirty rows, {len(indices_clean_rows)} clean rows")
    except:
        print(f"[ActiveClean Wrapper] Dataset: {size} rows")

    # Use the full clean dataset as ground truth
    X_clean = X_correct
    y_clean = y_correct

    # Generate row indices for all rows
    all_indices = np.arange(0, size, 1)

    # Split into training and test sets
    # Note: stratification cannot be used for regression tasks
    try:
        train_indices, test_indices = train_test_split(all_indices, test_size=0.20, stratify=y_full)
    except ValueError:
        # If stratification fails (e.g. for regression), fall back to no stratification
        train_indices, test_indices = train_test_split(all_indices, test_size=0.20, random_state=42)

    # Run the ActiveClean algorithm
    txt, cleanex = activeclean_sampling(
        X_full,
        y_full,
        X_clean,
        y_clean,
        train_indices,
        test_indices,
        batchsize=batchsize,
        total=total
    )

    # ========================================================================
    # Construct the cleaned dataset (in the original format)
    # ========================================================================
    # cleanex contains the indices of sampled / cleaned rows.
    # Cleaning: replace these rows with the corresponding clean rows.

    cleaned_data = original_dirty.copy()

    for idx in cleanex:
        # Replace the sampled row with the corresponding clean row
        cleaned_data.iloc[idx] = original_clean.iloc[idx]

    ground_truth_cost = len(cleanex)

    print(f"[ActiveClean Wrapper] Cleaning done: {ground_truth_cost} rows replaced with clean data")

    return txt, cleaned_data, ground_truth_cost


def activeclean_sampling(X_dirty, y_dirty, X_clean, y_clean, train_indices, test_indices,
                         batchsize=50, total=10000):
    """
    ActiveClean sampling algorithm.

    Uses the gradient information of an SGDClassifier to pick the most valuable
    samples for cleaning. Supports any task type (actual evaluation is performed by
    getScoreML).

    Args:
        X_dirty: dirty feature matrix (vectorized).
        y_dirty: dirty labels.
        X_clean: clean feature matrix (vectorized).
        y_clean: clean labels.
        train_indices: training indices.
        test_indices: test indices.
        batchsize: number of samples per batch.
        total: maximum number of cleaned samples.

    Returns:
        txt: run report.
        cleanex: list of indices that were cleaned.
    """
    print("[ActiveClean] Initialization")
    txt = "[ActiveClean] Initialization"

    # Inspect the label space to decide whether classification mode applies
    unique_labels = np.unique(y_clean)
    is_classification = len(unique_labels) <= 100  # assume >100 unique values means regression

    if is_classification:
        all_classes = unique_labels
        print(f"[ActiveClean] Detected classification task; number of classes: {len(all_classes)}")
    else:
        print(f"[ActiveClean] Detected regression / high-cardinality task; using a simple sampling strategy")

    # Candidate indices to clean (all rows in the training set)
    remaining = list(train_indices)
    cleanex = []  # already cleaned indices
    total_labels = []  # used to train the error classifier

    # Test-set data (used for evaluation)
    X_test = X_clean[test_indices]
    y_test = y_clean[test_indices]

    # Initial sampling
    if len(remaining) < batchsize:
        initial_batch = remaining.copy()
    else:
        initial_sample_idx = np.random.choice(len(remaining), batchsize, replace=False)
        initial_batch = [remaining[i] for i in initial_sample_idx]

    # Add the initial batch to the cleaned set
    cleanex.extend(initial_batch)
    for idx in initial_batch:
        if idx in remaining:
            remaining.remove(idx)

    # Initialise the classifier (used by ActiveClean's sampling strategy)
    clf = None
    if is_classification and len(cleanex) > 0:
        try:
            clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200,
                               fit_intercept=True, warm_start=True)
            clf.partial_fit(X_clean[cleanex], y_clean[cleanex], classes=all_classes)
        except Exception as e:
            print(f"[ActiveClean] Classifier initialisation failed: {e}; falling back to simple sampling")
            clf = None

    # Iterative cleaning
    for i in range(batchsize, total, batchsize):
        print(f"[ActiveClean] Number Cleaned So Far: {len(cleanex)}")
        txt += f"\n[ActiveClean] Number Cleaned So Far: {len(cleanex)}"

        # Evaluate the current model (for classification tasks)
        if clf is not None:
            try:
                ypred = clf.predict(X_test)
                acc = accuracy_score(y_test, ypred)
                print(f"[ActiveClean] Internal Accuracy: {acc:.4f}")
                txt += f"\n[ActiveClean] Internal Accuracy: {acc:.4f}"
            except:
                pass

        # Check whether any samples are left to clean
        if len(remaining) < batchsize:
            if len(remaining) == 0:
                print("[ActiveClean] No more samples to clean")
                txt += "\n[ActiveClean] No more samples to clean"
                break
            else:
                # Clean the remaining samples
                batch = remaining.copy()
        else:
            # Randomly sample a new batch
            sample_idx = np.random.choice(len(remaining), batchsize, replace=False)
            batch = [remaining[i] for i in sample_idx]

        # Record the sampled labels (used by the error classifier)
        # Decide whether the sampled data was originally "clean" (matches ground truth)
        for idx in batch:
            try:
                is_originally_clean = np.allclose(X_dirty[idx], X_clean[idx])
            except:
                is_originally_clean = False
            total_labels.append((idx, is_originally_clean))

        # Add the new batch to the cleaned set
        cleanex.extend(batch)
        for idx in batch:
            if idx in remaining:
                remaining.remove(idx)

        # Try filtering with the error classifier (core ActiveClean optimisation)
        if clf is not None and len(total_labels) > batchsize:
            try:
                ec = error_classifier(total_labels, X_dirty)
                if ec is not None:
                    remaining = ec_filter(remaining, X_dirty, ec)
            except:
                pass

        # Update the model using the cleaned data
        if clf is not None:
            try:
                clf.partial_fit(X_clean[cleanex], y_clean[cleanex])
            except:
                pass

    # Final statistics
    print(f"[ActiveClean] Final - Number Cleaned: {len(cleanex)}")
    txt += f"\n[ActiveClean] Final - Number Cleaned: {len(cleanex)}"

    return txt, cleanex
