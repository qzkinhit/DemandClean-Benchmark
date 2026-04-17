import pandas as pd

# Load the clean data subset and dirty data
clean_data_path = '../Data/5_tax/subset_clean_index_10k.csv'
dirty_data_path = '../Data/5_tax/dirty_index.csv'

clean_data = pd.read_csv(clean_data_path)
dirty_data = pd.read_csv(dirty_data_path)

# Ensure the index column is of string type to avoid potential type mismatches
clean_data['index'] = clean_data['index'].astype(str)
dirty_data['index'] = dirty_data['index'].astype(str)

# Perform an inner join on the index column
dirty_subset = dirty_data.merge(clean_data[['index']], on='index', how='inner')

# Save the result to a file
dirty_subset_path = '../Data/5_tax/subset_dirty_index_10k.csv'
dirty_subset.to_csv(dirty_subset_path, index=False)

print(f"Dirty data subset extracted and saved to {dirty_subset_path}")
