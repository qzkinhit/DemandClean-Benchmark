import pandas as pd
from pathlib import Path

path = Path("./data/soilmoisture/clean_with_index.csv")

df = pd.read_csv(path, encoding="utf-8-sig")

rows, cols = df.shape
print(f"Table dimensions: {rows} rows * {cols} columns")

cols_text = "[" + ",".join(df.columns.astype(str)) + "]"
print(cols_text)