import pandas as pd
from pathlib import Path

path = Path("D:/桌面/New Life/钱泽凯学长项目/ZzhProject/data/soilmoisture/clean_with_index.csv")

df = pd.read_csv(path, encoding="utf-8-sig")

rows, cols = df.shape
print(f"表格维度: {rows} 行 * {cols} 列")

cols_text = "[" + ",".join(df.columns.astype(str)) + "]"
print(cols_text)