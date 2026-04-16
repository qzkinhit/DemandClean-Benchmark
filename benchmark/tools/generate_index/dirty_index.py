import pandas as pd
from pathlib import Path

in_path = Path("./data/soilmoisture/dirty.csv")
out_path = Path("./data/soilmoisture/dirty_with_index.csv")

df = pd.read_csv(in_path)

df["index"] = range(1, len(df) + 1)

df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"已生成文件：{out_path.resolve()}")
