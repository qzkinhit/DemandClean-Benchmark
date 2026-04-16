import pandas as pd
from pathlib import Path

in_path = Path("D:/桌面/New Life/钱泽凯学长项目/ZzhProject/data/soilmoisture/clean.csv")
out_path = Path("D:/桌面/New Life/钱泽凯学长项目/ZzhProject/data/soilmoisture/clean_with_index.csv")

df = pd.read_csv(in_path)

df["index"] = range(1, len(df) + 1)

df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"已生成文件：{out_path.resolve()}")
