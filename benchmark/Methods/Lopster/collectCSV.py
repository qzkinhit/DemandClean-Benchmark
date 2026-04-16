import pandas as pd    
from pathlib import Path

p = Path('./repair')

dfs_list = []

for fname in p.glob('**/*.csv'):
    #print(fname.as_posix())
    parent = fname.parent.parent.name
    df = pd.read_csv(fname)
    df.insert(0, 'detector', parent)
    dfs_list.append(df)
    ## df.to_csv(fname, index=False)
    if fname.parent.name == "standardImputer":
        df.tool_name = 'standardImputer-' + df.tool_name
    df.tool_name = df.tool_name.replace(to_replace='_', regex=True, value='-')

final = pd.concat(dfs_list)
final.rename(columns={"tool_name": "cleaner"}, inplace= True)
final.to_csv("rein_soccer_PLAYER_cleaning_results.csv", index=False)
