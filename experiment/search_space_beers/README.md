# Search-space visualization

> Validate DemandClean's theoretical search space for cleaning strategies: map the performance distribution of all possible strategies onto the Authenticity-Diversity plane.

---

## Goal

Validate the theoretical model of the cleaning-strategy search space on the Beers (IPA) dataset:
1. **Four extreme points**: NoFix, FullFix, DeleteFix, RelaxFix.
2. **Random sampling**: sweep all action-probability combinations (no_action, repair, delete, replace_nearby).
3. **DQN inference**: mark DemandFix inside the search space.
4. **Heatmap**: show how accuracy is distributed over the search space.

### Theoretical model

```
Diversity ^
          |
FullFix  *-----------------*  DeleteFix (diversity lower bound)
          |\   search space |
          | \   (crescent)  |
          |  \             /
          |   * DemandFix /
          |    \         /
          |     \       /
NoFix  *--+------*-----+----> Authenticity
          |     RelaxFix
```

## Layout

```
search_space_beers/
├── run_search_space.py      # main experiment script (uses DemandClean release API)
├── README.md                # this file
├── datasets/                # symlink -> ../ablation_beers/datasets/beers/
├── result/                  # experiment outputs
│   ├── search_space_scatter.png      # scatter (color = performance)
│   ├── search_space_heatmap.png      # gridded heatmap
│   ├── search_space_combined.png     # combined figure
│   ├── search_space_cost_heatmap.png # cost heatmap
│   ├── search_space_cost_scatter.png # cost scatter
│   ├── repair_vs_nonrepair.png       # repair vs delete+nearby heatmap
│   ├── action_pairs/                 # action-pair heatmaps
│   ├── search_space_results.csv      # sampled results
│   ├── dqn_distribution.csv          # DQN inference results
│   ├── extreme_points.json           # extreme-point coordinates
│   └── dirty_data.csv               # data after error injection
└── model/                   # DQN model
```

## Usage

```bash
# Run from the project root
cd /path/to/TolerDM

# Re-plot from existing results (recommended)
python experiment/search_space_beers/run_search_space.py

# Re-run the full experiment (slow);
# set RUN_EXPERIMENT = True in the script first.
python experiment/search_space_beers/run_search_space.py
```

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| `N_RANDOM_SAMPLES` | 1000 | number of random samples |
| `N_DQN_RUNS` | 1 | DQN inference runs |
| `SEMANTIC_RATE` | 0.15 | semantic-error injection rate |
| `SYNTACTIC_RATE` | 0.20 | syntactic-error injection rate |
| `RUN_EXPERIMENT` | False | True = run experiment, False = re-plot from existing results |
| `LOAD_MODEL` | True | True = load a trained model, False = retrain the DQN |

## Visual outputs

### 1. Scatter (`search_space_scatter.png`)
- X: Authenticity. Y: Diversity.
- Color: classification accuracy (warm = high, cool = low).
- Star: DQN DemandFix position.

### 2. Gridded heatmap (`search_space_heatmap.png`)
- Search space partitioned into a grid.
- Each cell's color is the mean accuracy.

### 3. Combined (`search_space_combined.png`)
- Scatter + heatmap + extreme-point annotations.

### 4. Cost heatmap (`search_space_cost_heatmap.png`)
- Distribution of ground-truth budget use across the Auth-Div plane.

### 5. Cost scatter (`search_space_cost_scatter.png`)
- Cost distribution without interpolation.

### 6. Repair vs. non-repair (`repair_vs_nonrepair.png`)
- Heatmap of repair_value fraction vs. (delete + replace_nearby) fraction.

### 7. Action-pair heatmaps (`action_pairs/`)
- Performance heatmaps for every pair of actions.

## Extra dependencies

```bash
pip install alphashape shapely
```

## Key metrics

### Authenticity
```
Auth = # correct values / # total values after cleaning
```
How close the cleaned data is to the clean reference.

### Diversity
```
Div = sample retention * variance retention
```
How much information the cleaned data preserves.

## History

This experiment was migrated from `history/experiments/pre_exp_ablation/beers_ipa_experiment/history_logs/history/search_space_experiment.py`. Main changes:
1. DQN training/inference replaced with the DemandClean release API (drops the TensorFlow dependency).
2. All visualization functions kept intact; logic and style unchanged.
3. Random-sampling logic kept intact.
4. File save paths updated to match the current layout.
