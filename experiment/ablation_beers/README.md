# Beers (IPA) ablation study

> A small-scale ablation on the Beer dataset (ABV, IBU): 5 baseline strategies vs. 12 DQN strategy variants.

---

## Goal

On a low-dimensional (2-feature) Beer dataset, systematically compare cleaning performance across DemandClean variants:
1. **Baseline comparison**: NoFix / OverFix / RelaxFix / FullFix / DemandFix.
2. **DQN architecture**: plain DQN vs. Dueling Double DQN.
3. **Decision staging**: single-stage vs. two-stage.
4. **Detector**: Oracle (known errors) vs. Auto (automatic detection).
5. **Inference mode**: single-phase vs. two-phase.

## Layout

```
ablation_beers/
├── run_ablation.py          # main script (uses DemandClean release API)
├── README.md                # this file
├── datasets/beers/          # Beer dataset
│   ├── clean.csv            # clean data
│   ├── dirty.csv            # dirty data
│   └── README.md
├── result/                  # outputs (boundary plots + training curves)
│   ├── {strategy}.png       # decision-boundary plot per strategy
│   ├── dqn_*_training.png   # DQN training curves
│   └── detector.pkl         # detector cache
└── model/                   # trained PyTorch models (.pt)
```

## Usage

```bash
# Run from the project root
cd /path/to/TolerDM

# Default: run every strategy (400 training episodes)
python experiment/ablation_beers/run_ablation.py

# Load saved models (skip training, infer only)
python experiment/ablation_beers/run_ablation.py --load_model

# Custom training episodes
python experiment/ablation_beers/run_ablation.py --n_episodes 100

# Run a subset of strategies
python experiment/ablation_beers/run_ablation.py --strategies NoFix,FullFix,FullUnsup_Dueling_Single
```

## Strategies (17 total)

### Baselines (5)
| Strategy | Description |
|--------|------|
| NoFix | drop only rows with missing values; keep other errors |
| OverFix | drop every detected-error row |
| RelaxFix | fill every error with a KNN neighbor value |
| FullFix | repair every error with ground truth (highest cost) |
| DemandFix | repair on demand: ground truth near the boundary, KNN/delete elsewhere |

### DQN strategies (12)
| Strategy | Detector | Agent | Inference |
|--------|--------|-----------|---------|
| DQN_Single | oracle | single (plain) | single_phase |
| DQN_TwoStage | oracle | two_stage (plain) | single_phase |
| SemiSup_Single | oracle | single (plain) | single_phase |
| SemiSup_TwoStage | oracle | two_stage (plain) | single_phase |
| FullUnsup_Single | auto | single (plain) | single_phase |
| FullUnsup_TwoStage | auto | two_stage (plain) | single_phase |
| FullUnsup_Single_2P | auto | single (plain) | two_phase |
| FullUnsup_TwoStage_2P | auto | two_stage (plain) | two_phase |
| SemiSup_Dueling_Single | oracle | dueling_single | single_phase |
| SemiSup_Dueling_TwoStage | oracle | dueling_two_stage | single_phase |
| FullUnsup_Dueling_Single | auto | dueling_single | single_phase |
| FullUnsup_Dueling_TwoStage | auto | dueling_two_stage | single_phase |

## Mapping to the release API

Every DQN strategy in this ablation is invoked through the `DemandClean` API:

```python
from demandclean.api.demand_clean import DemandClean

dc = DemandClean(
    task_type='classification',
    model_type='svm',
    agent_type='...',          # single / two_stage / dueling_single / dueling_two_stage
    detector_mode='...',       # oracle / auto
    inference_mode='...',      # single_phase / two_phase
    n_episodes=400,
    column_names=['abv', 'ibu'],
    label_col='is_ipa',
)
```

## Dataset notes

- **Source**: Beer Reviews Dataset, keeping rows with non-null ABV and IBU.
- **Features**: ABV (alcohol by volume), IBU (bitterness).
- **Label**: is_ipa (whether the style is IPA).
- **Error injection**: semantic (15%) + syntactic (25%) + missing (5%).
- **Boundary region**: IBU in [35, 65]; errors are concentrated here.

## Visual outputs

### Decision boundary
One plot per strategy, showing:
- SVM linear decision boundary (actual vs. ideal).
- Training data distribution (IPA vs. non-IPA).
- Accuracy, truth cost, Authenticity (Auth), Diversity (Div).

### DQN training history
Each DQN strategy produces a training curve showing reward and epsilon.

## History

Migrated from `history/experiments/pre_exp_ablation/beers_ipa_experiment/real_beers_experiment_with_detector.py`. Main changes:
1. Dropped TensorFlow dependency in favor of PyTorch (release API).
2. Consolidated 12 separate strategy functions into parameterized calls of the DemandClean API.
3. Kept the original (beers-specific) error-injection logic.
4. Kept every visualization (boundary plots, training curves).
5. Updated save paths to the current layout.
