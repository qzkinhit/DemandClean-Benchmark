# Lopster - 潜在空间数据清洗

## 官方信息
- **论文**: Generalizable Data Cleaning of Tabular Data in Latent Space (VLDB 2024)
- **GitHub**: https://github.com/DataManagementLab/data_cleaning_with_latent_operators
- **作者**: Eduardo dos Reis, Mohamed Abdelaal, Carsten Binnig

## 方法描述
Lopster通过学习数据的潜在空间表示来检测和修复错误，是一种基于VAE的通用数据清洗方法。

## 真值使用情况

**Type 2 - 需要训练数据**

- Lopster使用 `clean.csv` 训练VAE模型来学习数据的潜在空间表示
- **真值成本 = clean.csv的行数**（全部数据用于训练）
- 这意味着Lopster需要一定量的干净数据来学习"正常"数据的分布

## 依赖安装

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib
```

或使用本目录的requirements.txt:
```bash
pip install -r Methods/Lopster/requirements.txt
```

## 数据格式要求

**重要**: Lopster官方实现要求特定的数据格式：

```
Data/{dataset_name}/
├── clean.csv      # 干净数据（用于训练）
└── dirty01.csv    # 脏数据（注意是dirty01.csv，不是dirty.csv）
```

## 配置文件

需要在 `dataset_configuration.json` 中配置数据集信息。参考现有配置。

## 运行方式

### 方式1: 使用wrapper
```python
from Methods.Lopster.lopster_wrapper import LopsterWrapper, prepare_data_for_lopster

# 准备数据格式
prepare_data_for_lopster(
    'Data/beers/dirty.csv',
    'Data/beers/clean.csv',
    'beers',
    'Data/'
)

# 运行清洗
wrapper = LopsterWrapper(epochs=100, latent_dim=120)
df, info = wrapper.clean('beers', 'Data/')
```

### 方式2: 直接运行官方脚本
```bash
python Methods/Lopster/lopster.py --dataset beers --path Data/ --epochs 100 --latent 120
```

## 与官方实现的差异

**无差异** - wrapper仅封装官方实现，不含任何简化版本。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| learning_rate | 0.001 | 学习率 |
| latent | 120 | 潜在空间维度 |
| batch_size | 256 | 批大小 |
| K | 12 | 翻译操作参数 |

## 输出

清洗后的数据保存到 `{path}/{dataset}/lopster.csv`
