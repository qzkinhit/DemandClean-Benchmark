# Results 目录说明

本目录存放各个清洗方法的运行结果。

## 目录结构

```
results/
├── simpleimputer/          # SimpleImputer结果
│   ├── {dataset}_repaired.csv
│   └── {dataset}_summary.txt
├── mlimputer/              # MLImputer结果
├── deleteall/              # DeleteAll结果
├── donothing/              # DoNothing结果
├── baran/                  # Baran_Raha结果
├── horizon/                # Horizon结果
├── holoclean/              # HoloClean结果
├── activeclean/            # ActiveClean结果
├── boostclean/             # BoostClean结果
├── ctxpipe/                # CtxPipe结果
├── uniclean/               # UniClean结果
└── lopster/                # Lopster结果
```

## 文件命名规范

### 修复后数据
```
{dataset}_repaired.csv
```

### 结果摘要
```
{dataset}_summary.txt
```

### 评估结果
```
{dataset}_evaluation.json
```

## 结果文件内容

### _repaired.csv
清洗/修复后的数据，与原始脏数据格式相同。

### _summary.txt
包含:
- 执行时间
- 执行状态
- 清洗方法参数
- 修复/删除统计
- 真值使用成本

### _evaluation.json (如有)
包含getScoreML统一评估结果:
- 传统清洗指标 (accuracy, recall, f1, edr等)
- 下游任务性能 (分类/回归/聚类)
- 模型容忍度指标

## 真值使用类型说明

| Type | 说明 | 代表方法 |
|------|------|---------|
| Type 1 | 全自动，无需人工参与 | SimpleImputer, MLImputer, HoloClean |
| Type 2 | 需少量验证集评估效果 | BoostClean, Baran |
| Type 3 | 迭代交互式标注 | ActiveClean |
