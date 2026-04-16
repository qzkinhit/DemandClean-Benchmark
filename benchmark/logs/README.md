# Logs 目录说明

本目录存放各个清洗方法运行时的日志文件。

## 命名规范

```
{method}_{dataset}_{timestamp}.log
```

### 示例
- `simpleimputer_beers_20260113_010830.log` - SimpleImputer在beers数据集上的运行日志
- `baran_adult_20260113_020000.log` - Baran在adult数据集上的运行日志
- `uniclean_hospital_20260113_030000.log` - UniClean在hospital数据集上的运行日志

## 方法名称对照

| 方法缩写 | 完整名称 | 说明 |
|---------|---------|------|
| simpleimputer | SimpleImputer | 简单统计插补 |
| mlimputer | MLImputer | 机器学习插补 |
| deleteall | DeleteAll | 删除缺失行 |
| donothing | DoNothing | 不做处理 |
| baran | Baran_Raha | Raha检测+Baran修复 |
| horizon | Horizon | 功能依赖模式选择 |
| holoclean | HoloClean | 概率图模型清洗 |
| activeclean | ActiveClean | 模型导向迭代清洗 |
| boostclean | BoostClean | 检测-修复器集成 |
| ctxpipe | CtxPipe | 上下文感知数据准备 |
| uniclean | UniClean | 多信号融合清洗 |
| lopster | Lopster | 潜在空间表示学习 |

## 数据集名称对照

| 数据集 | 任务类型 | 来源 |
|-------|---------|------|
| adult | 分类 | UCI |
| beers | 回归 | Kaggle |
| bike | 回归 | UCI |
| breast_cancer | 分类 | UCI |
| har | 分类 | UCI |
| mercedes | 回归 | Kaggle |
| nasa | 分类 | NASA |
| smartfactory | 分类 | 工业数据 |
| soilmoisture | 回归 | 传感器数据 |

## 日志内容

每个日志文件通常包含：
1. 运行开始时间
2. 输入数据信息（路径、行数、列数）
3. 清洗过程详情
4. 清洗结果统计
5. 运行结束时间和总耗时
