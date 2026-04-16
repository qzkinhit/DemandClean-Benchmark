# 项目目录简短说明

## .streamlit
**配置文件**
- `config.toml`: streamlit 项目的配置文件。

## resources
**外部资源(可能会用到的外部模型)**
- `readme.txt`: 外部资源文件的说明文档。

## CleanLogs
**日志文件**
- 清洗日志文件夹，用于记录一键端的清洗日志。

## AnalyticsCache
**分析与缓存,分析和处理各模块直接的输入输出，起到缓存区的作用**
- **ModuleTest**:模块测试代码
- `get_cleaner_excute_info.py`: 基于规则给算子分类权重,获取算子的执行顺序。
- `get_planuml_graph.py`: 利用 plantuml可视化执行顺序图。
- `cleaner_associations_cycle.py`: 在有环图上合并顶点，对合并后的顶点获取算子关联性。
- `handle_rules.py`: 规则转换,分析规则，将挖掘的规则实例化到 Spark 上。

## CoreSetSample
**核心集抽样**
- **ModuleTest**:模块测试代码
- `get_patition_block.py`: 分块算法。
- `distri_samplify.py`: 利用数据分布对数据进行抽样。
- `handle_data_distri.py`: 计算数据之间的分布。
- `mapping_samplify.py`: 利用映射关系采样。

## SampleScrubber
**样本清理工具**
- **ModuleTest**:模块测试代码
- **util**
    - `distance.py`: 计算距离。
    - `getNum.py`: 计算清洗准确度。
- `uniop_model.py`: 规则挖掘模型。
- `param_builder.py`: 规则参数构建。
- `param_selector.py`: 规则参数选择。
- **cleanOps**
    - `single.py`: 单属性算子。
    - `multiple.py`: 多属性关联算子。
    - `soft.py`: 其他尝试性算子。
    - `clean_penalty.py`: 计算清洗成本，包括edit惩罚，语义距离惩罚，jaccard距离。

## SparkClean
**Spark 清理模块**
- **ModuleTest**:模块测试代码
- - **util**
    - `distance.py`: 计算距离。
    - `get_types.py`: 获取数据类型。
    - `cleanudf.py`: Spark 用户自定义 UDF。
- `get_spark_rule.py`: 规则处理和分配参数
- `spark_rule_model.py`: 适用于 Spark 环境的操作模型。
- `selector.py`: 参数选择。
- `cleaner_model.py`: 算子的数据结构。
- `function_denpendency.py`: 依赖cleaner。

## TestDataset
**测试数据集**

## sysFlowVisualizer
**可视化工具**
- **cleanCache**: 可视化展示的数据和图表缓存区。
- **PageHistory**: 一些弃用的可视化方案。

## pages
**Streamlit 网页**
- Streamlit 的可视化网页前端代码。

## 主要脚本
- `Welcome.py`: Streamlit 前端启动一键端，前端首页代码。
- `main.py`: 利用终端一键端清洗的实例。
- `logsetting.py`: 一键端日志配置。
- `Clean.py`: 一键端终端清洗的代码。
- `requirements.txt`: 一键端依赖包。
- `Plantuml.svg`: 可视化清洗流程图。

# 下一步优化方向
- [ ] 更多样清洗算子
- [ ] 更多样检测器
- [ ] 更多样的数据