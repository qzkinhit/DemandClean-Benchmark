# HoloClean Wrapper 说明

## 概述
HoloClean是基于概率图模型的数据清洗系统，发表于VLDB 2017。

## 依赖要求
- **PostgreSQL 9.4+** (必需)
- PyTorch
- psycopg2

## PostgreSQL 配置

### 方式1: 本地安装
```sql
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
```

### 方式2: Docker
```bash
docker run --name pghc \
    -e POSTGRES_DB=holo -e POSTGRES_USER=holocleanuser -e POSTGRES_PASSWORD=abcd1234 \
    -p 5432:5432 \
    -d postgres:11
```

## 使用方式

```python
from Methods.HoloClean.holoclean_wrapper import HoloCleanWrapper

wrapper = HoloCleanWrapper(
    db_user="holocleanuser",
    db_pwd="abcd1234",
    db_host="localhost",
    db_name="holo"
)

repaired_df, info = wrapper.clean(
    dirty_path="data/dirty.csv",
    dc_path="data/constraints.txt",  # 可选
    output_path="results/repaired.csv"
)
```

## Baseline项目修改说明
- 添加了 `holoclean_wrapper.py` 统一接口
- 修复了Python 3.9兼容性问题

## 注意事项
- 由于需要PostgreSQL，本方法在无数据库环境下无法运行
- 建议使用Docker快速部署测试环境
