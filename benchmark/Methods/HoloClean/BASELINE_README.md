# HoloClean - 基于概率推断的数据清洗

## 官方信息
- **论文**: HoloClean: Holistic Data Repairs with Probabilistic Inference (VLDB 2017)
- **GitHub**: https://github.com/HoloClean/holoclean
- **网站**: http://www.holoclean.io

## 方法描述
HoloClean是一个统计推断引擎，通过结合质量规则、值相关性、参考数据等多种信号构建概率模型进行数据清洗。

## 依赖安装

### 1. PostgreSQL数据库（必需）

#### Windows:
从 https://www.postgresql.org/download/windows/ 下载安装

#### Ubuntu:
```bash
apt-get install postgresql postgresql-contrib
```

#### Docker方式:
```bash
docker run --name pghc \
    -e POSTGRES_DB=holo \
    -e POSTGRES_USER=holocleanuser \
    -e POSTGRES_PASSWORD=abcd1234 \
    -p 5432:5432 \
    -d postgres:11
```

### 2. 配置数据库

```sql
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
```

### 3. Python依赖

```bash
pip install psycopg2-binary peewee torch
pip install -r Methods/HoloClean/requirements.txt
```

## 运行方式

### 使用wrapper
```python
from Methods.HoloClean.holoclean_wrapper import HoloCleanWrapper

wrapper = HoloCleanWrapper(
    db_name='holo',
    db_user='holocleanuser',
    db_password='abcd1234'
)

df, info = wrapper.clean(
    dirty_path='Data/beers/dirty.csv',
    dc_path='Data/beers/constraints.txt'  # 约束文件
)
```

## 约束文件格式

Denial Constraints (DC) 格式:
```
t1&t2&EQ(t1.brewery_id,t2.brewery_id)&IQ(t1.brewery_name,t2.brewery_name)
```

## 与官方实现的差异

**无差异** - wrapper封装官方实现。

## 注意事项

1. PostgreSQL服务必须运行
2. 首次运行会在数据库中创建辅助表
3. 大数据集可能需要增加数据库和Python内存配置
