# HoloClean Wrapper

## Overview
HoloClean is a probabilistic-graphical-model-based data cleaning system published at VLDB 2017.

## Requirements
- **PostgreSQL 9.4+** (required)
- PyTorch
- psycopg2

## PostgreSQL Setup

### Option 1: Local installation
```sql
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
```

### Option 2: Docker
```bash
docker run --name pghc \
    -e POSTGRES_DB=holo -e POSTGRES_USER=holocleanuser -e POSTGRES_PASSWORD=abcd1234 \
    -p 5432:5432 \
    -d postgres:11
```

## Usage

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
    dc_path="data/constraints.txt",  # optional
    output_path="results/repaired.csv"
)
```

## Baseline Project Modifications
- Added `holoclean_wrapper.py` as a unified interface
- Fixed Python 3.9 compatibility issues

## Notes
- Requires PostgreSQL; this method cannot run without a database
- Docker is recommended for a quick test environment
