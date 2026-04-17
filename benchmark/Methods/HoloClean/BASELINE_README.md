# HoloClean — Probabilistic-Inference-Based Data Cleaning

## Upstream Information
- **Paper**: HoloClean: Holistic Data Repairs with Probabilistic Inference (VLDB 2017)
- **GitHub**: https://github.com/HoloClean/holoclean
- **Website**: http://www.holoclean.io

## Method Description
HoloClean is a statistical inference engine that builds a probabilistic model for data cleaning by combining multiple signals — quality rules, value correlations, reference data, and more.

## Installation

### 1. PostgreSQL (required)

#### Windows
Download the installer from https://www.postgresql.org/download/windows/.

#### Ubuntu
```bash
apt-get install postgresql postgresql-contrib
```

#### Docker
```bash
docker run --name pghc \
    -e POSTGRES_DB=holo \
    -e POSTGRES_USER=holocleanuser \
    -e POSTGRES_PASSWORD=abcd1234 \
    -p 5432:5432 \
    -d postgres:11
```

### 2. Configure the database

```sql
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
```

### 3. Python dependencies

```bash
pip install psycopg2-binary peewee torch
pip install -r Methods/HoloClean/requirements.txt
```

## Usage

### Via the wrapper
```python
from Methods.HoloClean.holoclean_wrapper import HoloCleanWrapper

wrapper = HoloCleanWrapper(
    db_name='holo',
    db_user='holocleanuser',
    db_password='abcd1234'
)

df, info = wrapper.clean(
    dirty_path='Data/beers/dirty.csv',
    dc_path='Data/beers/constraints.txt'  # constraints file
)
```

## Constraint File Format

Denial Constraints (DC) syntax:
```
t1&t2&EQ(t1.brewery_id,t2.brewery_id)&IQ(t1.brewery_name,t2.brewery_name)
```

## Differences from the Upstream Implementation

**None** — the wrapper packages the upstream implementation.

## Notes

1. PostgreSQL must be running
2. The first run creates auxiliary tables in the database
3. Large datasets may require increased database and Python memory limits
