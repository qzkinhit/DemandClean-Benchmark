# GTE-large 模型下载说明

CtxPipe 需要 GTE-large 嵌入模型来提取数据表的上下文向量。

## 模型信息

- **模型名称**: thenlper/gte-large
- **模型大小**: ~670MB
- **输出维度**: 1024

## 下载方法

### 方法1: Python 脚本下载（推荐，使用国内镜像）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="thenlper/gte-large",
    local_dir="./gte-large",
    local_dir_use_symlinks=False
)
print("下载完成!")
```

### 方法2: 命令行下载

```bash
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download thenlper/gte-large --local-dir ./gte-large
```

### 方法3: Git Clone（需要 Git LFS）

```bash
# 安装 Git LFS
git lfs install

# 克隆仓库（国内镜像）
git clone https://hf-mirror.com/thenlper/gte-large
```

## 下载后的目录结构

```
embed/
└── gte-large/
    ├── config.json
    ├── model.safetensors      # ~670MB（主模型文件）
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.txt
    └── ...
```

## 验证安装

```python
from transformers import AutoTokenizer, AutoModel

model_path = "./gte-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
# 输出: 模型参数量: 335,141,888 (约335M)
```

## 国内镜像源

- **HF-Mirror**: https://hf-mirror.com （推荐）
- **ModelScope**: https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large

## 注意事项

1. 下载完成后确保 `gte-large` 文件夹直接位于 `embed/` 目录下
2. 路径应为: `Methods/ctxpipe/embed/gte-large/`
3. 如果下载失败，可以多试几次或更换镜像源
