"""
Lopster Wrapper - 潜在空间表示学习数据清洗

Lopster是VLDB 2024提出的基于潜在空间表示学习的数据清洗方法，
通过学习数据的潜在表示来检测和修复错误。

论文: Generalizable Data Cleaning of Tabular Data in Latent Space (VLDB 2024)
GitHub: https://github.com/DataManagementLab/data_cleaning_with_latent_operators

**真值使用情况**:
- Lopster使用clean.csv训练VAE模型
- 官方实现使用100%的clean.csv训练
- 本wrapper提供clean_ratio参数控制使用比例
- 分类: Type 2 - 需要训练数据

依赖: tensorflow, keras, scikit-learn, pandas, numpy, matplotlib
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

# 添加当前目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


class LopsterWrapper:
    """
    Lopster清洗方法的封装类

    直接调用官方函数，而非通过subprocess间接调用。

    核心参数:
    - latent_dim: 潜在空间维度 (默认120)
    - K: 翻译操作参数 (默认12)
    - epochs: 训练轮数 (默认100)
    - batch_size: 批大小 (默认256)
    - clean_ratio: 使用clean数据的比例 (默认1.0，即100%)

    真值使用:
    - clean_ratio=1.0: 使用全部clean数据（官方默认）
    - clean_ratio=0.1: 仅使用10%的clean数据训练
    """

    def __init__(self,
                 latent_dim: int = 120,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 K: int = 12,
                 clean_ratio: float = 1.0,
                 verbose: bool = False):
        """
        初始化Lopster包装器

        Args:
            latent_dim: 潜在空间维度（官方默认120）
            learning_rate: 学习率
            batch_size: 批大小
            epochs: 训练轮数
            K: K参数（翻译操作参数）
            clean_ratio: clean数据使用比例（0.0-1.0）
            verbose: 是否打印详细信息
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.K = K
        self.clean_ratio = min(max(clean_ratio, 0.01), 1.0)  # 限制在0.01-1.0
        self.verbose = verbose
        self.ground_truth_used = 0

    def _check_requirements(self) -> bool:
        """检查TensorFlow是否已安装"""
        try:
            import tensorflow as tf
            if self.verbose:
                print(f"TensorFlow version: {tf.__version__}")
            return True
        except ImportError:
            raise ImportError(
                "Lopster需要TensorFlow。请安装: pip install tensorflow\n"
                "详见 Methods/Lopster/requirements.txt"
            )

    def _check_data_format(self, path: str, dataset: str) -> bool:
        """检查数据格式是否符合官方要求"""
        clean_path = os.path.join(path, dataset, 'clean.csv')
        dirty_path = os.path.join(path, dataset, 'dirty01.csv')
        dirty_path_alt = os.path.join(path, dataset, 'dirty.csv')
        config_path = os.path.join(_current_dir, 'dataset_configuration.json')

        missing = []
        if not os.path.exists(clean_path):
            missing.append(f"clean.csv: {clean_path}")
        if not os.path.exists(dirty_path) and not os.path.exists(dirty_path_alt):
            missing.append(f"dirty01.csv or dirty.csv: {dirty_path}")
        if not os.path.exists(config_path):
            missing.append(f"dataset_configuration.json: {config_path}")

        if missing:
            raise FileNotFoundError(
                f"Lopster官方实现需要以下文件:\n" +
                "\n".join(f"  - {m}" for m in missing)
            )
        return True

    def clean(self,
              dataset: str,
              path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执行Lopster清洗流程 - 直接调用官方函数

        Args:
            dataset: 数据集名称 (对应 {path}/{dataset}/ 目录)
            path: 数据集根目录
            output_path: 输出路径 (可选)

        Returns:
            修复后的数据和清洗信息
        """
        self._check_requirements()
        self._check_data_format(path, dataset)

        # 导入TensorFlow和官方模块
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from datasets import (
            get_tf_database, load_regression, load_features_and_data,
            reverse_to_input_domain, get_date_columns
        )
        from latent_operators import LatentOperator
        from utils import create_and_train_LOP
        from error_detection import predict_on_enhanced

        # 加载clean数据
        clean_path = os.path.join(path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path)
        total_clean_rows = len(clean_df)

        # 计算实际使用的clean数据量
        actual_clean_rows = int(total_clean_rows * self.clean_ratio)
        self.ground_truth_used = actual_clean_rows

        if self.verbose:
            print(f"Clean数据总量: {total_clean_rows} 行")
            print(f"使用比例: {self.clean_ratio:.1%}")
            print(f"实际使用: {actual_clean_rows} 行")

        # 设置采样大小
        sample_size = actual_clean_rows if self.clean_ratio < 1.0 else -1

        # 1. 加载clean数据训练VAE
        MISSING_REPLACE = '3.0'
        T = 'missing_values'

        x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(
            dataset,
            sample_size,  # n_train_instances
            sample_size,  # n_test_instances
            True,
            normalize_sklearn=True,
            path_to_dataset=path
        )

        COLS = x_clean_train.shape[1]

        # 2. 训练VAE
        train_dataset = get_tf_database(x_clean_train, x_clean_train, self.batch_size)
        val_dataset = get_tf_database(x_clean, x_clean, self.batch_size)

        encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(
            train_dataset,
            val_dataset,
            COLS,
            self.latent_dim,
            self.K,
            1,
            T,
            epochs=self.epochs,
            model_name=dataset
        )

        # 3. 定义翻译函数
        @tf.function
        def _translate_all_columns_by_1(inputs):
            zs = inputs
            for column in range(zs.shape[0]):
                new_z = LOP.translate_operator(zs[column], 1)
                zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
            return zs

        def generate_cleaned_data(x_, Zs, Ks, decoder):
            xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis=1)), [1, 0, 2]))
            x_p = tf.where(Ks == 0, x_, xs)
            return x_p

        # 4. 加载dirty数据进行清洗
        headers, target_name, dirty_data, _, data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(
            dataset,
            -1,  # 使用全部dirty数据
            -1,
            MISSING_REPLACE,
            SCALER, CAT_ENCODER,
            True, MAX, MIN,
            normalize_sklearn=True,
            path_to_dataset=path
        )
        filtered_header = headers["filtered_header"]

        # 5. 执行清洗
        X_dirty = deepcopy(data_with_y[filtered_header]).to_numpy()
        Zs_csv, Ks_csv = predict_on_enhanced(X_dirty, LOP, encoder, decoder, _translate_all_columns_by_1)
        clean_csv = generate_cleaned_data(X_dirty, Zs_csv, Ks_csv, decoder)

        # 6. 转换结果
        if hasattr(clean_csv, 'numpy'):
            clean_csv_np = clean_csv.numpy()
        else:
            clean_csv_np = np.array(clean_csv)

        dirty_data[filtered_header] = clean_csv_np
        lop_data = reverse_to_input_domain(dataset, dirty_data, FULL_SCALER, CAT_ENCODER)
        lop_data = get_date_columns(dataset, lop_data, dirty_data)

        # 7. 后处理：将空值统一为 "empty"
        lop_data = lop_data.fillna('empty')
        lop_data = lop_data.replace('', 'empty')

        # 8. 保存结果
        default_output = os.path.join(path, dataset, 'lopster.csv')
        lop_data.to_csv(default_output, index=False)

        if output_path and output_path != default_output:
            lop_data.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'total_clean_rows': total_clean_rows,
            'clean_ratio': self.clean_ratio,
            'method': 'Lopster',
            'type': 'data-oriented',
            'auto_level': 2,  # Type 2: 需要训练数据(clean.csv)
            'latent_dim': self.latent_dim,
            'K': self.K,
            'epochs': self.epochs,
            'output_path': output_path or default_output,
            'note': f'Used {self.clean_ratio:.1%} of clean.csv for VAE training ({self.ground_truth_used} rows)'
        }

        return lop_data, info

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本"""
        return self.ground_truth_used


def lopster_clean(dataset: str,
                  path: str,
                  output_path: str = None,
                  clean_ratio: float = 1.0,
                  **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Lopster清洗的便捷函数

    Args:
        dataset: 数据集名称
        path: 数据集根目录
        output_path: 输出路径
        clean_ratio: clean数据使用比例（默认1.0=100%）
        **kwargs: 传递给LopsterWrapper的参数

    Returns:
        修复后的数据和清洗信息
    """
    wrapper = LopsterWrapper(clean_ratio=clean_ratio, **kwargs)
    return wrapper.clean(dataset, path, output_path)


def prepare_data_for_lopster(dirty_csv: str, clean_csv: str, dataset_name: str, output_dir: str):
    """
    准备Lopster所需的数据格式

    将标准的dirty.csv和clean.csv转换为Lopster要求的格式:
    - {output_dir}/{dataset_name}/clean.csv
    - {output_dir}/{dataset_name}/dirty01.csv
    """
    import shutil

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    shutil.copy(clean_csv, os.path.join(dataset_dir, 'clean.csv'))
    shutil.copy(dirty_csv, os.path.join(dataset_dir, 'dirty01.csv'))

    print(f"数据已准备到: {dataset_dir}/")
    return dataset_dir
