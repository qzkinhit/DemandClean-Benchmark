"""
Lopster Wrapper - 潜at空间marker学习datacleaning

LopsterisVLDB 2024提出基于潜at空间marker学习datacleaningmethod，
通过学习data潜atmarker来detectionandrepairError。

论文: Generalizable Data Cleaning of Tabular Data in Latent Space (VLDB 2024)
GitHub: https://github.com/DataManagementLab/data_cleaning_with_latent_operators

**ground truthusestats**:
- Lopsteruseclean.csvtrainingVAEmodel
- official implementationuse100%clean.csvtraining
- 本wrapperprovideclean_ratioParameter控制useratio
- classification: Type 2 - 需needtrainingdata

依赖: tensorflow, keras, scikit-learn, pandas, numpy, matplotlib
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

# 添加currentdirectorytopath
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


class LopsterWrapper:
    """
    Lopstercleaningmethod封装class

    直接Invokeofficialfunction，but非通过subprocess间接Invoke。

    核心Parameter:
    - latent_dim: 潜at空间维度 (default120)
    - K: translation actionParameter (default12)
    - epochs: trainingrounds (default100)
    - batch_size: batch size (default256)
    - clean_ratio: usecleandataratio (default1.0，i.e.100%)

    ground truthuse:
    - clean_ratio=1.0: useallcleandata（officialdefault）
    - clean_ratio=0.1: 仅use10%cleandatatraining
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
        initializeLopsterpackage装器

        Args:
            latent_dim: 潜at空间维度（officialdefault120）
            learning_rate: 学习率
            batch_size: batch size
            epochs: trainingrounds
            K: KParameter（translation actionParameter）
            clean_ratio: cleandatauseratio（0.0-1.0）
            verbose: whether打印详细信息
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.K = K
        self.clean_ratio = min(max(clean_ratio, 0.01), 1.0)  # 限制at0.01-1.0
        self.verbose = verbose
        self.ground_truth_used = 0

    def _check_requirements(self) -> bool:
        """checkTensorFlowwhether安装"""
        try:
            import tensorflow as tf
            if self.verbose:
                print(f"TensorFlow version: {tf.__version__}")
            return True
        except ImportError:
            raise ImportError(
                "Lopsterrequires TensorFlow。Please install: pip install tensorflow\n"
                "See Methods/Lopster/requirements.txt"
            )

    def _check_data_format(self, path: str, dataset: str) -> bool:
        """checkdataformatwhether符合officialneed求"""
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
                f"Lopsterofficial implementationrequires the following files:\n" +
                "\n".join(f"  - {m}" for m in missing)
            )
        return True

    def clean(self,
              dataset: str,
              path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执rowLopstercleaning流程 - 直接Invokeofficialfunction

        Args:
            dataset: Dataset名称 (corresponding {path}/{dataset}/ directory)
            path: Dataset根directory
            output_path: outputpath (optional)

        Returns:
            repairafterdataandcleaning信息
        """
        self._check_requirements()
        self._check_data_format(path, dataset)

        # 导入TensorFlowandofficialmodule
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from datasets import (
            get_tf_database, load_regression, load_features_and_data,
            reverse_to_input_domain, get_date_columns
        )
        from latent_operators import LatentOperator
        from utils import create_and_train_LOP
        from error_detection import predict_on_enhanced

        # loadcleandata
        clean_path = os.path.join(path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path)
        total_clean_rows = len(clean_df)

        # 计算实际usecleanrow count
        actual_clean_rows = int(total_clean_rows * self.clean_ratio)
        self.ground_truth_used = actual_clean_rows

        if self.verbose:
            print(f"Cleandata总量: {total_clean_rows} row")
            print(f"useratio: {self.clean_ratio:.1%}")
            print(f"实际use: {actual_clean_rows} row")

        # set采样size
        sample_size = actual_clean_rows if self.clean_ratio < 1.0 else -1

        # 1. loadcleandatatrainingVAE
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

        # 2. trainingVAE
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

        # 3. definition翻译function
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

        # 4. loaddirtydata进rowcleaning
        headers, target_name, dirty_data, _, data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(
            dataset,
            -1,  # usealldirtydata
            -1,
            MISSING_REPLACE,
            SCALER, CAT_ENCODER,
            True, MAX, MIN,
            normalize_sklearn=True,
            path_to_dataset=path
        )
        filtered_header = headers["filtered_header"]

        # 5. Run cleaning
        X_dirty = deepcopy(data_with_y[filtered_header]).to_numpy()
        Zs_csv, Ks_csv = predict_on_enhanced(X_dirty, LOP, encoder, decoder, _translate_all_columns_by_1)
        clean_csv = generate_cleaned_data(X_dirty, Zs_csv, Ks_csv, decoder)

        # 6. Convertresult
        if hasattr(clean_csv, 'numpy'):
            clean_csv_np = clean_csv.numpy()
        else:
            clean_csv_np = np.array(clean_csv)

        dirty_data[filtered_header] = clean_csv_np
        lop_data = reverse_to_input_domain(dataset, dirty_data, FULL_SCALER, CAT_ENCODER)
        lop_data = get_date_columns(dataset, lop_data, dirty_data)

        # 7. after处理：将empty valuenormalize to "empty"
        lop_data = lop_data.fillna('empty')
        lop_data = lop_data.replace('', 'empty')

        # 8. Save results
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
            'auto_level': 2,  # Type 2: 需needtrainingdata(clean.csv)
            'latent_dim': self.latent_dim,
            'K': self.K,
            'epochs': self.epochs,
            'output_path': output_path or default_output,
            'note': f'Used {self.clean_ratio:.1%} of clean.csv for VAE training ({self.ground_truth_used} rows)'
        }

        return lop_data, info

    def get_ground_truth_cost(self) -> int:
        """getground truthuse成本"""
        return self.ground_truth_used


def lopster_clean(dataset: str,
                  path: str,
                  output_path: str = None,
                  clean_ratio: float = 1.0,
                  **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Lopstercleaning便捷function

    Args:
        dataset: Dataset名称
        path: Dataset根directory
        output_path: outputpath
        clean_ratio: cleandatauseratio（default1.0=100%）
        **kwargs: 传递toLopsterWrapperParameter

    Returns:
        repairafterdataandcleaning信息
    """
    wrapper = LopsterWrapper(clean_ratio=clean_ratio, **kwargs)
    return wrapper.clean(dataset, path, output_path)


def prepare_data_for_lopster(dirty_csv: str, clean_csv: str, dataset_name: str, output_dir: str):
    """
    准备Lopster所需dataformat

    将标准dirty.csvandclean.csvConverttoLopsterneed求format:
    - {output_dir}/{dataset_name}/clean.csv
    - {output_dir}/{dataset_name}/dirty01.csv
    """
    import shutil

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    shutil.copy(clean_csv, os.path.join(dataset_dir, 'clean.csv'))
    shutil.copy(dirty_csv, os.path.join(dataset_dir, 'dirty01.csv'))

    print(f"data准备to: {dataset_dir}/")
    return dataset_dir
