"""
CtxPipe Adapter for Clean4MLBaseline Project
将官方 ctxpipe 的 API 适配到项目的统一接口

重要说明：
- Methods/ctxpipe 目录保持与官方仓库完全一致
- 所有兼容性处理（设备检测、路径处理）都在此适配器中完成
- 使用预训练模型进行推理，无需训练
"""
import os
import sys
import warnings
import pandas as pd
from typing import Tuple, Optional

# ============================================================================
# 第一步：在导入 ctxpipe 之前设置兼容性补丁
# ============================================================================

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 添加 ctxpipe 目录到路径（官方结构：根目录有 env.py, config.py 等）
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
CTXPIPE_ROOT = os.path.join(PROJECT_ROOT, 'Methods', 'ctxpipe')
sys.path.insert(0, CTXPIPE_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# 在导入 torch 和 ctxpipe 之前设置设备兼容性
import torch

# 自动检测设备
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[CtxPipe Adapter] Using device: CUDA")
else:
    DEVICE = torch.device("cpu")
    print(f"[CtxPipe Adapter] Using device: CPU (CUDA not available)")


# ============================================================================
# 第二步：导入 ctxpipe 模块并应用运行时补丁
# ============================================================================

# 导入 ctxpipe 的 env 模块并修补设备设置
try:
    import env as ctxpipe_env
    # 修补设备设置（官方代码硬编码 cuda）
    ctxpipe_env.DEVICE = DEVICE
except ImportError as e:
    print(f"[CtxPipe Adapter] Error importing env module: {e}")
    raise

# 导入配置模块并修补
try:
    import config as ctxpipe_config
    # 修补 GlobalConfig 的设备设置
    ctxpipe_config.GlobalConfig.device = DEVICE
except ImportError as e:
    print(f"[CtxPipe Adapter] Error importing config module: {e}")
    raise

# 导入其他必要模块
try:
    from ctxpipe.pipegen import PipelineGenerator
    from ctxpipe.dataset import Dataset
    from ctxpipe.info import Info
    from ctxpipe.stats import init_stats_db
    import util as ctxpipe_util
except ImportError as e:
    print(f"[CtxPipe Adapter] Error importing ctxpipe modules: {e}")
    print(f"[CtxPipe Adapter] CTXPIPE_ROOT: {CTXPIPE_ROOT}")
    raise


# ============================================================================
# 第三步：适配器类
# ============================================================================

class CtxPipeAdapter:
    """
    适配器类，封装官方 ctxpipe 的调用逻辑
    """

    def __init__(self, model_tag: str = "ctx_50000"):
        """
        初始化适配器

        Args:
            model_tag: 预训练模型标签，默认为 "ctx_50000"
        """
        self.model_tag = model_tag
        self._initialized = False

    def _init_ctxpipe(self):
        """初始化 ctxpipe 环境"""
        if self._initialized:
            return

        # 调用官方的 init 函数（已修补设备）
        import numpy as np
        import deterministic

        deterministic.seed_everything()
        np.set_printoptions(suppress=True)
        torch.set_printoptions(sci_mode=False)
        warnings.filterwarnings("ignore")

        # 创建必要目录
        ctxpipe_config.default_config.makedirs()

        self._initialized = True

    def _normalize_path(self, path: str) -> str:
        """
        规范化路径，处理 Windows/Linux 兼容性

        Args:
            path: 原始路径

        Returns:
            规范化后的路径（使用正斜杠）
        """
        return path.replace("\\", "/")

    def has_index_column(self, df: pd.DataFrame) -> bool:
        """检测数据框是否有 index 列作为第一列"""
        first_col = df.columns[0]
        return first_col.strip().lower().replace('\ufeff', '') == 'index'

    def detect_label_column(self, df: pd.DataFrame) -> int:
        """自动检测标签列索引"""
        label_candidates = ['label', 'target', 'class', 'y', 'output', 'income',
                          'style', 'cnt', 'gt', 'labels', 'sound_pressure_level',
                          'soil_moisture']
        columns_lower = [col.lower() for col in df.columns]

        for candidate in label_candidates:
            if candidate in columns_lower:
                return columns_lower.index(candidate)

        # 默认使用最后一列
        print(f"[CtxPipe Adapter] Warning: No label column found. Using last column.")
        return len(df.columns) - 1

    def run_ctxpipe(
        self,
        dirty_path: str,
        task_name: str,
        label_index: Optional[int] = None,
        output_path: Optional[str] = None,
        schema_preserving: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        运行 ctxpipe 进行数据准备管道生成

        Args:
            dirty_path: 脏数据 CSV 文件路径
            task_name: 任务名称
            label_index: 标签列索引（如果为 None，则自动检测）
            output_path: 输出路径
            schema_preserving: 是否保留原始 schema（仅做缺失值填充）

        Returns:
            (处理后的数据 DataFrame, 结果字典)
        """
        # 初始化 ctxpipe
        self._init_ctxpipe()

        # 规范化路径
        dirty_path = self._normalize_path(dirty_path)

        # 读取数据
        df_original = pd.read_csv(dirty_path)
        original_index_series = None

        # 检测是否有 index 列
        has_index_col = self.has_index_column(df_original)
        if has_index_col:
            print(f"[CtxPipe Adapter] Detected index column. Removing for processing.")
            index_col_name = df_original.columns[0]
            original_index_series = df_original[index_col_name].copy()
            df = df_original.set_index(index_col_name)
            df.index.name = None
        else:
            df = df_original.copy()

        # 检测或使用指定的标签列
        if label_index is None:
            label_index = self.detect_label_column(df)
            print(f"[CtxPipe Adapter] Auto-detected label column: {label_index} ({df.columns[label_index]})")
        else:
            if has_index_col and label_index > 0:
                label_index = label_index - 1
            print(f"[CtxPipe Adapter] Using label column: {label_index} ({df.columns[label_index]})")

        # 验证标签列索引
        if label_index < 0 or label_index >= len(df.columns):
            raise ValueError(f"Invalid label_index {label_index}. Must be in [0, {len(df.columns)-1}]")

        # schema_preserving 模式：仅做简单填充
        if schema_preserving:
            print("[CtxPipe Adapter] Schema-preserving mode: simple imputation only.")
            df_sp = df.copy()
            df_sp = df_sp.replace({"empty": pd.NA, "": pd.NA})

            for col in df_sp.columns:
                if pd.api.types.is_numeric_dtype(df_sp[col]):
                    if df_sp[col].isna().any():
                        df_sp[col] = df_sp[col].fillna(df_sp[col].median())
                else:
                    if df_sp[col].isna().any():
                        mode = df_sp[col].mode()
                        df_sp[col] = df_sp[col].fillna(mode.iloc[0] if not mode.empty else "")

            results = {
                'ai_sequence': ['ImputerNum', 'ImputerCat'],
                'ml_score': None,
                'logical_pipeline': None,
            }

            if has_index_col and original_index_series is not None:
                df_sp.insert(0, 'index', original_index_series.values)

            return df_sp, results

        # 创建临时 CSV（移除 index 列，并将 empty 转换为真正的空值）
        temp_csv_path = dirty_path

        # 将 "empty" 字符串转换为真正的空值（让ctxpipe能正确检测缺失值）
        df_for_ctxpipe = df.copy()
        df_for_ctxpipe = df_for_ctxpipe.replace({"empty": "", "Empty": "", "EMPTY": ""})
        print(f"[CtxPipe Adapter] Converted 'empty' strings to actual empty values for NaN detection.")

        if has_index_col or True:  # 总是创建临时文件以确保empty被正确处理
            import tempfile
            dataset_dir = os.path.dirname(dirty_path)
            temp_fd, temp_csv_path = tempfile.mkstemp(suffix='.csv', prefix='ctxpipe_', dir=dataset_dir)
            os.close(temp_fd)
            df_for_ctxpipe.to_csv(temp_csv_path, index=False)
            temp_csv_path = self._normalize_path(temp_csv_path)

        # 设置 Info 和配置
        if output_path:
            output_path = self._normalize_path(output_path)
            dataset_dir = os.path.dirname(temp_csv_path)
            dataset_root = os.path.dirname(dataset_dir)

            info = Info(
                aipipe_core_prefix=os.path.join(output_path, "aipipe"),
                result_prefix=os.path.join(output_path, "result"),
                dataset_prefix=dataset_root,
            )

            # 构造任务信息
            task_entry = {
                "dataset": os.path.basename(dataset_dir),
                "csv_file": os.path.basename(temp_csv_path),
                "label": str(label_index),
                "model": "LogisticRegression",
                "task_name": task_name,
            }

            import json
            os.makedirs(os.path.dirname(info.task_info_path), exist_ok=True)
            with open(info.task_info_path, 'w', encoding='utf-8') as f:
                json.dump({"0": task_entry}, f)

            with open(info.task_index_path, 'w', encoding='utf-8') as f:
                json.dump(["0"], f)

            # 注入配置
            ctxpipe_config.set_info(info)
            ctxpipe_config.init()
            init_stats_db(info.stats_db_file_path)

        # 创建数据集对象（官方 Dataset 类）
        dataset = Dataset(
            name=task_name,
            path=temp_csv_path,
            label_column_id=label_index
        )

        # 运行管道生成
        print(f"[CtxPipe Adapter] Generating pipeline with model tag: {self.model_tag}")
        pg = PipelineGenerator(dataset, model_tag=self.model_tag)
        pg.generate()

        # 获取输出
        try:
            pg_stats = pg.output()
        except Exception as e:
            print(f"[CtxPipe Adapter] Warning: pg.output() failed: {e}")
            pg_stats = None

        results = {
            'ai_sequence': pg.ai_sequence,
            'ml_score': pg.ml_score,
            'logical_pipeline': getattr(pg, 'logical_pipeline', None),
        }

        # 应用ctxpipe生成的管道到数据上
        print(f"[CtxPipe Adapter] Applying generated pipeline to data...")
        cleaned_df = self._apply_pipeline(df_for_ctxpipe, pg.ai_sequence, label_index)
        print(f"[CtxPipe Adapter] Pipeline applied successfully.")

        # 清理临时文件
        if temp_csv_path != dirty_path:
            try:
                os.remove(temp_csv_path)
            except:
                pass

        # 恢复 index 列
        if has_index_col and original_index_series is not None:
            cleaned_df.insert(0, 'index', original_index_series.values)

        return cleaned_df, results

    def _apply_pipeline(self, df: pd.DataFrame, ai_sequence: list, label_index: int) -> pd.DataFrame:
        """
        手动应用ctxpipe生成的管道到数据

        Args:
            df: 原始数据DataFrame
            ai_sequence: ctxpipe生成的AI序列
            label_index: 标签列索引

        Returns:
            处理后的DataFrame
        """
        from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer
        from sklearn.impute import SimpleImputer
        import numpy as np

        # 分离特征和标签
        label_col = df.columns[label_index]
        X = df.drop(columns=[label_col])
        y = df[label_col].copy()

        # 关键步骤：先把 "empty" 转换为 NaN，然后尝试转换数据类型
        # 这样 ibu 这样的列才能被正确识别为数值列
        X = X.replace({"empty": np.nan, "Empty": np.nan, "EMPTY": np.nan, "": np.nan})

        # 尝试将可以转换为数值的列转换
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='ignore')
            except:
                pass

        # 识别数值列和类别列
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

        print(f"[CtxPipe Adapter] Data shape: {X.shape}, Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}")
        print(f"[CtxPipe Adapter] Num columns: {num_cols}")
        print(f"[CtxPipe Adapter] AI Sequence: {ai_sequence}")

        # 应用管道中的每个步骤
        # 注意：为了保持与原始数据相同的值和结构（便于传统清洗评估），
        # 我们只应用缺失值填充操作，跳过会改变数据值或结构的操作
        #
        # 只执行的操作（数据清洗）：
        #   - ImputerNum, ImputerCat（缺失值填充）
        #
        # 跳过的操作（ML数据准备，会改变数据值/结构）：
        #   - LabelEncoder, OneHotEncoder（编码）
        #   - StandardScaler, MinMaxScaler, PowerTransformer 等（缩放/变换）
        #   - PCA, RandomTreesEmbedding 等（特征工程）

        IMPUTATION_OPS = ['imputernum', 'imputercat', 'simpleimputer']

        for step_name in ai_sequence:
            step_name_lower = step_name.lower() if step_name else ""

            # 跳过blank操作
            if step_name == "blank" or not step_name:
                continue

            # 只执行缺失值填充操作
            is_imputation = any(op in step_name_lower for op in IMPUTATION_OPS)

            if not is_imputation:
                print(f"[CtxPipe Adapter] Skipped {step_name} (not imputation, would change data values)")
                continue

            try:
                # 缺失值填充 - 数值
                if "imputernum" in step_name_lower or "simpleimputer" in step_name_lower:
                    if len(num_cols) > 0:
                        imputer = SimpleImputer(strategy='median')
                        X[num_cols] = imputer.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied ImputerNum (median)")

                # 缺失值填充 - 类别
                elif "imputercat" in step_name_lower:
                    if len(cat_cols) > 0:
                        imputer = SimpleImputer(strategy='most_frequent')
                        X[cat_cols] = imputer.fit_transform(X[cat_cols])
                        print(f"[CtxPipe Adapter] Applied ImputerCat (most_frequent)")

                elif "minmaxscaler" in step_name_lower:
                    if len(num_cols) > 0:
                        scaler = MinMaxScaler()
                        X[num_cols] = scaler.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied MinMaxScaler")

                elif "maxabsscaler" in step_name_lower:
                    if len(num_cols) > 0:
                        scaler = MaxAbsScaler()
                        X[num_cols] = scaler.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied MaxAbsScaler")

                elif "powertransformer" in step_name_lower:
                    if len(num_cols) > 0:
                        # PowerTransformer需要正值，先处理
                        for col in num_cols:
                            if (X[col] <= 0).any():
                                X[col] = X[col] - X[col].min() + 1
                        transformer = PowerTransformer(method='yeo-johnson')
                        X[num_cols] = transformer.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied PowerTransformer")

                elif "quantiletransformer" in step_name_lower:
                    if len(num_cols) > 0:
                        transformer = QuantileTransformer(output_distribution='normal')
                        X[num_cols] = transformer.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied QuantileTransformer")

                # 特征工程
                elif "numericdata" in step_name_lower:
                    # 只保留数值列
                    if len(cat_cols) > 0:
                        X = X.drop(columns=cat_cols)
                        cat_cols = []
                    print(f"[CtxPipe Adapter] Applied NumericData (dropped cat cols)")

                elif "interactionfeatures" in step_name_lower:
                    # 添加交互特征（简化版，只对数值列）
                    if len(num_cols) >= 2:
                        from itertools import combinations
                        for c1, c2 in list(combinations(num_cols, 2))[:5]:  # 限制数量
                            X[f"{c1}_{c2}_mult"] = X[c1] * X[c2]
                        print(f"[CtxPipe Adapter] Applied InteractionFeatures")

                elif "pca" in step_name_lower:
                    from sklearn.decomposition import PCA
                    if len(num_cols) > 0:
                        n_components = min(len(num_cols), X.shape[0] - 1, 10)
                        if n_components > 0:
                            pca = PCA(n_components=n_components)
                            pca_result = pca.fit_transform(X[num_cols])
                            pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index)
                            X = X.drop(columns=num_cols)
                            X = pd.concat([X, pca_df], axis=1)
                            num_cols = list(pca_df.columns)
                        print(f"[CtxPipe Adapter] Applied PCA")

                elif "randomtreesembedding" in step_name_lower:
                    from sklearn.ensemble import RandomTreesEmbedding
                    if len(num_cols) > 0:
                        rte = RandomTreesEmbedding(n_estimators=10, max_depth=3, random_state=42, sparse_output=False)
                        rte_result = rte.fit_transform(X[num_cols].fillna(0))
                        # 限制特征数量
                        n_features = min(rte_result.shape[1], 20)
                        rte_df = pd.DataFrame(rte_result[:, :n_features], columns=[f"RTE{i+1}" for i in range(n_features)], index=X.index)
                        X = X.drop(columns=num_cols)
                        X = pd.concat([X, rte_df], axis=1)
                        num_cols = list(rte_df.columns)
                        print(f"[CtxPipe Adapter] Applied RandomTreesEmbedding")

                elif "truncatedsvd" in step_name_lower:
                    from sklearn.decomposition import TruncatedSVD
                    if len(num_cols) > 0:
                        n_components = min(len(num_cols) - 1, X.shape[0] - 1, 10)
                        if n_components > 0:
                            svd = TruncatedSVD(n_components=n_components)
                            svd_result = svd.fit_transform(X[num_cols].fillna(0))
                            svd_df = pd.DataFrame(svd_result, columns=[f"SVD{i+1}" for i in range(n_components)], index=X.index)
                            X = X.drop(columns=num_cols)
                            X = pd.concat([X, svd_df], axis=1)
                            num_cols = list(svd_df.columns)
                        print(f"[CtxPipe Adapter] Applied TruncatedSVD")

                elif "polynomialfeatures" in step_name_lower:
                    from sklearn.preprocessing import PolynomialFeatures
                    if len(num_cols) > 0 and len(num_cols) <= 5:  # 限制以避免特征爆炸
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        poly_result = poly.fit_transform(X[num_cols].fillna(0))
                        poly_names = [f"poly_{i}" for i in range(poly_result.shape[1])]
                        poly_df = pd.DataFrame(poly_result, columns=poly_names, index=X.index)
                        X = X.drop(columns=num_cols)
                        X = pd.concat([X, poly_df], axis=1)
                        num_cols = poly_names
                        print(f"[CtxPipe Adapter] Applied PolynomialFeatures")

                elif "variancethreshold" in step_name_lower:
                    from sklearn.feature_selection import VarianceThreshold
                    if len(num_cols) > 0:
                        selector = VarianceThreshold(threshold=0.01)
                        try:
                            X_selected = selector.fit_transform(X[num_cols].fillna(0))
                            selected_cols = [num_cols[i] for i in selector.get_support(indices=True)]
                            X = X.drop(columns=num_cols)
                            selected_df = pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
                            X = pd.concat([X, selected_df], axis=1)
                            num_cols = selected_cols
                            print(f"[CtxPipe Adapter] Applied VarianceThreshold")
                        except:
                            print(f"[CtxPipe Adapter] VarianceThreshold failed, skipping...")

                else:
                    print(f"[CtxPipe Adapter] Unknown step: {step_name}, skipping...")

            except Exception as e:
                print(f"[CtxPipe Adapter] Error applying {step_name}: {e}, skipping...")

        # 合并特征和标签
        result_df = X.copy()
        result_df[label_col] = y.values

        # 重新排列列，把标签放回原位置
        cols = list(result_df.columns)
        cols.remove(label_col)
        cols.insert(label_index, label_col)
        result_df = result_df[cols]

        return result_df


if __name__ == "__main__":
    # 测试
    adapter = CtxPipeAdapter(model_tag="ctx_50000")
    print("[Test] CtxPipeAdapter initialized successfully.")
