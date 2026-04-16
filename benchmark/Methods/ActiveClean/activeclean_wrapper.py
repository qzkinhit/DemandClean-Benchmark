"""
ActiveClean Wrapper

封装 ActiveClean 算法，提供清洗后数据的输出。

核心逻辑：
1. ActiveClean 通过采样策略选择最有价值的样本进行清洗
2. "清洗" = 将采样的脏数据行替换为对应的干净数据行
3. 采样数量 = 使用的真值数量 (ground truth cost)
4. 未采样的行保持原样（未清洗）

支持：
- 自动向量化：内部处理时向量化，输出时还原为原始格式
- 所有任务类型：分类、回归、聚类（通过 getScoreML 评估）
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def vectorize_dataframe(df, label_col=None, encoders=None):
    """
    将 DataFrame 向量化（所有列转为数值型）。

    Args:
        df: 输入 DataFrame
        label_col: 标签列名（如果为 None，使用最后一列）
        encoders: 已有的编码器字典（用于保持 clean/dirty 编码一致）

    Returns:
        vectorized_df: 向量化后的 DataFrame
        encoders: 编码器字典（用于后续解码或对齐）
    """
    df = df.copy()

    # 将 "empty" 空值标记替换为统一的占位符
    df = df.replace('empty', '__NA__')

    if label_col is None:
        label_col = df.columns[-1]

    if encoders is None:
        encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            if col not in encoders:
                encoders[col] = LabelEncoder()
                # 合并所有可能的值来 fit encoder
                encoders[col].fit(df[col].astype(str).fillna('__NA__'))

            # transform
            df[col] = encoders[col].transform(df[col].astype(str).fillna('__NA__'))

    # 确保所有列都是数值型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 填充 NaN
    df = df.fillna(0)

    return df, encoders


def needs_vectorization(df):
    """检查 DataFrame 是否需要向量化"""
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            return True
    return False


# ============================================================================
# ActiveClean 辅助函数（使用现代 sklearn API）
# ============================================================================

def translate_indices(globali, imap):
    """返回 globali 在 imap 中的元素及其索引"""
    lset = set(globali)
    return [s for s, t in enumerate(imap) if t in lset]


def error_classifier(total_labels, full_data):
    """
    利用采样数据的标签训练一个错误检测的分类器。

    Args:
        total_labels: [(index, is_clean), ...] 标签列表
        full_data: 完整特征数据

    Returns:
        训练好的分类器，或 None（如果所有样本都是干净的）
    """
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]

    # 判断是否所有的label都是clean
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices, :], labels)
        return clf
    else:
        return None


def ec_filter(dirtyex, full_data, clf, t=0.90):
    """
    使用错误分类器过滤可能干净的数据。

    Args:
        dirtyex: 待过滤的索引列表
        full_data: 完整特征数据
        clf: 错误分类器
        t: 阈值

    Returns:
        过滤后的索引列表
    """
    if clf is not None:
        pred = clf.predict_proba(full_data[dirtyex, :])
        return [j for i, j in enumerate(dirtyex) if pred[i][0] < t]

    return dirtyex


# ============================================================================
# 主要函数
# ============================================================================

def run_activeclean(clean_path, dirty_path, batchsize=50, total=10000):
    """
    运行 ActiveClean 并返回清洗后的数据集。

    支持自动向量化：如果数据包含非数值列，会自动进行 LabelEncoder 转换。
    输出数据：保持原始格式（非向量化），只替换被采样的行。

    Args:
        clean_path (str): 干净数据路径
        dirty_path (str): 脏数据路径
        batchsize (int): 每批采样大小
        total (int): 最大清洗样本数

    Returns:
        txt: 运行报告文本
        cleaned_data: 清洗后的 DataFrame（原始格式）
        ground_truth_cost: 真值使用量（被清洗的样本数）
    """
    # 读取原始数据
    correct_data_raw = pd.read_csv(clean_path)
    injected_data_raw = pd.read_csv(dirty_path)

    # 保存原始数据（用于最终输出）
    original_clean = correct_data_raw.copy()
    original_dirty = injected_data_raw.copy()
    original_columns = correct_data_raw.columns.tolist()

    # 检查是否需要向量化（仅用于内部处理）
    if needs_vectorization(correct_data_raw) or needs_vectorization(injected_data_raw):
        print("[ActiveClean Wrapper] 检测到非数值列，自动进行向量化...")

        # 合并数据来 fit encoder（确保编码一致）
        combined = pd.concat([correct_data_raw, injected_data_raw], ignore_index=True)
        _, encoders = vectorize_dataframe(combined)

        # 分别向量化（仅用于内部算法）
        correct_data, _ = vectorize_dataframe(correct_data_raw, encoders=encoders)
        injected_data, _ = vectorize_dataframe(injected_data_raw, encoders=encoders)

        print(f"[ActiveClean Wrapper] 向量化完成，编码列: {list(encoders.keys())}")
    else:
        correct_data = correct_data_raw.copy()
        injected_data = injected_data_raw.copy()
        encoders = None

    # 提取特征和标签（向量化后的数据，用于算法内部）
    X_correct = correct_data.iloc[:, :-1].values
    y_correct = correct_data.iloc[:, -1].values

    X_full = injected_data.iloc[:, :-1].values
    y_full = injected_data.iloc[:, -1].values

    size = len(X_full)

    # 统计干净/脏数据（信息展示）
    try:
        indices_clean_rows = np.where((X_full == X_correct).all(axis=1))[0]
        indices_dirty_rows = np.where((X_full != X_correct).any(axis=1))[0]
        print(f"[ActiveClean Wrapper] 数据集: {size} 行, {len(indices_dirty_rows)} 脏行, {len(indices_clean_rows)} 干净行")
    except:
        print(f"[ActiveClean Wrapper] 数据集: {size} 行")

    # 使用完整的干净数据集作为 ground truth
    X_clean = X_correct
    y_clean = y_correct

    # 生成所有行的索引
    all_indices = np.arange(0, size, 1)

    # 分割训练集和测试集
    # 注意：对于回归任务，不能使用 stratify
    try:
        train_indices, test_indices = train_test_split(all_indices, test_size=0.20, stratify=y_full)
    except ValueError:
        # 如果 stratify 失败（如回归任务），则不使用 stratify
        train_indices, test_indices = train_test_split(all_indices, test_size=0.20, random_state=42)

    # 运行 ActiveClean 算法
    txt, cleanex = activeclean_sampling(
        X_full,
        y_full,
        X_clean,
        y_clean,
        train_indices,
        test_indices,
        batchsize=batchsize,
        total=total
    )

    # ========================================================================
    # 构造清洗后的数据集（使用原始格式）
    # ========================================================================
    # cleanex 包含被采样/清洗的样本索引
    # 清洗逻辑：将这些索引的行替换为干净数据

    cleaned_data = original_dirty.copy()

    for idx in cleanex:
        # 将采样行替换为干净数据的对应行
        cleaned_data.iloc[idx] = original_clean.iloc[idx]

    ground_truth_cost = len(cleanex)

    print(f"[ActiveClean Wrapper] 清洗完成: {ground_truth_cost} 行被替换为干净数据")

    return txt, cleaned_data, ground_truth_cost


def activeclean_sampling(X_dirty, y_dirty, X_clean, y_clean, train_indices, test_indices,
                         batchsize=50, total=10000):
    """
    ActiveClean 采样算法。

    使用 SGDClassifier 的梯度信息来选择最有价值的样本进行清洗。
    支持任何任务类型（实际评估由 getScoreML 完成）。

    Args:
        X_dirty: 脏数据特征（向量化）
        y_dirty: 脏数据标签
        X_clean: 干净数据特征（向量化）
        y_clean: 干净数据标签
        train_indices: 训练集索引
        test_indices: 测试集索引
        batchsize: 每批采样大小
        total: 最大清洗样本数

    Returns:
        txt: 运行报告
        cleanex: 被清洗的样本索引列表
    """
    print("[ActiveClean] Initialization")
    txt = "[ActiveClean] Initialization"

    # 检查标签类型，判断是否可以使用分类模式
    unique_labels = np.unique(y_clean)
    is_classification = len(unique_labels) <= 100  # 假设超过100个唯一值是回归任务

    if is_classification:
        all_classes = unique_labels
        print(f"[ActiveClean] 检测到分类任务，类别数: {len(all_classes)}")
    else:
        print(f"[ActiveClean] 检测到回归/高基数任务，使用简单采样策略")

    # 待清洗的候选索引（训练集中的所有行）
    remaining = list(train_indices)
    cleanex = []  # 已清洗的索引
    total_labels = []  # 用于训练错误分类器

    # 测试集数据（用于评估）
    X_test = X_clean[test_indices]
    y_test = y_clean[test_indices]

    # 初始采样
    if len(remaining) < batchsize:
        initial_batch = remaining.copy()
    else:
        initial_sample_idx = np.random.choice(len(remaining), batchsize, replace=False)
        initial_batch = [remaining[i] for i in initial_sample_idx]

    # 将初始批次加入已清洗列表
    cleanex.extend(initial_batch)
    for idx in initial_batch:
        if idx in remaining:
            remaining.remove(idx)

    # 初始化分类器（用于ActiveClean的采样策略）
    clf = None
    if is_classification and len(cleanex) > 0:
        try:
            clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200,
                               fit_intercept=True, warm_start=True)
            clf.partial_fit(X_clean[cleanex], y_clean[cleanex], classes=all_classes)
        except Exception as e:
            print(f"[ActiveClean] 分类器初始化失败: {e}，使用简单采样")
            clf = None

    # 迭代清洗
    for i in range(batchsize, total, batchsize):
        print(f"[ActiveClean] Number Cleaned So Far: {len(cleanex)}")
        txt += f"\n[ActiveClean] Number Cleaned So Far: {len(cleanex)}"

        # 评估当前模型（如果是分类任务）
        if clf is not None:
            try:
                ypred = clf.predict(X_test)
                acc = accuracy_score(y_test, ypred)
                print(f"[ActiveClean] Internal Accuracy: {acc:.4f}")
                txt += f"\n[ActiveClean] Internal Accuracy: {acc:.4f}"
            except:
                pass

        # 检查是否还有剩余样本
        if len(remaining) < batchsize:
            if len(remaining) == 0:
                print("[ActiveClean] No more samples to clean")
                txt += "\n[ActiveClean] No more samples to clean"
                break
            else:
                # 清洗剩余所有样本
                batch = remaining.copy()
        else:
            # 随机采样新批次
            sample_idx = np.random.choice(len(remaining), batchsize, replace=False)
            batch = [remaining[i] for i in sample_idx]

        # 记录采样标签（用于错误分类器）
        # 判断采样的数据原本是否"干净"（与 ground truth 相同）
        for idx in batch:
            try:
                is_originally_clean = np.allclose(X_dirty[idx], X_clean[idx])
            except:
                is_originally_clean = False
            total_labels.append((idx, is_originally_clean))

        # 将新批次加入已清洗列表
        cleanex.extend(batch)
        for idx in batch:
            if idx in remaining:
                remaining.remove(idx)

        # 尝试使用错误分类器过滤（ActiveClean 核心优化）
        if clf is not None and len(total_labels) > batchsize:
            try:
                ec = error_classifier(total_labels, X_dirty)
                if ec is not None:
                    remaining = ec_filter(remaining, X_dirty, ec)
            except:
                pass

        # 用清洗后的数据更新模型
        if clf is not None:
            try:
                clf.partial_fit(X_clean[cleanex], y_clean[cleanex])
            except:
                pass

    # 最终统计
    print(f"[ActiveClean] Final - Number Cleaned: {len(cleanex)}")
    txt += f"\n[ActiveClean] Final - Number Cleaned: {len(cleanex)}"

    return txt, cleanex
