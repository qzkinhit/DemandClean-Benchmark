import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, jaccard_score
import numpy as np


def calculate_all_metrics(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', calculate_precision_recall=True,
                          calculate_edr=True, calculate_hybrid=True, calculate_r_edr=True, mse_attributes=[], relax=True,
                          save_debug_files=True):
    """
    计算多个指标的统一函数，包括修复准确率和召回率、EDR、混合距离以及基于条目的 R-EDR。

    :param clean: 干净数据 DataFrame
    :param dirty: 脏数据 DataFrame
    :param cleaned: 清洗后数据 DataFrame
    :param attributes: 指定的属性集合
    :param output_path: 保存结果的目录路径
    :param task_name: 任务名称
    :param calculate_precision_recall: 是否计算修复的准确率和召回率
    :param calculate_edr: 是否计算错误减少率（EDR）
    :param calculate_hybrid: 是否计算混合距离指标
    :param calculate_r_edr: 是否计算基于条目的错误减少率（R-EDR）
    :param relax: 比对时是否忽略大小写(有些baseline系统（例如holoclean），强制清洗后的数据统一变成小写字母)
    :param save_debug_files: 是否将差异CSV保存到debug子目录（默认True）
    :return: 所有计算的指标值
    """
    results = {}

    # 计算准确率和召回率
    if calculate_precision_recall:
        try:
            accuracy, recall = calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name,
                                                             index_attribute=index_attribute, relax=relax,
                                                             save_debug_files=save_debug_files)
            results['accuracy'] = accuracy       # 历史兼容：实际是 precision = TP/(TP+FP)
            results['precision'] = accuracy      # 显式 precision 字段
            results['recall'] = recall
            f1_score = calF1(accuracy, recall)
            results['f1_score'] = f1_score
            print(f"修复准确率: {accuracy}, 修复召回率: {recall}, F1值: {f1_score}")
            print("=" * 40)
        except Exception as e:
            print(f"准确率/召回率计算出错: {e}")

    # 计算EDR
    if calculate_edr:
        try:
            edr = get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute=index_attribute, relax=relax)
            results['edr'] = edr
            print(f"错误减少率 (EDR): {edr}")
            print("=" * 40)
        except Exception as e:
            print(f"EDR计算出错: {e}")

    # 计算混合距离
    if calculate_hybrid:
        try:
            hybrid_distance = get_hybrid_distance(clean, cleaned, attributes, output_path, task_name,
                                                  index_attribute=index_attribute, mse_attributes=mse_attributes, relax=relax)
            results['hybrid_distance'] = hybrid_distance
            print(f"混合距离 (Hybrid Distance): {hybrid_distance}")
            print("=" * 40)
        except Exception as e:
            print(f"混合距离计算出错: {e}")

    # 计算基于条目的 R-EDR
    if calculate_r_edr:
        try:
            r_edr = get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute=index_attribute, relax=relax)
            results['r_edr'] = r_edr
            print(f"基于条目的错误减少率 (R-EDR): {r_edr}")
            print("=" * 40)
        except Exception as e:
            print(f"R-EDR计算出错: {e}")

    return results

def normalize_value(value):
    """
    将数值规范化为字符串格式，去掉小数点及其后的零
    :param value: 要规范化的值
    :return: 规范化后的字符串
    """
    try:
        # 尝试将值转换为浮点数，再转换为整数，然后转换为字符串
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))  # 去掉小数点及其后的零
        else:
            return str(float_value)
    except ValueError:
        # 如果值无法转换为浮点数，则返回原始值的字符串形式
        return str(value)


def default_distance_func(value1, value2):
    """
    默认的距离计算函数：
    如果两个值不同，则距离为1；
    如果两个值相同，则距离为0。
    """
    return (value1 != value2).sum()

def record_based_distance_func(row1, row2):
    """
    基于条目的距离计算函数：
    遍历每一行中的每一个值，如果任意一个值不相同，则返回1；
    如果所有值都相同，则返回0。
    """
    for val1, val2 in zip(row1, row2):
        if val1 != val2:
            return 1  # 只要有一个值不相同，立即返回1
    return 0  # 如果所有值都相同，返回0
def calF1(precision, recall):
    """
    计算F1值

    :param precision: 精度
    :param recall: 召回率
    :return: F1值
    """
    return 2 * precision * recall / (precision + recall + 1e-10)


def calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', relax=False,
                                  save_debug_files=True):
    """
    计算指定属性集合下的修复准确率和召回率，并将结果输出到文件中，同时生成差异 CSV 文件。

    :param save_debug_files: 是否将差异CSV保存到debug子目录（默认True）
    """
    import os
    import sys
    import pandas as pd

    os.makedirs(output_path, exist_ok=True)

    # 定义输出文件路径：评估结果始终保存到根目录，差异CSV根据参数决定
    out_path = os.path.join(output_path, f"{task_name}_evaluation.txt")
    if save_debug_files:
        debug_dir = os.path.join(output_path, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        diff_dir = debug_dir
    else:
        diff_dir = output_path
    clean_dirty_diff_path = os.path.join(diff_dir, f"{task_name}_clean_vs_dirty.csv")
    dirty_cleaned_diff_path = os.path.join(diff_dir, f"{task_name}_dirty_vs_cleaned.csv")
    clean_cleaned_diff_path = os.path.join(diff_dir, f"{task_name}_clean_vs_cleaned.csv")
    repair_errors_path = os.path.join(diff_dir, f"{task_name}_repair_errors.csv")
    unrepaired_path = os.path.join(diff_dir, f"{task_name}_unrepaired.csv")

    # 备份原始的标准输出
    original_stdout = sys.stdout

    # 将指定的属性设置为索引
    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    # 如果忽略大小写，将所有值转换为小写
    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # 重定向输出到文件（用 try/finally 确保 stdout 恢复）
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            total_true_positives = 0
            total_false_positives = 0
            total_true_negatives = 0

            clean_dirty_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Clean Value', 'Dirty Value'])
            dirty_cleaned_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value', 'Cleaned Value'])
            clean_cleaned_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Clean Value', 'Cleaned Value'])
            repair_errors = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value', 'Cleaned Value'])
            unrepaired = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value'])

            for attribute in attributes:
                clean_values = clean[attribute].apply(normalize_value)
                dirty_values = dirty[attribute].apply(normalize_value)
                cleaned_values = cleaned[attribute].apply(normalize_value)

                common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
                clean_values = clean_values.loc[common_indices]
                dirty_values = dirty_values.loc[common_indices]
                cleaned_values = cleaned_values.loc[common_indices]

                true_positives = ((cleaned_values == clean_values) & (dirty_values != cleaned_values)).sum()
                false_positives = ((cleaned_values != clean_values) & (dirty_values != cleaned_values)).sum()
                true_negatives = (dirty_values != clean_values).sum()

                mismatched_indices = dirty_values[dirty_values != clean_values].index
                clean_dirty_diff = pd.concat([clean_dirty_diff, pd.DataFrame({
                    'Attribute': attribute, 'Index': mismatched_indices,
                    'Clean Value': clean_values.loc[mismatched_indices],
                    'Dirty Value': dirty_values.loc[mismatched_indices]
                })], ignore_index=True)

                cleaned_indices = cleaned_values[cleaned_values != dirty_values].index
                dirty_cleaned_diff = pd.concat([dirty_cleaned_diff, pd.DataFrame({
                    'Attribute': attribute, 'Index': cleaned_indices,
                    'Dirty Value': dirty_values.loc[cleaned_indices],
                    'Cleaned Value': cleaned_values.loc[cleaned_indices]
                })], ignore_index=True)

                clean_cleaned_indices = cleaned_values[cleaned_values != clean_values].index
                clean_cleaned_diff = pd.concat([clean_cleaned_diff, pd.DataFrame({
                    'Attribute': attribute, 'Index': clean_cleaned_indices,
                    'Clean Value': clean_values.loc[clean_cleaned_indices],
                    'Cleaned Value': cleaned_values.loc[clean_cleaned_indices]
                })], ignore_index=True)

                repair_error_indices = cleaned_values[
                    (cleaned_values != clean_values) & (dirty_values != cleaned_values)].index
                repair_errors = pd.concat([repair_errors, pd.DataFrame({
                    'Attribute': attribute, 'Index': repair_error_indices,
                    'Clean Value': clean_values.loc[repair_error_indices],
                    'Dirty Value': dirty_values.loc[repair_error_indices],
                    'Cleaned Value': cleaned_values.loc[repair_error_indices]
                })], ignore_index=True)

                unrepaired_indices = cleaned_values[(cleaned_values == dirty_values) & (dirty_values != clean_values)].index
                unrepaired = pd.concat([unrepaired, pd.DataFrame({
                    'Attribute': attribute, 'Index': unrepaired_indices,
                    'Clean Value': clean_values.loc[unrepaired_indices],
                    'Dirty Value': dirty_values.loc[unrepaired_indices]
                })], ignore_index=True)

                total_true_positives += true_positives
                total_false_positives += false_positives
                total_true_negatives += true_negatives
                print("Attribute:", attribute, "修复正确的数据:", true_positives, "修复错误的数据:", false_positives,
                      "应该修复的数据:", true_negatives)
                print("=" * 40)

            accuracy = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
            recall = total_true_positives / total_true_negatives if total_true_negatives > 0 else 0

            print(f"修复准确率: {accuracy}")
            print(f"修复召回率: {recall}")
    finally:
        sys.stdout = original_stdout

    # 保存差异数据到 CSV 文件
    clean_dirty_diff.to_csv(clean_dirty_diff_path, index=False)
    dirty_cleaned_diff.to_csv(dirty_cleaned_diff_path, index=False)
    clean_cleaned_diff.to_csv(clean_cleaned_diff_path, index=False)
    repair_errors.to_csv(repair_errors_path, index=False)
    unrepaired.to_csv(unrepaired_path, index=False)

    if save_debug_files:
        print(f"差异文件已保存到: {diff_dir}")
    else:
        print(f"差异文件已保存到:\n{clean_dirty_diff_path}\n{dirty_cleaned_diff_path}\n{clean_cleaned_diff_path}")
    print(f"修复错误数据文件已保存到: {repair_errors_path}")
    print(f"未修复但是应该修复数据文件已保存到: {unrepaired_path}")

    return accuracy, recall


def get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', distance_func=default_distance_func, relax=False):
    """
    计算指定属性集合下的错误减少率 (EDR)，并将结果输出到文件中。
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_edr_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            total_distance_dirty_to_clean = 0
            total_distance_repaired_to_clean = 0

            for attribute in attributes:
                clean_values = clean[attribute].apply(normalize_value)
                dirty_values = dirty[attribute].apply(normalize_value)
                cleaned_values = cleaned[attribute].apply(normalize_value)

                common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
                clean_values = clean_values.loc[common_indices]
                dirty_values = dirty_values.loc[common_indices]
                cleaned_values = cleaned_values.loc[common_indices]

                distance_dirty_to_clean = distance_func(dirty_values, clean_values)
                distance_repaired_to_clean = distance_func(cleaned_values, clean_values)

                total_distance_dirty_to_clean += distance_dirty_to_clean
                total_distance_repaired_to_clean += distance_repaired_to_clean

                print(f"Attribute: {attribute}")
                print(f"Distance (Dirty to Clean): {distance_dirty_to_clean}")
                print(f"Distance (Repaired to Clean): {distance_repaired_to_clean}")
                print("=" * 40)

            if total_distance_dirty_to_clean == 0:
                edr = 0
            else:
                edr = (total_distance_dirty_to_clean - total_distance_repaired_to_clean) / total_distance_dirty_to_clean

            print(f"总的脏数据到干净数据距离: {total_distance_dirty_to_clean}")
            print(f"总的修复后数据到干净数据距离: {total_distance_repaired_to_clean}")
            print(f"错误减少率 (EDR): {edr}")
    finally:
        sys.stdout = original_stdout

    print(f"EDR 结果已保存到: {out_path}")
    return edr

def get_hybrid_distance(clean, cleaned, attributes, output_path, task_name, index_attribute='index', mse_attributes=[], w1=0.5, w2=0.5, relax=False):
    """
    计算混合距离指标，包括MSE和Jaccard距离，并将结果输出到文件中。
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_hybrid_distance_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            total_mse = 0
            total_jaccard = 0
            attribute_count = 0

            for attribute in attributes:
                clean_values = clean[attribute].apply(normalize_value).replace('empty', np.nan).dropna()
                cleaned_values = cleaned[attribute].apply(normalize_value).replace('empty', np.nan).dropna()

                if attribute in mse_attributes and not clean_values.empty and not cleaned_values.empty:
                    try:
                        clean_float = clean_values.astype(float)
                        cleaned_float = cleaned_values.astype(float)
                        # Min-Max 归一化：基于 clean 的值域，避免不同量纲导致 MSE 不可比
                        col_min = clean_float.min()
                        col_max = clean_float.max()
                        col_range = col_max - col_min
                        if col_range > 1e-10:
                            clean_norm = (clean_float - col_min) / col_range
                            cleaned_norm = (cleaned_float - col_min) / col_range
                        else:
                            # 值域为零（常量列），直接用原值
                            clean_norm = clean_float
                            cleaned_norm = cleaned_float
                        mse = mean_squared_error(clean_norm, cleaned_norm)
                    except ValueError:
                        print(f"检查你指定的属性 {attribute} 是否为数值型！")
                        mse = np.nan
                else:
                    mse = np.nan

                if not clean_values.empty and not cleaned_values.empty:
                    try:
                        common_indices = clean_values.index.intersection(cleaned_values.index)
                        jaccard = 1 - jaccard_score(
                            clean_values.loc[common_indices],
                            cleaned_values.loc[common_indices],
                            average='macro'
                        )
                    except ValueError:
                        print(f"无法计算Jaccard距离，因为 {attribute} 不是类别型数据")
                        jaccard = np.nan
                else:
                    jaccard = np.nan

                if not np.isnan(mse):
                    total_mse += mse
                if not np.isnan(jaccard):
                    total_jaccard += jaccard

                if not np.isnan(mse) or not np.isnan(jaccard):
                    attribute_count += 1

                print(f"Attribute: {attribute}, MSE: {mse}, Jaccard: {jaccard}")

            if attribute_count == 0:
                hybrid_distance = None
            else:
                avg_mse = total_mse / attribute_count if attribute_count > 0 else 0
                avg_jaccard = total_jaccard / attribute_count if attribute_count > 0 else 0
                hybrid_distance = w1 * avg_mse + w2 * avg_jaccard
                print(f"加权混合距离: {hybrid_distance}")
    finally:
        sys.stdout = original_stdout

    print(f"混合距离结果已保存到: {out_path}")
    return hybrid_distance

def get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute='index', relax=False):
    """
    计算基于条目的错误减少率 (R-EDR)，并将每条记录的距离和最终的 R-EDR 输出到文件中。
    """
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_record_based_edr_evaluation.txt")
    original_stdout = sys.stdout

    clean = clean.set_index(index_attribute, drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    if relax:
        clean = clean.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        dirty = dirty.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        cleaned = cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    total_distance_dirty_to_clean = 0
    total_distance_repaired_to_clean = 0

    # 三方 index 交集: 当 cleaned 删除了部分行时, 只在共有行上计算
    common_indices = clean.index.intersection(dirty.index).intersection(cleaned.index)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            sys.stdout = f

            for idx in common_indices:
                clean_row = clean.loc[idx].apply(normalize_value)
                dirty_row = dirty.loc[idx].apply(normalize_value)
                cleaned_row = cleaned.loc[idx].apply(normalize_value)

                distance_dirty_to_clean = record_based_distance_func(dirty_row, clean_row)
                distance_repaired_to_clean = record_based_distance_func(cleaned_row, clean_row)

                total_distance_dirty_to_clean += distance_dirty_to_clean
                total_distance_repaired_to_clean += distance_repaired_to_clean

                print(f"Record {idx}")
                print(f"Distance (Dirty to Clean): {distance_dirty_to_clean}")
                print(f"Distance (Repaired to Clean): {distance_repaired_to_clean}")
                print("=" * 40)

            if total_distance_dirty_to_clean == 0:
                r_edr = 0
            else:
                r_edr = (total_distance_dirty_to_clean - total_distance_repaired_to_clean) / total_distance_dirty_to_clean

            print(f"总的脏数据到干净数据距离: {total_distance_dirty_to_clean}")
            print(f"总的修复后数据到干净数据距离: {total_distance_repaired_to_clean}")
            print(f"基于条目的错误减少率 (R-EDR): {r_edr}")
    finally:
        sys.stdout = original_stdout

    print(f"R-EDR 结果已保存到: {out_path}")
    return r_edr

def calculate_all_metrics_TEST():
    data = {
        'index1': [1, 2, 3, 4, 5],
        'Attribute1': [1, 2, 3, 4, 5],
        'Attribute2': ['A', 'B', 'C', 'D', 'E'],
        'Attribute3': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    clean_df = pd.DataFrame(data)
    dirty_data = {
        'index1': [1, 2, 3, 4, 5],
        'Attribute1': [1, 9, 3, 4, 5],
        'Attribute2': ['A', 'B', 'X', 'D', 'E'],
        'Attribute3': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    dirty_df = pd.DataFrame(dirty_data)
    cleaned_data = {
        'index1': [1, 2, 3, 4, 5],
        'Attribute1': [1, 9, 3, 4, 5],
        'Attribute2': ['A', 'X', 'C', 'D', 'E'],
        'Attribute3': [1.1, 2.2, 3.3, 4.4, 5.7]
    }
    cleaned_df = pd.DataFrame(cleaned_data)
    attributes = ['Attribute1', 'Attribute2', 'Attribute3']
    output_path = './temp_test_output'
    task_name = 'test_task'
    results = calculate_all_metrics(clean_df, dirty_df, cleaned_df, attributes, output_path, task_name, index_attribute='index1', mse_attributes=['Attribute3'])
    print("测试结果:")
    print(f"Accuracy: {results.get('accuracy')}")
    print(f"Recall: {results.get('recall')}")
    print(f"F1 Score: {results.get('f1_score')}")
    print(f"EDR: {results.get('edr')}")
    print(f"Hybrid Distance: {results.get('hybrid_distance')}")
    print(f"R-EDR: {results.get('r_edr')}")
    print("测试通过！")

if __name__ == "__main__":
    clean_path = '../Data/1_hospitals/clean_index.csv'
    dirty_path = '../Data/1_hospitals/dirty_index.csv'
    cleaned_path = '../results/holoclean/1_hospital_ori/1_hospital_ori_repaired.csv'
    output_path = './'
    task_name = '11111'
    clean=pd.read_csv(clean_path)
    dirty=pd.read_csv(dirty_path)
    cleaned=pd.read_csv(cleaned_path)
    attributes = clean.columns.tolist()
    calculate_all_metrics(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index',
                              calculate_precision_recall=True,
                              calculate_edr=True, calculate_hybrid=True, calculate_r_edr=True)
