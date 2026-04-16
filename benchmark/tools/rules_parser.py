"""
Rules Parser - 统一规则文件解析器

解析Data/{dataset}/rules.txt文件，为不同baseline提取对应的规则。

文件格式:
[HOLOCLEAN_DC]
t1&t2&EQ(t1.attr1,t2.attr2)&IQ(t1.attr3,t2.attr3)

[UNICLEAN]
Number("attr")
AttrRelation(["lhs"], ["rhs"], "name")

[HORIZON_FD]
lhs => rhs

[MSE_ATTRIBUTES]
attr1
attr2
"""

import os
import re
from typing import List, Dict, Optional, Tuple


def parse_rules_file(rules_path: str) -> Dict[str, List[str]]:
    """
    解析统一规则文件

    Args:
        rules_path: rules.txt文件路径

    Returns:
        字典，key为section名称，value为规则列表
    """
    if not os.path.exists(rules_path):
        return {}

    sections = {}
    current_section = None

    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 检测section标题
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

    return sections


def get_holoclean_dcs(rules_path: str) -> List[str]:
    """获取HoloClean的Denial Constraints"""
    sections = parse_rules_file(rules_path)
    return sections.get('HOLOCLEAN_DC', [])


def get_uniclean_cleaners(rules_path: str) -> List[str]:
    """获取UniClean的cleaner定义字符串"""
    sections = parse_rules_file(rules_path)
    return sections.get('UNICLEAN', [])


def get_horizon_fds(rules_path: str) -> List[Tuple[str, str]]:
    """
    获取Horizon的功能依赖规则

    Returns:
        列表，每项为(lhs, rhs)元组
    """
    sections = parse_rules_file(rules_path)
    fds = []

    for line in sections.get('HORIZON_FD', []):
        # 支持 => 和 ⇒ 两种格式
        if '=>' in line:
            parts = line.split('=>')
        elif '⇒' in line:
            parts = line.split('⇒')
        else:
            continue

        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            fds.append((lhs, rhs))

    return fds


def get_mse_attributes(rules_path: str) -> List[str]:
    """获取MSE评估属性列表"""
    sections = parse_rules_file(rules_path)
    return sections.get('MSE_ATTRIBUTES', [])


def parse_uniclean_cleaner(cleaner_str: str):
    """
    解析UniClean cleaner字符串为可执行对象

    Args:
        cleaner_str: 如 'Number("age")' 或 'AttrRelation(["a"], ["b"], "0")'

    Returns:
        UniClean cleaner对象
    """
    # 动态导入UniClean cleaners
    try:
        from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
        from SampleScrubber.cleaner.multiple import AttrRelation
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Methods', 'UniClean'))
        from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
        from SampleScrubber.cleaner.multiple import AttrRelation

    # 安全执行cleaner定义
    local_vars = {
        'Number': Number,
        'Pattern': Pattern,
        'Outlier': Outlier,
        'Date': Date,
        'AttrRelation': AttrRelation
    }

    try:
        return eval(cleaner_str, {"__builtins__": {}}, local_vars)
    except Exception as e:
        print(f"Warning: Failed to parse cleaner '{cleaner_str}': {e}")
        return None


def get_uniclean_cleaner_objects(rules_path: str) -> List:
    """
    获取UniClean cleaner对象列表

    Args:
        rules_path: rules.txt文件路径

    Returns:
        cleaner对象列表
    """
    cleaner_strs = get_uniclean_cleaners(rules_path)
    cleaners = []

    for cleaner_str in cleaner_strs:
        cleaner = parse_uniclean_cleaner(cleaner_str)
        if cleaner:
            cleaners.append(cleaner)

    return cleaners


def write_holoclean_dc_file(rules_path: str, output_path: str) -> str:
    """
    从统一规则文件提取HoloClean DC并写入单独文件

    Args:
        rules_path: 统一规则文件路径
        output_path: 输出DC文件路径

    Returns:
        输出文件路径
    """
    dcs = get_holoclean_dcs(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for dc in dcs:
            f.write(dc + '\n')

    return output_path


def write_horizon_fd_file(rules_path: str, output_path: str) -> str:
    """
    从统一规则文件提取Horizon FD并写入单独文件

    Args:
        rules_path: 统一规则文件路径
        output_path: 输出FD文件路径

    Returns:
        输出文件路径
    """
    fds = get_horizon_fds(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for lhs, rhs in fds:
            f.write(f"{lhs} => {rhs}\n")

    return output_path


# 便捷函数
def get_dataset_rules(dataset_name: str, data_dir: str = 'Data') -> Dict[str, List[str]]:
    """
    获取指定数据集的所有规则

    Args:
        dataset_name: 数据集名称
        data_dir: 数据根目录

    Returns:
        规则字典
    """
    rules_path = os.path.join(data_dir, dataset_name, 'rules.txt')
    return parse_rules_file(rules_path)


if __name__ == '__main__':
    # 测试
    import sys
    if len(sys.argv) > 1:
        rules_path = sys.argv[1]
    else:
        rules_path = 'Data/beers/rules.txt'

    print(f"Parsing: {rules_path}")
    sections = parse_rules_file(rules_path)

    for section, rules in sections.items():
        print(f"\n[{section}]")
        for rule in rules:
            print(f"  {rule}")
