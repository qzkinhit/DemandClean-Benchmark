"""
datasets_config.py - 数据集配置文件

定义每个数据集的特征列、目标列、排除列、任务类型等信息。
所有baseline脚本应引用此配置文件，确保一致性。
"""

# 数据集配置
DATASETS_CONFIG = {
    'adult': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'income',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],  # 无需额外排除的列
        'feature_columns': [
            'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
        ],
        'rules_file': None,
        'description': 'UCI Adult Census Income数据集'
    },

    'beers': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'style',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': ['id', 'beer_name', 'brewery_id', 'brewery_name', 'city', 'state'],
        'feature_columns': ['ounces', 'abv', 'ibu'],
        'rules_file': None,
        'description': 'Craft Beers啤酒数据集'
    },

    'bike': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'cnt',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': ['dteday'],  # 日期列不作为特征
        'feature_columns': [
            'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered'
        ],
        'rules_file': 'rules.txt',
        'description': 'Bike Sharing自行车租赁数据集'
    },

    'breast_cancer': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'class',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': [
            'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
        ],
        'rules_file': 'rules.txt',
        'description': 'Wisconsin Breast Cancer乳腺癌诊断数据集'
    },

    'har': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'gt',
        'task_type': 'clustering',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': ['x', 'y', 'z'],
        'rules_file': 'rules.txt',
        'description': 'Human Activity Recognition人类活动识别数据集'
    },

    'mercedes': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'y',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': [],  # X0-X385都是特征
        'feature_columns': 'auto',  # 自动检测: 除index和y之外的所有列
        'rules_file': 'rules.txt',
        'description': 'Mercedes-Benz汽车测试时间预测数据集'
    },

    'nasa': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'sound_pressure_level',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': ['frequency', 'angle', 'chord_length', 'velocity', 'thickness'],
        'rules_file': 'rules.txt',
        'description': 'NASA Airfoil Self-Noise机翼噪声数据集'
    },

    'smartfactory': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'labels',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': [
            'i_w_blo_weg', 'o_w_blo_power', 'o_w_blo_voltage',
            'i_w_bhl_weg', 'o_w_bhl_power', 'o_w_bhl_voltage',
            'i_w_bhr_weg', 'o_w_bhr_power', 'o_w_bhr_voltage',
            'i_w_bru_weg', 'o_w_bru_power', 'o_w_bru_voltage',
            'i_w_hr_weg', 'o_w_hr_power', 'o_w_hr_voltage',
            'i_w_hl_weg', 'o_w_hl_power', 'o_w_hl_voltage'
        ],
        'rules_file': 'rules.txt',
        'description': 'Smart Factory智能工厂设备状态数据集'
    },

    'soilmoisture': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'soil_moisture',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': ['datetime'],  # 时间列不作为特征
        'feature_columns': 'auto',  # 自动检测: 除index, datetime, soil_moisture之外的所有列
        'rules_file': 'rules.txt',
        'description': 'Hyperspectral Soil Moisture高光谱土壤湿度数据集'
    }
}


def get_dataset_config(dataset_name: str) -> dict:
    """获取数据集配置"""
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"未知数据集: {dataset_name}. 可用数据集: {list(DATASETS_CONFIG.keys())}")
    return DATASETS_CONFIG[dataset_name]


def get_all_datasets() -> list:
    """获取所有数据集名称"""
    return list(DATASETS_CONFIG.keys())


def get_dataset_path(dataset_name: str, base_path: str = 'Data') -> dict:
    """获取数据集的完整路径"""
    config = get_dataset_config(dataset_name)
    import os
    return {
        'dirty_path': os.path.join(base_path, dataset_name, config['dirty_file']),
        'clean_path': os.path.join(base_path, dataset_name, config['clean_file']),
        'rules_path': os.path.join(base_path, dataset_name, config['rules_file']) if config['rules_file'] else None
    }


def print_dataset_info(dataset_name: str = None):
    """打印数据集信息"""
    if dataset_name:
        config = get_dataset_config(dataset_name)
        print(f"\n=== {dataset_name} ===")
        print(f"  描述: {config['description']}")
        print(f"  任务类型: {config['task_type']}")
        print(f"  目标列: {config['label_column']}")
        print(f"  索引列: {config['index_column']}")
        print(f"  排除列: {config['exclude_columns']}")
        if config['feature_columns'] == 'auto':
            print(f"  特征列: 自动检测")
        else:
            print(f"  特征列: {config['feature_columns']}")
    else:
        print("\n=== 所有数据集 ===")
        for name in get_all_datasets():
            config = DATASETS_CONFIG[name]
            print(f"  {name}: {config['task_type']}, label={config['label_column']}")


if __name__ == '__main__':
    # 测试配置
    print_dataset_info()
    print("\n" + "=" * 50)
    print_dataset_info('beers')
