"""
Methods package containing various data cleaning and preparation methods.

DemandClean-Benchmark Methods Package
面向下游机器学习任务的数据清洗方法集合

包含方法分类:
1. 面向数据的清洗方法: HoloClean, Baran_Raha, Horizon, UniClean, Lopster
2. 面向模型的清洗方法: ActiveClean, BoostClean
3. 数据准备方法: ctxpipe, MLImputer, SimpleImputer
"""

# 已完整实现的方法
# from .Horizon import *
# from .Baran_Raha import *

# 框架实现或需要安装额外依赖的方法
# from .HoloClean import HoloCleanWrapper
# from .UniClean import UniCleanWrapper
# from .Lopster import LopsterWrapper
# from .ActiveClean import ActiveCleanWrapper
# from .BoostClean import BoostCleanWrapper
# from .MLImputer import MLImputerWrapper
# from .SimpleImputer import SimpleImputerWrapper

__all__ = [
    # 面向数据的清洗方法
    'HoloClean',
    'Baran_Raha',
    'Horizon',
    'UniClean',
    'Lopster',
    # 面向模型的清洗方法
    'ActiveClean',
    'BoostClean',
    # 数据准备方法
    'ctxpipe',
    'MLImputer',
    'SimpleImputer',
]

__version__ = '1.0.0'
