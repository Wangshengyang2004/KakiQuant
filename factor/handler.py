"""
传入DataFrame, 计算因子并返回DataFrame
"""
from utils.check_gpu import use_cudf

if use_cudf():
    import cudf as pd
else:
    import pandas as pd
