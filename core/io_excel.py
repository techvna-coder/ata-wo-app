# core/io_excel.py - FIXED VERSION
import pandas as pd
from .mapping import MAP_IN2INTERNAL, MAP_ALIASES, INCOLS

# CÁC CỘT NỘI BỘ CHO PHÉP THÊM KHI THIẾU (KHÔNG GỒM WO_Number)
_INTERNAL_REQUIRED = INCOLS

def load_wo_excel(file):
    """
    Load WO Excel file và chuẩn hóa columns.
    
    Args:
        file: File path hoặc file-like object
        
    Returns:
        pd.DataFrame với columns chuẩn hóa
    """
    df = pd.read_excel(file, dtype=str)
    
    # Đổi tên theo map (nếu khớp)
    cols = {}
    for k, v in MAP_IN2INTERNAL.items():
        if k in df.columns:
            cols[k] = v
    df = df.rename(columns=cols)

    # Chỉ bù các cột nội bộ BẮT BUỘC (không tạo WO_Number nếu thiếu)
    for need in _INTERNAL_REQUIRED:
        if need not in df.columns:
            df[need] = None

    # Parse ngày
    for c in ("Open_Date", "Close_Date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df

def write_result(df, path="WO_ATA_checked.xlsx"):
    """
    Ghi DataFrame ra Excel.
    
    Args:
        df: DataFrame kết quả
        path: Đường dẫn file output
        
    Returns:
        str: path của file đã ghi
    """
    df.to_excel(path, index=False)
    return path
