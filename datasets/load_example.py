import pandas as pd

def load_parquet_file(file_path: str) -> pd.DataFrame:
    """
    加载 Parquet 文件并返回 Pandas DataFrame。
    
    Args:
        file_path (str): Parquet 文件的路径。
    
    Returns:
        pd.DataFrame: 加载的 Parquet 数据。
    """
    try:
        # 使用 Pandas 的 read_parquet 方法加载 Parquet 文件
        df = pd.read_parquet(file_path)
        print(f"成功加载 Parquet 文件: {file_path}")
        return df
    except Exception as e:
        print(f"加载 Parquet 文件失败: {e}")
        return None

# 示例用法
if __name__ == "__main__":
    file_path = "/data/tzhang/dataset/geometry3k/train-00000-of-00001.parquet"  # 替换为你的 Parquet 文件路径
    df = load_parquet_file(file_path)
    import pdb; pdb.set_trace()
    if df is not None:
        print(f"数据形状: {df.shape}")
        print("数据预览:")
        print(df.head())