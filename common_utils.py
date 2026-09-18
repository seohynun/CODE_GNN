import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path="yelpzip_preprocessed.parquet"):
    """전처리된 데이터를 로드하고 기본 시계열 처리를 수행합니다.
    Parquet 형식을 우선적으로 지원하며, 필요 시 CSV도 로드합니다.
    """
    # 확장자가 .csv인데 .parquet 파일이 존재하면 자동으로 Parquet 로드
    if file_path.endswith(".csv"):
        parquet_path = file_path.replace(".csv", ".parquet")
        if os.path.exists(parquet_path):
            file_path = parquet_path

    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, index_col=0)
    
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.to_period("M")
    return df

def get_burst_months(df, label_col='label', threshold_sigma=1.5):
    """사기 리뷰 빈도를 바탕으로 버스트 월(Month) 리스트를 반환합니다."""
    monthly_fraud = df[df[label_col] == 1].groupby('ym').size()
    burst_thresh = monthly_fraud.mean() + threshold_sigma * monthly_fraud.std()
    burst_months = monthly_fraud[monthly_fraud >= burst_thresh].index
    return burst_months

def save_features(df, features_df, name):
    """추출된 피처를 기존 데이터프레임 구조에 맞춰 저장합니다."""
    output_path = f"features_{name}.csv"
    features_df.to_csv(output_path)
    print(f"Feature '{name}' saved to {output_path}")

def setup_plotting():
    """시각화 스타일을 통일하고 한글 깨짐 문제를 해결합니다."""
    import platform
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    
    # 1. OS별 한글 폰트 설정
    system_os = platform.system()
    font_name = ""
    if system_os == 'Windows':
        font_name = 'Malgun Gothic'
    elif system_os == 'Darwin':  # macOS
        font_name = 'AppleGothic'
    else:  # Linux (Ubuntu 등)
        # 나눔고딕이 설치되어 있다고 가정하거나 폰트 경로를 찾음
        font_name = 'NanumGothic'
        
    # 2. 폰트 적용
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 3. 스타일 설정 (스타일 설정이 폰트를 초기화할 수 있으므로 폰트 설정 후 혹은 스타일 설정 후 다시 적용)
    try:
        plt.style.use('seaborn-v0_8-muted')
        # 스타일 적용 후 폰트가 초기화될 수 있으므로 다시 설정
        rc('font', family=font_name)
    except:
        plt.style.use('ggplot')
        rc('font', family=font_name)

    # 4. Seaborn이 설치되어 있다면 Seaborn에도 폰트 설정 적용
    try:
        import seaborn as sns
        sns.set(font=font_name, rc={'axes.unicode_minus': False}, style='whitegrid')
    except ImportError:
        pass
        
    plt.rcParams["figure.figsize"] = (10, 6)
    print(f"Plotting setup complete for {system_os} (Font: {font_name})")
