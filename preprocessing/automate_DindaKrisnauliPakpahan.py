"""
automate_DindaKrisnauliPakpahan.py
=====================
Script otomatisasi preprocessing untuk dataset Heart Failure Prediction.
Menjalankan seluruh pipeline preprocessing secara otomatis dan
menghasilkan dataset yang siap dilatih.

Usage:
    python automate_DindaKrisnauliPakpahan.py
    python automate_DindaKrisnauliPakpahan.py --input ../heart_raw/heart.csv --output ../heart_preprocessing

Dataset: Heart Failure Prediction (fedesoriano, Kaggle)
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')



# 1. LOAD DATA
def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath: Path ke file CSV dataset.

    Returns:
        DataFrame berisi data mentah.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✅ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df



# 2. VALIDASI DATA
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memvalidasi kolom wajib dan tipe data dasar.

    Args:
        df: DataFrame input.

    Returns:
        DataFrame yang sudah divalidasi.

    Raises:
        ValueError: Jika kolom wajib tidak ditemukan.
    """
    required_cols = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
        'Oldpeak', 'ST_Slope', 'HeartDisease'
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom tidak ditemukan: {missing_cols}")

    print(f"✅ Validasi kolom: semua {len(required_cols)} kolom wajib tersedia")
    return df



# 3. TANGANI ZERO / NILAI TIDAK VALID
def handle_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengganti nilai 0 yang tidak valid secara medis dengan median kolom.
    - Cholesterol = 0  → tidak mungkin secara medis
    - RestingBP   = 0  → tidak mungkin secara medis

    Args:
        df: DataFrame input.

    Returns:
        DataFrame dengan zero values yang sudah ditangani.
    """
    df = df.copy()

    # Cholesterol
    zero_chol = (df['Cholesterol'] == 0).sum()
    if zero_chol > 0:
        median_chol = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].median()
        df['Cholesterol'] = df['Cholesterol'].replace(0, median_chol)
        print(f"  Cholesterol: {zero_chol} nilai 0 → diganti median ({median_chol:.1f})")

    # RestingBP
    zero_bp = (df['RestingBP'] == 0).sum()
    if zero_bp > 0:
        median_bp = df.loc[df['RestingBP'] != 0, 'RestingBP'].median()
        df['RestingBP'] = df['RestingBP'].replace(0, median_bp)
        print(f"  RestingBP  : {zero_bp} nilai 0 → diganti median ({median_bp:.1f})")

    print("✅ Penanganan zero values selesai")
    return df



# 4. TANGANI MISSING VALUES
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values.
    - Numerik    → isi dengan median
    - Kategorikal → isi dengan modus

    Args:
        df: DataFrame input.

    Returns:
        DataFrame tanpa missing values.
    """
    df = df.copy()
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print("✅ Tidak ada missing values ditemukan")
        return df

    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue
        if df[col].dtype in ['float64', 'int64']:
            fill_val = df[col].median()
            strategy = "median"
        else:
            fill_val = df[col].mode()[0]
            strategy = "modus"
        df[col].fillna(fill_val, inplace=True)
        print(f"  {col}: {n_miss} missing → diisi {strategy} ({fill_val})")

    print(f"✅ Missing values ditangani: {total_missing} nilai diisi")
    return df



# 5. TANGANI OUTLIER (IQR CAPPING)
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Capping (Winsorization).
    Nilai di luar [Q1-1.5*IQR, Q3+1.5*IQR] di-clip ke batasnya.

    Args:
        df: DataFrame input.

    Returns:
        DataFrame dengan outlier yang sudah ditangani.
    """
    df = df.copy()
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_outliers > 0:
            print(f"  {col:15s}: {n_outliers} outlier di-cap ke [{lower:.2f}, {upper:.2f}]")

    print("✅ Penanganan outlier (IQR Capping) selesai")
    return df



# 6. ENCODING FITUR KATEGORIKAL
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding fitur kategorikal.
    - Binary encoding  : Sex, ExerciseAngina
    - One-Hot Encoding : ChestPainType, RestingECG, ST_Slope

    Args:
        df: DataFrame input.

    Returns:
        DataFrame dengan fitur yang sudah di-encode.
    """
    df = df.copy()

    # Binary encoding
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    print("  Binary encoding: Sex (M=1,F=0), ExerciseAngina (Y=1,N=0)")

    # One-Hot Encoding
    ohe_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Konversi bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"  One-Hot Encoding: {ohe_cols}")
    print(f"✅ Encoding selesai. Total kolom: {df.shape[1]}")
    return df



# 7. SPLIT & SCALING
def split_and_scale(df: pd.DataFrame):
    """
    Melakukan train-test split (80:20 stratified) dan
    StandardScaler pada fitur numerik.

    Args:
        df: DataFrame yang sudah di-encode.

    Returns:
        Tuple (X_train, X_test, y_train, y_test, scaler)
    """
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling fitur numerik
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print(f"✅ Split: train={X_train.shape}, test={X_test.shape}")
    print(f"✅ StandardScaler diterapkan pada: {num_cols}")
    return X_train, X_test, y_train, y_test, scaler



# 8. SIMPAN HASIL
def save_preprocessed(X_train, X_test, y_train, y_test, output_dir: str):
    """
    Menyimpan dataset hasil preprocessing ke folder output.

    Args:
        X_train, X_test: Fitur train dan test.
        y_train, y_test: Target train dan test.
        output_dir: Path folder output.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['HeartDisease'] = y_train.values

    test_df = X_test.copy()
    test_df['HeartDisease'] = y_test.values

    train_path = os.path.join(output_dir, 'heart_train.csv')
    test_path = os.path.join(output_dir, 'heart_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✅ Dataset disimpan ke: {output_dir}")
    print(f"   → heart_train.csv : {train_df.shape}")
    print(f"   → heart_test.csv  : {test_df.shape}")


# MAIN PIPELINE
def run_preprocessing(input_path: str, output_dir: str):
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan.

    Args:
        input_path: Path ke file CSV raw dataset.
        output_dir: Path folder untuk menyimpan hasil preprocessing.
    """
    print("=" * 55)
    print("  PIPELINE PREPROCESSING - HEART FAILURE PREDICTION")
    print("=" * 55)

    print("\n[1/7] Memuat data...")
    df = load_data(input_path)

    print("\n[2/7] Validasi data...")
    df = validate_data(df)

    print("\n[3/7] Menangani zero values...")
    df = handle_zero_values(df)

    print("\n[4/7] Menangani missing values...")
    df = handle_missing_values(df)

    print("\n[5/7] Menangani outlier...")
    df = handle_outliers(df)

    print("\n[6/7] Encoding fitur kategorikal...")
    df = encode_features(df)

    print("\n[7/7] Split & scaling...")
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)

    print("\n[OUTPUT] Menyimpan hasil preprocessing...")
    save_preprocessed(X_train, X_test, y_train, y_test, output_dir)

    print("\n" + "=" * 55)
    print("  ✅ PREPROCESSING SELESAI!")
    print("=" * 55)



# ENTRYPOINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing - Heart Failure Prediction Dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../heart_raw/heart.csv',
        help='Path ke file CSV raw (default: ../heart_raw/heart.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../heart_preprocessing',
        help='Folder output dataset hasil preprocessing (default: ../heart_preprocessing)'
    )
    args = parser.parse_args()

    run_preprocessing(args.input, args.output)
