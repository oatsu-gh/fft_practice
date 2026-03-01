#!/usr/bin/env python3
# Copyright (c) 2026 oatsu
"""
SEM 画像解析データの周波数解析用
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NM_PER_PX = 10.1  # [nm/px] 例: SEM画像の解像度が5.1nm/pxの場合
WAVELENGTH_THRESHOLD_NM = 200.0  # [nm]=[Hz-1] 例: 200nm以上の波長成分で切り分ける閾値


def csv_to_array(
    csv_path,
    columns_names: str | list[str],
    encoding: str = 'cp932',
    delimiter: str = ',',
) -> list[np.ndarray]:
    """
    CSVファイルを読み込んで2D配列に変換する関数
    """
    # columns_names が文字列の場合は1要素のリストに変換
    if isinstance(columns_names, str):
        columns_names = [columns_names]

    # CSVファイルを読み込んで指定された列を抽出し、2D配列に変換
    df = pd.read_csv(csv_path, usecols=columns_names, encoding=encoding, delimiter=delimiter)
    l_arrays = [np.array(serieses.values) for _, serieses in df.items()]

    return l_arrays


def main(
    csv_path: str,
    columns_names: str | list[str],
    *,
    nm_per_px: float,
    window: None | str = 'hanning',
    threshold_wavelength_nm: float = WAVELENGTH_THRESHOLD_NM,
):
    # CSVファイルのパスと列名を指定してデータを読み込む
    l_arrays = csv_to_array(csv_path, columns_names)
    # 元データの要素数を確認
    original_lengths = [len(array) for array in l_arrays]
    # 読み込んだデータを確認
    print('l_arrays:', l_arrays)
    # 窓関数を適用（Hanning窓）
    if window is None:
        pass
    elif window == 'hanning':
        l_arrays = [array * np.hanning(len(array)) for array in l_arrays]
    else:
        raise ValueError(f'Unsupported window type: {window}')

    # fftを実行
    l_fft_results = [np.fft.rfft(array) for array in l_arrays]
    print('l_fft_results:', l_fft_results)
    l_fft_freqs = [np.fft.rfftfreq(len(array), d=nm_per_px) for array in l_arrays]
    # print('l_fft_freqs:', l_fft_freqs)
    # 振幅スペクトルを計算
    l_magnitudes = [np.abs(fft_result) for fft_result in l_fft_results]
    # print('l_magnitudes:', l_magnitudes)
    # shape確認
    for freqs, magnitudes in zip(l_fft_freqs, l_magnitudes, strict=True):
        print('Frequencies shape:', freqs.shape)
        print('Magnitude shape  :', magnitudes.shape)

    # スペクトルをプロット
    for freqs, magnitudes in zip(l_fft_freqs, l_magnitudes, strict=True):
        # ピーク周波数を特定（上位4つのピーク）
        peaks = np.argsort(magnitudes)[-4:]
        print('Peaks indices:', peaks)
        # print('Peaks magnitudes:', magnitudes[peaks])
        peak_freqs = freqs[peaks]
        print('Detected peak frequencies (1/nm):', peak_freqs)
        print('Detected peak wavelengths (nm):', 1 / peak_freqs)
        plt.figure(figsize=(10, 6))
        plt.plot(1 / freqs, magnitudes)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Magnitude')
        plt.title('FFT Spectrum of SEM Data')
        plt.grid(True)
        plt.show()

    # 長波長成分のみ取り出す
    freq_threshold = 1 / threshold_wavelength_nm  # 周波数の閾値を波長の閾値から計算
    l_filtered_arrays = []
    # shape確認
    print('Original arrays shapes:', [array.shape for array in l_arrays])
    print('FFT frequencies shapes:', [freqs.shape for freqs in l_fft_freqs])
    print('FFT magnitudes shapes:', [magnitudes.shape for magnitudes in l_magnitudes])

    for freqs, magnitudes in zip(l_fft_freqs, l_magnitudes, strict=True):
        # 閾値以下の周波数成分をフィルタリング
        filtered_magnitudes = magnitudes[freqs <= freq_threshold]
        l_filtered_arrays.append(filtered_magnitudes)
        print('Filtered magnitudes shape:', filtered_magnitudes.shape)
        # フィルタ後のスペクトルをプロット
        plt.figure(figsize=(10, 6))
        plt.plot(1 / freqs[freqs <= freq_threshold], filtered_magnitudes)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Magnitude')
        plt.title(f'Filtered FFT Spectrum (Wavelength > {threshold_wavelength_nm} nm)')
        plt.grid(True)
        plt.show()

    # フィルタ後のスペクトルを逆FFTで取得
    l_filtered_data = [
        np.fft.irfft(filtered_magnitudes, n=original_length)
        for filtered_magnitudes, original_length in zip(
            l_filtered_arrays,
            original_lengths,
            strict=True,
        )
    ]
    print('Filtered data shapes:', [filtered_data.shape for filtered_data in l_filtered_data])


if __name__ == '__main__':
    # CSVファイルのパスと列名を指定してデータを読み込む
    csv_path = 'samples/sem_sample.csv'  # 例: SEM画像解析データのCSVファイル
    columns_names = ['y']  # 例: X列とY列を読み込む
    main(csv_path, columns_names, nm_per_px=NM_PER_PX, window='hanning')
