from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def main():
    # WAVファイルを読み込み
    sample_rate, data = wavfile.read("synthesized_wave.wav")

    # データをfloatに変換（正規化）
    data_original = data.astype(np.float32) / 32767.0

    print("Data length:", len(data_original))
    print("Sample rate:", sample_rate)

    # 周波数解析用にHanning窓を適用
    data_windowed = data_original * np.hanning(len(data_original))

    # FFTを実行（窓適用版で周波数解析）
    fft_result = np.fft.rfft(data_windowed)
    freqs = np.fft.rfftfreq(len(data_original), 1 / sample_rate)

    # 振幅スペクトルを計算
    magnitude = np.abs(fft_result)

    # shape確認
    print("Frequencies shape:", freqs.shape)
    print("Magnitude shape:", magnitude.shape)

    # スペクトルをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Spectrum of Synthesized Wave")
    plt.grid(True)
    plt.show()

    # ピーク周波数を特定（上位4つのピーク）
    peaks = np.argsort(magnitude)[-4:]
    print("Peaks indices:", peaks)
    print("Peaks magnitudes:", magnitude[peaks])
    peak_freqs = freqs[peaks]
    print("Detected peak frequencies (Hz):", peak_freqs)

    # ローパスフィルタ（窓なしデータを使用）
    cutoff_freq = 1500  # カットオフ周波数（Hz）
    fft_filtered = np.fft.rfft(data_original)  # 形状: (N//2 + 1,)
    freqs = np.fft.rfftfreq(len(data_original), d=1 / sample_rate)  # 形状: (N//2 + 1,)

    # フィルタリング
    fft_filtered[np.abs(freqs) > cutoff_freq] = 0
    # フィルタ後のスペクトルをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.abs(fft_filtered))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Filtered FFT Spectrum (Low-pass)")
    plt.grid(True)
    plt.show()

    # 逆FFTでフィルタ後の信号を取得
    filtered_data = np.fft.irfft(fft_filtered)
    # filtered_data = filtered_data / np.max(np.abs(filtered_data)) * 0.5
    wavfile.write(
        "filtered_wave.wav", sample_rate, (filtered_data * 32767).astype(np.int16)
    )
    print("Filtered WAVファイル 'filtered_wave.wav' が作成されました。")


if __name__ == "__main__":
    main()
