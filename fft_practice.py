from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def main():
    # WAVファイルを読み込み
    sample_rate, data = wavfile.read("synthesized_wave.wav")

    # データをfloatに変換（正規化）
    data = data.astype(np.float32) / 32767.0

    # FFTを実行
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1 / sample_rate)

    # 振幅スペクトルを計算
    magnitude = np.abs(fft_result)

    # 正の周波数のみを使用
    freqs = freqs[: len(freqs) // 2]
    magnitude = magnitude[: len(magnitude) // 2]

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
    peak_freqs = freqs[peaks]
    print("Detected peak frequencies (Hz):", peak_freqs)


if __name__ == "__main__":
    main()
