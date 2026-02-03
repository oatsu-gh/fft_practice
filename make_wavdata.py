import numpy as np
from scipy.io import wavfile

# サンプルレート
SAMPLE_RATE = 44100

# 周波数と振幅の比率
FREQUENCIES = [500, 1000, 2000, 5000]
AMPLITUDES = [10, 5, 1, 1]

# 波形の長さ（秒）
DURATION = 10

# ファイル名
WAV_FILENAME = "synthesized_wave.wav"


def main(frequencies, amplitudes, sample_rate, duration):
    # 時間軸の作成
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 合成波形の初期化
    wave = np.zeros_like(t)

    # 各周波数のサイン波を合成
    for freq, amp in zip(frequencies, amplitudes, strict=True):
        wave += amp * np.sin(2 * np.pi * freq * t)

    # 波形を正規化（最大振幅を0.5に）
    wave = wave / np.max(np.abs(wave)) * 0.5

    # WAVファイルとして保存（16ビット整数に変換）
    wavfile.write("synthesized_wave.wav", sample_rate, (wave * 32767).astype(np.int16))

    print("WAVファイル 'synthesized_wave.wav' が作成されました。")


if __name__ == "__main__":
    main(FREQUENCIES, AMPLITUDES, SAMPLE_RATE, DURATION)
