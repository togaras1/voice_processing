import sys

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as multi

#音声関係のライブラリ
import pyaudio
import struct
import pyworld as pw

class a:
    def __init__(self):
        #マイクインプット設定
        self.CHUNK=2048            #1度に読み取る音声のデータ幅
        self.RATE=16000            #サンプリング周波数
        self.update_seconds=50      #更新時間[ms]
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    frames_per_buffer=self.CHUNK,
                                    input=True,
                                    output=True)

        #音声データの格納場所(プロットデータ)
        self.data=np.zeros(self.CHUNK) # 処理するデータ
        self.buffer=np.zeros(self.CHUNK) # バッファー
        self.axis=np.fft.fftfreq(len(self.data), d=1.0/self.RATE)

    def update(self):
        # バイナリにエンコードしてストリームに書き込む
        self.stream.write(self.buffer)

        self.buffer = self.data
        self.data = self.VoiceChanger()
        #p = Pool(multi.cpu_count())
        #self.data = p.map(self.VoiceConversion, self)
        #p.close()

        #print(sp)

    def VoiceChanger(self):
        # 音変換部分
        # スペクトルの伸び率 短い方が太くなり 長いほうが細くなる
        sp_rate = 1.2
        # 周波数の伸び率
        f0_rate = 2.0

        x = self.AudioInput()
        f0, sp, ap = pw.wav2world(x, self.RATE)
        converted_sp = np.zeros_like(sp)
        for f in range(sp.shape[1]):
            if int(f/sp_rate) < sp.shape[1]:
                converted_sp[:, f] = sp[:, int(f/sp_rate)] # スペクトルの変換(縮める)
            else:
                converted_sp[:, f] = sp[:, sp.shape[1]-1] # 範囲外を補完
        sp = converted_sp
        # 周波数の変換
        f0 = f0 * f0_rate
        x = pw.synthesize(f0, sp, ap, self.RATE)
        x = struct.pack("h" * len(x), *x.astype(np.int16))
        return x

    def AudioInput(self):
        ret=self.stream.read(self.CHUNK, False)    #音声の読み取り(バイナリ) CHUNKが大きいとここで時間かかる
        #バイナリ → 数値(int16)に変換
        #32768.0=2^16で割ってるのは正規化(絶対値を1以下にすること)
        ret=np.frombuffer(ret, dtype="int16") #/32768.0
        ret=ret.astype(np.float)
        return ret

b = a()
plt.ion()
print("now start")
while b.stream.is_active():
    b.update()
