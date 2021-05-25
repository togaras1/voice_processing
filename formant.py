import sys

import numpy as np
import wave
import scipy.io.wavfile
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import glob

from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

#音声関係のライブラリ
import pyaudio
import struct

font = {'family':'monospace', 'size':'9'}
mpl.rc('font', **font)

# フォルマント周波数の次元数
dim = 5
# lpcの次元数
lpcOrder = 20

# 出典
# https://ameblo.jp/erikki-chann/entry-10410227225.html
idealformant = np.array([
    [338.4,2127.9,3036.4,3696.1],
    [502.2,1965.3,2803.7,3739.3],
    [718.2,1145.9,2775.6,3758.5],
    [496.5,859.1,2754.8,3476.7],
    [326.1,1435.1,2511,3400.1]])
idealvowel = np.array(['i','e','a','o','u'])

# 参考サイト
# http://blog.wktk.co.jp/ja/entry/2013/06/14/formant-detection-with-numpy
def wavread(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16") / 32768.0  # (-1, 1)に正規化
    wf.close()
    return x, float(fs)

# 参考サイト
# https://qiita.com/kkdd/items/77a421366f39ea1103e1
def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

# 参考サイト
# http://aidiary.hatenablog.com/entry/20120415/1334458954
def autocorr(x, nlags=None):
    """自己相関関数を求める
    x:     信号
    nlags: 自己相関関数のサイズ（lag=0からnlags-1まで）
           引数がなければ（lag=0からlen(x)-1まですべて）
    """
    N = len(x)
    if nlags == None: nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r

# 参考サイト
# http://aidiary.hatenablog.com/entry/20120415/1334458954
def LevinsonDurbin(r, lpcOrder):
    """Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める"""
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]

# 理想状態の母音と距離だけで認識してみる
def detect_ideal(f):
    distance = []
    f=f[:dim]
    for i in idealformant:
        distance.append(np.square(f-i).sum())

    return idealvowel[np.argmin(distance)]

class aaaa:
    def __init__(self):
        #マイクインプット設定
        self.CHUNK=2048            #1度に読み取る音声のデータ幅
        self.RATE=16000             #サンプリング周波数
        self.update_seconds=0.05      #更新時間[ms]
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    frames_per_buffer=self.CHUNK)

        #音声データの格納場所(プロットデータ)
        self.data=np.zeros(self.CHUNK)
        self.axis=np.fft.fftfreq(len(self.data), d=1.0/self.RATE)

    def load(self,filename):
        self.clf = joblib.load(filename)
        return

    # 参考
    # http://momijiame.tumblr.com/post/114751531866/python-iris-%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%82%92%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88%E3%83%99%E3%82%AF%E3%82%BF%E3%83%BC%E3%83%9E%E3%82%B7%E3%83%B3%E3%81%A7%E5%88%86%E9%A1%9E%E3%81%97%E3%81%A6%E3%81%BF%E3%82%8B
    def learn(self):
        labels = []
        vectors = []
        ds = glob.glob('dataset/*')
        for d in ds:
            print(d)
            f = self.lpcFile(d)
            vectors.append(f[:dim])
            labels.append(d.split(".")[0][-1]) # ファイル名の一文字目がラベル名

        self.clf = svm.SVC(kernel='poly')
        self.clf.fit(vectors, labels)
        filename = 'finalized_model.sav'
        joblib.dump(self.clf, filename)
        return

    def predict(self,f):
        return self.clf.predict(f)

    def lpcFile(self,name):
        # 音声をロード
        wav, fs = wavread(name)
        t = np.arange(0.0, len(wav) / fs, 1/fs)

        # 音声波形の中心部分を切り出す
        center = len(wav) // 2  # 中心のサンプル番号
        s = wav[center - 512 : center + 512]

        # プリエンファシスフィルタをかける
        p = 0.97         # プリエンファシス係数
        s = preEmphasis(s, p)

        # ハミング窓をかける
        s = s * np.hamming(len(s))

        # LPC係数を求める
        r = autocorr(s, lpcOrder + 1)

        a, e  = LevinsonDurbin(r, lpcOrder)

        # フォルマント検出( by Tasuku SUENAGA a.k.a. gunyarakun )

        # 平方根
        rts = np.roots(a)
        # 共役解のうち、虚部が負のものは取り除く
        rts = rts[np.imag(rts) >= 0]

        # 根から角度を計算
        angz = np.arctan2(np.imag(rts), np.real(rts))
        # 角度の低い順にソート
        sorted_index = angz.argsort()
        # 角度からフォルマント周波数を計算
        freqs = angz.take(sorted_index) * (fs / (2 * np.pi))
        # 角度からフォルマントの帯域幅も計算
        bw = -1 / 2 * (fs / (2 * np.pi)) * np.log(np.abs(rts.take(sorted_index)))
        formant = []
        for i in range(len(freqs)):
            if freqs[i] > 90 and bw[i] < 400:
                formant.append(freqs[i])
        return formant

    def update(self):
        self.data = self.AudioInput()
        #self.data=np.append(self.data,self.AudioInput())
        #if len(self.data)/1024 > 5:
        #    self.data=self.data[1024:]
        self.lpc(self.data) #mfcc(self.data)
        plt.draw()
        return 0

    def lpc(self,data):
        # 左のフォルマント解析のグラフ
        plt.subplot(1,2,1)
        plt.xlim([0,1200])
        plt.ylim([0,3000])
        plt.grid()
        plt.title(idealvowel)
        # 右のケプストラムグラフ
        plt.subplot(1,2,2)
        plt.xlabel("Frequency (Hz)")
        plt.ylim((-60,30))
        plt.xlim((-100, 8100))
        plt.grid()

        t = np.arange(0.0, len(data) / self.RATE, 1 / self.RATE)
        # プリエンファシスフィルタをかける
        p = 0.97         # プリエンファシス係数
        s = preEmphasis(data, p)
        # ハミング窓をかける
        s = s * np.hamming(len(s))
        # LPC係数を求める
        r = autocorr(s, lpcOrder + 1)
        a, e  = LevinsonDurbin(r, lpcOrder)

        # LPCで前向き予測した信号を求める
        predicted = np.copy(s)
        # 過去lpcOrder分から予測するので開始インデックスはlpcOrderから
        # それより前は予測せずにオリジナルの信号をコピーしている
        for i in range(lpcOrder, len(predicted)):
            predicted[i] = 0.0
            for j in range(1, lpcOrder):
                predicted[i] -= a[j] * s[i - j]
        # オリジナルの信号をプロット
        # plt.plot(t, s)
        # LPCで前向き予測した信号をプロット
        # plt.plot(t, predicted,"r",alpha=0.4)
        # plt.xlabel("Time (s)")
        # plt.xlim((-0.001, t[-1]+0.001))
        # plt.grid()

        # LPC係数の振幅スペクトルを求める
        nfft = 2048   # FFTのサンプル数
        fscale = np.fft.fftfreq(nfft, d = 1.0 / self.RATE)[:nfft//2]
        # オリジナル信号の対数スペクトル
        spec = np.abs(np.fft.fft(s, nfft))
        logspec = 20 * np.log10(spec)
        # LPC対数スペクトル
        w, h = scipy.signal.freqz(np.sqrt(e), a, nfft, "whole")
        lpcspec = np.abs(h)
        loglpcspec = 20 * np.log10(lpcspec)

        plt.subplot(1,2,2)
        plt.plot(fscale, logspec[:nfft//2])
        plt.plot(fscale, loglpcspec[:nfft//2], "r", linewidth=2)

        rts = np.roots(a)
        rts = rts[np.imag(rts) >= 0]

        # 根から角度を計算
        angz = np.arctan2(np.imag(rts), np.real(rts))
        sorted_index = angz.argsort()
        freqs = angz.take(sorted_index) * (self.RATE / (2 * np.pi))
        bw = -1 / 2 * (self.RATE / (2 * np.pi)) * np.log(np.abs(rts.take(sorted_index)))

        formant = []
        dbs = []
        for i in range(len(freqs)):
            if freqs[i] > 90 and bw[i] < 400:
                formant.append(freqs[i])
                dbs.append(loglpcspec[i])

        plt.subplot(1,2,1)
        plt.plot(formant[0],formant[1],'o',color=cm.Blues((dbs[0]+70)/50))

        interval = 100
        xx = np.arange(0,1200+interval,interval)
        yy = np.arange(0,3000+interval,interval)

        zz = []
        for y in yy:
            zz.append([])
            for x in xx:
                zz[-1].append(np.where(self.clf.predict([([x,y,formant[2],formant[3],formant[4]])[:dim]])[0]==idealvowel)[0][0])
        plt.contourf(xx,yy,zz,cmap=plt.cm.bone, alpha=0.2)

        if dbs[0] < -30:
            print("sil")
        else:
            print(self.predict(np.reshape(formant[:dim],(1,dim)))[0])
            #print(detect_ideal(formant))

        return 0

    def AudioInput(self):
        ret=self.stream.read(self.CHUNK, False)    #音声の読み取り(バイナリ) CHUNKが大きいとここで時間かかる
        #バイナリ → 数値(int16)に変換
        #32768.0=2^16で割ってるのは正規化(絶対値を1以下にすること)
        ret=np.frombuffer(ret, dtype="int16")/32768.0
        return ret

args = sys.argv
b = aaaa()
if len(args) > 1:
    b.load(args[1])
else:
    b.learn()
plt.ion()

print("now start")
while True:
    plt.clf()
    b.update()
    plt.pause(0.000000000001)

plt.close()
