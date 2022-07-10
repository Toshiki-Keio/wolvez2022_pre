import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    print(samplerate, fp, fs, gpass, gstop)
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    print(wp,ws)
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    print(N,Wn)
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y 

samplerate = 25600
x = np.arange(0, 12800) / samplerate                    # 波形生成のための時間軸の作成
data = np.random.normal(loc=0, scale=1, size=len(x))    # ガウシアンノイズを生成
data2=np.random.normal(loc=0, scale=1, size=len(x))+3
data3=np.concatenate([data[0:int(len(x)/2)],data2[0:int(len(x)/2)]])
print(data3.shape)

fp = 300 # 通過域端周波数[Hz]
fs = 600 # 阻止域端周波数[Hz]
gpass = 3 # 通過域端最大損失[dB]
gstop = 40 # 阻止域端最小損失[dB]
 
# ローパスをする関数を実行
data_lofilt = lowpass(data3, samplerate, fp, fs, gpass, gstop)
plt.plot(x,data3)
plt.plot(x,data_lofilt,'r')
plt.legend(['Raw','Lowpass'])

plt.show()