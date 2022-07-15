import numpy as np
import matplotlib.pyplot as plt
samplerate = 25600
x = np.arange(0, 12800) / samplerate                    # 波形生成のための時間軸の作成
data = np.random.normal(loc=0, scale=1, size=len(x))    # ガウシアンノイズを生成
print(data.shape)
ave_data=np.convolve(data,np.ones(100)/100, mode='valid')
plt.plot(np.arange(len(x)),data)
plt.plot(np.arange(len(ave_data)),ave_data,'r')
plt.show()