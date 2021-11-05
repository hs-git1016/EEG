import mne
import numpy as np
import ntpath
import os
import math
import pywt
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from mne import read_annotations
from mne.io import read_raw_edf
from scipy import signal
EPOCH_SEC_SIZE=30
from glob import glob

def label_read(file_path):             #读取标签数据
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    events = root.find('SleepStages').getchildren()
    sleep_stages = [int(event.text) for event in events]
    return sleep_stages
    

def WPEnergy(data, fs=256, wavelet='db4', maxlevel=6):
    # 小波包分解
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    # 定义能量数组
    energy = []
    # 循环遍历计算四个频段对应的能量
    for iter in range(len(iter_freqs)):
        iterEnergy = 0.0
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 计算对应频段的累加和
                iterEnergy += pow(np.linalg.norm(wp[freqTree[i]].data, ord=None), 2)
        # 保存四个频段对应的能量和
        energy.append(iterEnergy)
    return energy
    # 绘制能量分布图
    # plt.plot([xLabel['name'] for xLabel in iter_freqs], energy, lw=0, marker='o')
    # plt.title('能量分布')
    # plt.show()


# 需要分析的频带及其范围
iter_freqs  = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 4.5},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta1', 'fmin': 13, 'fmax': 22},
    {'name': 'Beta2', 'fmin': 22, 'fmax': 30},
    {'name': 'Gamma', 'fmin': 30, 'fmax': 40}
]

Delta = []
Theta = []
Alpha = []
Beta = []
Gamma = []

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
mne.set_log_level(False)


shhs_base_dir = r'C:\Users\DELL\Desktop\EEG\children'
psg_files = glob(os.path.join(shhs_base_dir,'*.edf'))
ann_files = glob(os.path.join(shhs_base_dir,'*.xml'))
select_chs=['F3-M2','F4-M1','C3-M2','C4-M1','O1-M2','O2-M1']
# select_chs=['C4-M1']

psg_fnames=[]
ann_fnames=[]

for i in range(len(psg_files)):
    psg_fnames.append(psg_files[i])
    ann_fnames.append(ann_files[i])



for i in range(len(psg_fnames)):
    raw=read_raw_edf(psg_fnames[i],preload=True,stim_channel=None)
    sampling_rate=raw.info['sfreq']
    print("*******采样频率********************")
    print(sampling_rate)
    print("***********************************")
    raw_ch_df=raw.to_data_frame(scaling_time=sampling_rate)[select_chs]
    
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))
    labels=label_read(ann_fnames[i])
    print("***********************************")
    print(len(labels))
    print("***********************************")
    #Generate label and remove indice
    labels = np.hstack(labels)
   

    #Verify that we can split into 30-s epochs
    if len(raw_ch_df)%(EPOCH_SEC_SIZE*sampling_rate)!=0:  #不满足总数据/30/100
        raise Exception("Something wrong")
    n_epochs=len(raw_ch_df)/(EPOCH_SEC_SIZE*sampling_rate)#轮次
    
    #Get epochs and their corresponding labels
    x=np.asarray(np.split(raw_ch_df,n_epochs)).astype(np.float32)
    data = np.zeros((x.shape[0],x.shape[2],x.shape[1]))
    for item in range(x.shape[0]):
        data[item] = x[item].T
    print(data.shape)
    y=labels.astype(np.int32)
    assert len(x)==len(y)
    energy = []
    for item in range(data.shape[0]):
        energy_temp = []
        for j in range(data.shape[1]):
            temp = WPEnergy(data[item][j])
            energy_temp.extend(temp)
        energy.append(energy_temp)
    energy = np.array(energy)
    print(energy.shape)
    print(energy)
  
    # Save
    filepath = "C:/Users/DELL/Desktop/EEG/children"
    os.makedirs(filepath, exist_ok=True)
    filename = ntpath.basename(psg_files[i]).replace(".edf", ".npz")
    print(filename)
    save_dict={"x":energy,"y":y,"fs":sampling_rate}
    np.savez(filepath +'/'+filename, **save_dict)

    print("\n======================================================\n")
