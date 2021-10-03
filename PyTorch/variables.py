resample_rate = 22050
signal_length = resample_rate

frame_size = 1024
hop_length = frame_size // 2
n_mels = 64

learning_rate = 0.001
epochs = 20
batch_size = 32

import os 
data_dir = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
audio_dir = data_dir + '\\data\\'