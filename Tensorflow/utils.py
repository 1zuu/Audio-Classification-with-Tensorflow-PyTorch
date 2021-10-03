import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf 

from variables import *

def read_audio(file_path):
    audio_file = tf.io.read_file(file_path)
    audio_signal, sampling_rate = tf.audio.decode_wav(audio_file)
    return sampling_rate, audio_signal

def load_audio_files():
    audio_classes = os.listdir(data_dir)
    audio_files = []
    aduio_labels = []
    for audio_class in tqdm(audio_classes):
        audio_class_dir = os.path.join(data_dir, audio_class)
        audio_files_in_class = os.listdir(audio_class_dir)
        for audio_file in audio_files_in_class:
            audio_file_path = os.path.join(audio_class_dir, audio_file)
            if audio_file_path.endswith('.wav'):
                sampling_rate, audio_signal = read_audio(audio_file_path)
                sampling_rate = sampling_rate.numpy().squeeze()
                audio_signal = audio_signal.numpy().squeeze()

                audio_files.append((sampling_rate, audio_signal))
                aduio_labels.append(audio_class)
            
    return audio_files, aduio_labels

def calculate_audio_length(row):
    sampling_rate = row['sampling_rate']
    audio_signal = row['audio_signal']
    audio_length = audio_signal.shape[0] / sampling_rate
    return audio_length

def create_audio_dataframe():
    audio_files, audio_labels = load_audio_files()
    audio_df = pd.DataFrame(audio_files, columns=['sampling_rate', 'audio_signal'])
    audio_df['audio_length'] = audio_df.apply(calculate_audio_length, axis=1)
    audio_df['audio_label'] = audio_labels
    return audio_df

def plot_lengths(audio_df):
    classes = audio_df['audio_label'].unique().tolist()
    class_dict = audio_df.groupby('audio_label')['audio_length'].mean()

    _, ax = plt.subplots()
    ax.set_title('Class Distribution by Audio Length', y=1.08)
    ax.pie(class_dict, labels=classes, autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()

def fast_fourier_transform(signal, sampling_rate):
    length_signal  = len(signal)
    freq = np.fft.rfftfreq(length_signal, d = 1/sampling_rate)
    fft_signal = np.abs(np.fft.rfft(signal)) / length_signal # Magnitude
    return fft_signal, freq

def plot_signals(audio_df):
    signals = {}
    fft = {}
    filter_banks = {}
    mfccs = {}

    classes = audio_df['audio_label'].unique().tolist()
    for c in classes:
        audio_df_c = audio_df[audio_df['audio_label'] == c]
        audio_df_c = audio_df_c.sample(n=1)
        signals[c] = audio_df_c['audio_signal'].values[0].squeeze()

        sampling_rate = audio_df_c['sampling_rate'].values[0].squeeze()
        fft[c] = fast_fourier_transform(signals[c], sampling_rate)
    print(signals)

audio_df = create_audio_dataframe()
# plot_lengths(audio_df)
plot_signals(audio_df)