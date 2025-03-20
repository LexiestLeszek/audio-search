import pydub
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from math import sqrt
import random

# Установка параметров для графиков
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def load_audio_file(file_path):
    # Загрузка аудиофайла
    audio_segment = pydub.AudioSegment.from_file(file_path, format="wav")
    audio_segment = audio_segment.set_channels(1)  # Преобразование в моно
    sample_rate = audio_segment.frame_rate  # Получение частоты дискретизации
    num_samples = audio_segment.frame_count()  # Получение количества семплов
    # Нормализация семплов в диапазон [-1,1]
    samples = np.divide(audio_segment.get_array_of_samples(), 
                        audio_segment.max_possible_amplitude)
    return samples, sample_rate

def compute_stft(samples, sample_rate):
    # Вычисление краткосрочного преобразования Фурье
    window_size_in_seconds = 50/1000.0  # Длительность окна 50 мс
    window_hop_size_in_seconds = 10/1000.0  # Сдвиг окна 10 мс
    window_overlap_in_seconds = window_size_in_seconds - window_hop_size_in_seconds

    window_size_in_samples = np.ceil(window_size_in_seconds*sample_rate)
    window_overlap_in_samples = np.ceil(window_overlap_in_seconds*sample_rate)

    f, t, Zxx = signal.stft(samples, window='hann', nfft=8192,
                            nperseg=window_size_in_samples,
                            noverlap=window_overlap_in_samples)
    
    # Вычисление dB магнитуды
    stft_in_db = 20*np.log10(np.abs(Zxx)/np.max(np.abs(Zxx)))
    return f, t, stft_in_db

def find_anchors(stft_in_db, t, f):
    # Поиск якорей в спектрограмме
    num_bands = 25
    delta_t = 10  # 100 мс, но в единицах времени
    
    anchors = []
    
    for i in range(num_bands):  # 25 полос по частоте
        for j in range(t.size//delta_t):  # Все отрезки по времени
            starting_point = [i*(f.size//25), j*(t.size//100)]  # Начало отрезка
            max_coor = starting_point
            max_value = stft_in_db[starting_point[0], starting_point[1]]

            # Поиск локальных максимумов внутри отрезка
            for k in range(f.size//25):
                for l in range(delta_t):
                    if(max_value < stft_in_db[starting_point[0] + k, starting_point[1] + l]):
                        max_value = stft_in_db[starting_point[0] + k, starting_point[1] + l]
                        max_coor = [starting_point[0] + k, starting_point[1] + l]
            
            # Добавление якоря
            anchors.append([t[max_coor[1]]/sample_rate, f[max_coor[0]]*sample_rate/1000])
    
    return anchors

def compute_hashes(anchors):
    # Вычисление хэшей из якорей
    hashes = []
    
    for anchor in anchors:
        for target in anchors:
            if(target[0] > anchor[0]+0.1 and 
               target[0] < anchor[0] + 0.6 and 
               target[1] > anchor[1]/sqrt(2) and 
               target[1] < anchor[1]*sqrt(2)):
                hashes.append([anchor[0], anchor[1], target[1], target[0]-anchor[0]])
    
    return hashes

def compare_fingerprints(hashes1, hashes2):
    # Сравнение отпечатков
    matches = []
    
    for hash1 in hashes1:
        for hash2 in hashes2:
            if(abs(hash1[1] - hash2[1]) + abs(hash1[2] - hash2[2]) + abs(hash1[3] - hash2[3]) == 0):
                matches.append([hash1[0], hash2[0]])
    
    return matches

# Основная программа
if __name__ == "__main__":
    # Загрузка аудиофайла
    samples, sample_rate = load_audio_file("Q1.wav")
    
    # Вычисление STFT
    f, t, stft_in_db = compute_stft(samples, sample_rate)
    
    # Поиск якорей
    anchors = find_anchors(stft_in_db, t, f)
    
    # Вычисление хэшей
    hashes = compute_hashes(anchors)
    
    # Визуализация якорей
    fig, ax = plt.subplots()
    for anchor in anchors:
        plt.scatter(anchor[0], anchor[1], marker="x", c="black")
    plt.title("Constellation map")
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    plt.show()
    
    # Сравнение отпечатков
    # Извлечение 1-секундного сегмента из аудиофайла
    segment_samples = samples[6*sample_rate:7*sample_rate]
    
    # Вычисление STFT для сегмента
    f_segment, t_segment, stft_in_db_segment = compute_stft(segment_samples, sample_rate)
    
    # Поиск якорей для сегмента
    anchors_segment = find_anchors(stft_in_db_segment, t_segment, f_segment)
    
    # Вычисление хэшей для сегмента
    hashes_segment = compute_hashes(anchors_segment)
    
    # Сравнение отпечатков
    matches = compare_fingerprints(hashes_segment, hashes_segment)
    
    # Визуализация совпадений
    plt.scatter([item[0] for item in matches], [item[1] for item in matches], marker="x", c="black")
    plt.title('Scatterplot of matching hash locations. Diagonal present')
    plt.xlabel('Time in query signal (s)')
    plt.ylabel('Time in query signal (s)')
    plt.show()
