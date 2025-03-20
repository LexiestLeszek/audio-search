# Импортирование необходимых библиотек
import pydub
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from math import sqrt
import random

def setup_visualization_params():
    """
    # Настройка параметров визуализации для графиков
    """
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize':'x-large',
              'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

def load_audio(file_path):
    """
    # Загрузка аудиофайла и подготовка данных
    """
    audio_segment = pydub.AudioSegment.from_file(file_path, format="wav")
    audio_segment = audio_segment.set_channels(1)  # Преобразование в моно
    sample_rate = audio_segment.frame_rate  # Получение частоты дискретизации
    num_samples = audio_segment.frame_count()  # Получение количества отсчетов
    
    # Нормализация данных в диапазоне [-1,1]
    sound_data = np.divide(audio_segment.get_array_of_samples(), 
                         audio_segment.max_possible_amplitude)
    
    return sound_data, sample_rate, num_samples

def display_waveform(sound_data, sample_rate, num_samples, title):
    """
    # Визуализация волновой формы аудиосигнала
    """
    fig, ax = plt.subplots()
    plt.plot(np.arange(num_samples)/sample_rate, sound_data)
    ax.set_title(title)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    plt.grid()

def compute_spectrogram(sound_data, sample_rate):
    """
    # Вычисление спектрограммы с помощью STFT
    """
    # Параметры окна для STFT
    window_size_seconds = 50/1000.0     # 50мс длительность окна
    window_hop_seconds = 10/1000.0      # 10мс шаг перекрытия
    window_overlap_seconds = window_size_seconds - window_hop_seconds
    
    window_size_samples = np.ceil(window_size_seconds * sample_rate)
    window_overlap_samples = np.ceil(window_overlap_seconds * sample_rate)
    
    # Вычисление STFT
    freq, time_points, Zxx = signal.stft(sound_data, window='hann', nfft=8192,
                                     nperseg=window_size_samples,
                                     noverlap=window_overlap_samples)
    
    # Преобразование в децибелы
    stft_db = 20 * np.log10(np.abs(Zxx) / np.max(np.abs(Zxx)))
    
    return freq, time_points, stft_db

def plot_spectrogram(freq, time_points, stft_db, sample_rate):
    """
    # Отображение спектрограммы
    """
    fig, ax = plt.subplots()
    plt.pcolormesh(time_points/sample_rate, freq*sample_rate/1000, stft_db, 
                   vmin=-100, vmax=0, cmap='plasma')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(bottom=0, top=5)
    fig.tight_layout()

def find_anchor_points(freq, time_points, stft_db, sample_rate, num_bands=25, time_delta=10, show_plot=True):
    """
    # Поиск якорных точек в спектрограмме (точек с максимальной энергией)
    """
    anchors = []
    
    if show_plot:
        fig, ax = plt.subplots()
        plt.pcolormesh(time_points/sample_rate, freq*sample_rate/1000, stft_db, 
                       vmin=-100, vmax=0, cmap='plasma')
    
    for i in range(num_bands):     # 25 частотных полос
        for j in range(time_points.size//time_delta):  # временные интервалы
            start_box = [i*(freq.size//25), j*(time_points.size//100)]  # левый нижний угол блока
            max_position = start_box
            max_energy = stft_db[start_box[0], start_box[1]]
    
            # Перебор точек внутри блока для поиска максимальной энергии
            for k in range(freq.size//25):
                for l in range(time_delta):
                    if (start_box[0] + k < stft_db.shape[0] and 
                        start_box[1] + l < stft_db.shape[1] and
                        max_energy < stft_db[start_box[0] + k, start_box[1] + l]):
                        max_energy = stft_db[start_box[0] + k, start_box[1] + l]
                        max_position = [start_box[0] + k, start_box[1] + l]
            
            # Отметка найденного максимума
            if show_plot:
                plt.scatter(time_points[max_position[1]]/sample_rate, freq[max_position[0]]*sample_rate/1000, marker="x", c="black")
            anchors.append([time_points[max_position[1]]/sample_rate, freq[max_position[0]]*sample_rate/1000])
    
    if show_plot:
        ax.set_ylabel('Frequency (kHz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(bottom=0, top=5)
        fig.tight_layout()
    
    return anchors

def show_constellation_map(anchors, max_time=10):
    """
    # Отображение карты созвездий из якорных точек
    """
    fig, ax = plt.subplots()
    for anchor in anchors:
        plt.scatter(anchor[0], anchor[1], marker="x", c="black")
    
    plt.title("Constellation map")
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(left=0, right=max_time)
    ax.set_ylim(bottom=0, top=5)
    fig.tight_layout()

def visualize_target_zones(anchors):
    """
    # Демонстрация целевых зон для случайных якорей
    """
    fig, ax = plt.subplots()
    
    # Выбор трех случайных якорей
    random_indices = []
    for i in range(3):
        idx = random.randint(1, min(1000, len(anchors)-1))
        random_indices.append(idx)
    
    for idx in random_indices:
        anchor = anchors[idx]
        for target in anchors:
            if(target[0] > anchor[0]+0.1 and 
               target[0] < anchor[0] + 0.6 and 
               target[1] > anchor[1]/sqrt(2) and 
               target[1] < anchor[1]*sqrt(2)):
                plt.scatter(anchor[0], anchor[1], marker="X", c="black")
                plt.scatter(target[0], target[1], marker="x", c="blue")
    
    plt.title("Examples of anchors inside the target region of another anchor")
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()

def create_fingerprint_hashes(anchors):
    """
    # Создание хешей для аудиоотпечатка из якорных точек
    """
    hashes = []
    for anchor in anchors:
        for target in anchors:
            # Целевая зона: 100-600мс после якоря и в диапазоне полуоктавы по частоте
            if(target[0] > anchor[0]+0.1 and 
               target[0] < anchor[0] + 0.6 and 
               target[1] > anchor[1]/sqrt(2) and 
               target[1] < anchor[1]*sqrt(2)):
                hashes.append([anchor[0], anchor[1], target[1], target[0]-anchor[0]])
    
    return hashes

def compare_self_fingerprint(segment_hashes):
    """
    # Сравнение отпечатка с самим собой (для проверки)
    """
    matches = []
    for hash1 in segment_hashes:
        for hash2 in segment_hashes:
            if(abs(hash1[1] - hash2[1]) + abs(hash1[2] - hash2[2]) + abs(hash1[3] - hash2[3]) == 0):
                matches.append([hash1[0], hash2[0]])
    
    plt.figure()
    plt.scatter([item[0] for item in matches], [item[1] for item in matches], marker="x", c="black")
    plt.title('Scatterplot of matching hash locations. Diagonal present')
    plt.xlabel('Time in query signal (s)')
    plt.ylabel('Time in query signal (s)')
    plt.show()
    
    return matches

def compare_fingerprints(full_file_hashes, query_hashes, max_time=10):
    """
    # Сравнение отпечатков для поиска совпадений
    """
    match_times = []
    match_pairs = []
    
    for hash1 in full_file_hashes:
        for hash2 in query_hashes:
            if(abs(hash1[1] - hash2[1]) + abs(hash1[2] - hash2[2]) + abs(hash1[3] - hash2[3]) == 0):
                match_times.append(hash1[0])
                match_pairs.append([hash1[0], hash2[0]])
    
    plt.figure()
    plt.scatter([item[0] for item in match_pairs], [item[1] for item in match_pairs], marker="x", c="black")
    plt.xlim(0, max_time)
    plt.title('Scatterplot of matching hash locations. Diagonal present')
    plt.xlabel('Full soundfile time (s)')
    plt.ylabel('Segment soundfile time (s)')
    plt.show()
    
    plt.figure()
    plt.hist(match_times, bins=100, range=[0, max_time])
    plt.title('Histogram of the times of the matches for the segment')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.show()
    
    return match_times, match_pairs

def extract_audio_segment(sound_data, sample_rate, start_time, duration):
    """
    # Извлечение сегмента аудио по заданному времени начала и длительности
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    return sound_data[start_sample:end_sample]

def main():
    """
    # Основная функция для демонстрации работы алгоритма аудиопоиска
    """
    # Настройка параметров графика
    setup_visualization_params()
    
    # Загрузка аудиофайла
    file_path = "Q1.wav"
    sound_data, sample_rate, num_samples = load_audio(file_path)
    
    # Отображение формы волны
    display_waveform(sound_data, sample_rate, num_samples, file_path)
    
    # Вычисление спектрограммы
    freq, time_points, stft_db = compute_spectrogram(sound_data, sample_rate)
    
    # Отображение спектрограммы
    plot_spectrogram(freq, time_points, stft_db, sample_rate)
    
    # Вычисление якорных точек для полного аудиофайла
    anchors = find_anchor_points(freq, time_points, stft_db, sample_rate)
    
    # Отображение карты созвездий
    show_constellation_map(anchors)
    
    # Демонстрация целевых зон
    visualize_target_zones(anchors)
    
    # Вычисление хешей для полного аудиофайла
    full_file_hashes = create_fingerprint_hashes(anchors)
    
    # Извлечение сегмента из аудиофайла (6-7 секунды)
    segment_data = extract_audio_segment(sound_data, sample_rate, 6, 1)
    
    # Вычисление спектрограммы для сегмента
    segment_freq, segment_time, segment_stft = compute_spectrogram(segment_data, sample_rate)
    
    # Вычисление якорных точек для сегмента
    segment_anchors = find_anchor_points(segment_freq, segment_time, segment_stft, sample_rate)
    
    # Вычисление хешей для сегмента
    segment_hashes = create_fingerprint_hashes(segment_anchors)
    
    # Сравнение отпечатков сегмента с самим собой
    compare_self_fingerprint(segment_hashes)
    
    # Сравнение отпечатков сегмента с полным аудиофайлом
    match_times, match_pairs = compare_fingerprints(full_file_hashes, segment_hashes)
    
    print("# Ограничение: из-за подхода к созданию отпечатков, Shazam работает только с точными совпадениями.")
    print("# Поскольку у него нет музыкального понимания песен в базе данных, кавер-версия песни или запрос напеванием не будут правильно определены.")

if __name__ == "__main__":
    main()
