import numpy as np
import scipy.io
import librosa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import pandas as pd

# 設定視覺化風格
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_music_data():
    """載入重新採樣的音樂數據"""
    music_file = Path("data/music_resample_seeg_2024_2048hz.mat")
    mat_data = scipy.io.loadmat(music_file)
    
    music_data = {}
    y_resample = mat_data['y_resample']
    
    for i in range(3):  # 三首歌
        song_data = y_resample['data'][0, i].flatten()
        song_name = y_resample['name'][0, i][0]
        song_env = y_resample['env'][0, i].flatten()
        
        music_data[i] = {
            'name': song_name,
            'audio': song_data,
            'envelope': song_env,
            'fs': mat_data['target_fs'][0, 0]
        }
    
    return music_data

def load_seeg_data():
    """載入所有病人的 SEEG 數據"""
    seeg_files = list(Path("data").glob("seeg_s*_5mm_110824.mat"))
    seeg_data = {}
    
    for seeg_file in seeg_files:
        patient_id = seeg_file.stem.split('_')[1]  # 提取病人編號 (s011, s038, etc.)
        
        mat_data = scipy.io.loadmat(seeg_file)
        
        # 載入每首歌的 epoch 數據
        epoch_all = mat_data['epoch_all']
        patient_data = {}
        
        for song_idx in range(3):
            epoch_data = epoch_all[0, song_idx]
            patient_data[song_idx] = {
                'data': epoch_data,
                'channels': mat_data['ch_select'].flatten(),
                'fs': mat_data['fs'][0, 0]
            }
        
        seeg_data[patient_id] = patient_data
    
    return seeg_data

def plot_music_overview(music_data, save_plots=True):
    """Plot music data overview"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('🎵 Music Data Overview (2048 Hz)', fontsize=16, fontweight='bold')
    
    song_names = ['Brahms Piano Concerto', 'Lost Stars', 'Doraemon']
    
    for i, (song_idx, data) in enumerate(music_data.items()):
        # 時間軸
        time = np.arange(len(data['audio'])) / data['fs']
        
        # 音頻波形
        axes[i, 0].plot(time, data['audio'], alpha=0.7, linewidth=0.5)
        axes[i, 0].set_title(f'{song_names[i]} - Audio Waveform')
        axes[i, 0].set_xlabel('Time (seconds)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 包絡線
        axes[i, 1].plot(time, data['envelope'], color='red', linewidth=1)
        axes[i, 1].set_title(f'{song_names[i]} - Envelope')
        axes[i, 1].set_xlabel('Time (seconds)')
        axes[i, 1].set_ylabel('Envelope Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('music_overview.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_seeg_overview(seeg_data, music_data, save_plots=True):
    """Plot SEEG data overview"""
    n_patients = len(seeg_data)
    fig, axes = plt.subplots(n_patients, 3, figsize=(18, 4*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('SEEG Brain Signal Overview', fontsize=16, fontweight='bold')
    
    song_names = ['Brahms Piano Concerto', 'Lost Stars', 'Doraemon']
    patient_names = list(seeg_data.keys())
    
    for p_idx, (patient_id, patient_data) in enumerate(seeg_data.items()):
        for song_idx in range(3):
            seeg_epoch = patient_data[song_idx]['data']
            fs = patient_data[song_idx]['fs']
            
            # 計算平均跨所有電極的訊號
            if len(seeg_epoch.shape) == 3:
                avg_signal = np.mean(seeg_epoch, axis=0)[:, 0]  # 平均電極，選第一個條件
            else:
                avg_signal = seeg_epoch[:, 0]
            
            time = np.arange(len(avg_signal)) / fs
            
            axes[p_idx, song_idx].plot(time, avg_signal, alpha=0.8, linewidth=0.5)
            axes[p_idx, song_idx].set_title(f'Patient {patient_id} - {song_names[song_idx]}')
            axes[p_idx, song_idx].set_xlabel('Time (seconds)')
            axes[p_idx, song_idx].set_ylabel('SEEG Signal (μV)')
            axes[p_idx, song_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('seeg_overview.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def calculate_correlation_analysis(seeg_data, music_data):
    """計算 SEEG 與音樂包絡線的相關性分析"""
    correlation_results = {}
    
    for patient_id, patient_data in seeg_data.items():
        correlation_results[patient_id] = {}
        
        for song_idx in range(3):
            seeg_epoch = patient_data[song_idx]['data']
            music_envelope = music_data[song_idx]['envelope']
            
            # 確保時間長度一致
            min_length = min(seeg_epoch.shape[-2], len(music_envelope))
            
            if len(seeg_epoch.shape) == 3:
                # 多電極情況
                n_channels = seeg_epoch.shape[0]
                correlations = []
                
                for ch in range(n_channels):
                    seeg_signal = seeg_epoch[ch, :min_length, 0]  # 使用第一個條件
                    env_signal = music_envelope[:min_length]
                    
                    # 計算相關係數
                    corr, p_value = pearsonr(seeg_signal, env_signal)
                    correlations.append({'channel': ch, 'correlation': corr, 'p_value': p_value})
                
                correlation_results[patient_id][song_idx] = correlations
            else:
                # 單電極情況
                seeg_signal = seeg_epoch[:min_length, 0]
                env_signal = music_envelope[:min_length]
                
                corr, p_value = pearsonr(seeg_signal, env_signal)
                correlation_results[patient_id][song_idx] = [{'channel': 0, 'correlation': corr, 'p_value': p_value}]
    
    return correlation_results

def plot_correlation_heatmap(correlation_results, save_plots=True):
    """Plot correlation heatmap"""
    song_names = ['Brahms', 'Lost Stars', 'Doraemon']
    
    # 準備數據矩陣
    patients = list(correlation_results.keys())
    correlation_matrix = []
    
    for patient_id in patients:
        patient_corrs = []
        for song_idx in range(3):
            # 取該病人該首歌所有電極的平均相關係數
            correlations = correlation_results[patient_id][song_idx]
            avg_corr = np.mean([c['correlation'] for c in correlations])
            patient_corrs.append(avg_corr)
        correlation_matrix.append(patient_corrs)
    
    correlation_matrix = np.array(correlation_matrix)
    
    # 繪製熱力圖
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=song_names,
                yticklabels=[f'Patient {p}' for p in patients],
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('SEEG-Music Envelope Correlation Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Music')
    plt.ylabel('Patient')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return correlation_matrix

def plot_detailed_correlation(seeg_data, music_data, patient_id='s011', song_idx=0, save_plots=True):
    """Plot detailed correlation analysis"""
    song_names = ['Brahms Piano Concerto', 'Lost Stars', 'Doraemon']
    
    # 獲取數據
    seeg_epoch = seeg_data[patient_id][song_idx]['data']
    music_envelope = music_data[song_idx]['envelope']
    fs = seeg_data[patient_id][song_idx]['fs']
    
    # 確保時間長度一致
    min_length = min(seeg_epoch.shape[-2], len(music_envelope))
    
    if len(seeg_epoch.shape) == 3:
        seeg_signal = np.mean(seeg_epoch, axis=0)[:min_length, 0]  # 平均所有電極
    else:
        seeg_signal = seeg_epoch[:min_length, 0]
    
    env_signal = music_envelope[:min_length]
    time = np.arange(min_length) / fs
    
    # 計算相關係數
    corr, p_value = pearsonr(seeg_signal, env_signal)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(f'Detailed Correlation Analysis - Patient {patient_id} - {song_names[song_idx]}', 
                 fontsize=14, fontweight='bold')
    
    # 音樂包絡線
    axes[0].plot(time, env_signal, color='red', linewidth=1)
    axes[0].set_title('Music Envelope')
    axes[0].set_ylabel('Envelope Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # SEEG 訊號
    axes[1].plot(time, seeg_signal, color='blue', linewidth=0.5, alpha=0.8)
    axes[1].set_title('SEEG Signal (Average)')
    axes[1].set_ylabel('SEEG Amplitude (μV)')
    axes[1].grid(True, alpha=0.3)
    
    # 散點圖顯示相關性
    axes[2].scatter(env_signal, seeg_signal, alpha=0.1, s=1)
    axes[2].set_xlabel('Music Envelope Amplitude')
    axes[2].set_ylabel('SEEG Amplitude (μV)')
    axes[2].set_title(f'Correlation Scatter Plot (r = {corr:.3f}, p = {p_value:.2e})')
    axes[2].grid(True, alpha=0.3)
    
    # 添加回歸線
    z = np.polyfit(env_signal, seeg_signal, 1)
    p = np.poly1d(z)
    axes[2].plot(env_signal, p(env_signal), "r--", alpha=0.8)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'detailed_correlation_{patient_id}_{song_idx}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def spectrogram_correlation_analysis(seeg_data, music_data, patient_id='s011', song_idx=0, save_plots=True):
    """分析音樂 spectrogram 與 SEEG 訊號的時頻相關性"""
    song_names = ['Brahms Piano Concerto', 'Lost Stars', 'Doraemon']
    
    # 獲取數據
    seeg_epoch = seeg_data[patient_id][song_idx]['data']
    music_audio = music_data[song_idx]['audio']
    music_envelope = music_data[song_idx]['envelope']
    fs = seeg_data[patient_id][song_idx]['fs']
    
    # 確保時間長度一致
    min_length = min(seeg_epoch.shape[-2], len(music_audio), len(music_envelope))
    
    if len(seeg_epoch.shape) == 3:
        seeg_signal = np.mean(seeg_epoch[:, :min_length, :], axis=(0, 2))
    else:
        seeg_signal = np.mean(seeg_epoch[:min_length, :], axis=1)
    
    music_signal = music_audio[:min_length]
    env_signal = music_envelope[:min_length]
    
    # 計算音樂的 spectrogram
    f_music, t_music, Sxx_music = signal.spectrogram(
        music_signal, fs, 
        window='hann', 
        nperseg=int(fs*0.1),  # 100ms window
        noverlap=int(fs*0.05),  # 50ms overlap
        nfft=1024
    )
    
    # 計算 SEEG 的 spectrogram
    f_seeg, t_seeg, Sxx_seeg = signal.spectrogram(
        seeg_signal, fs,
        window='hann',
        nperseg=int(fs*0.1),  # 100ms window
        noverlap=int(fs*0.05),  # 50ms overlap
        nfft=1024
    )
    
    # 限制頻率範圍到有意義的範圍
    freq_max = 50  # Hz
    freq_idx_music = f_music <= freq_max
    freq_idx_seeg = f_seeg <= freq_max
    
    f_music_filtered = f_music[freq_idx_music]
    f_seeg_filtered = f_seeg[freq_idx_seeg]
    Sxx_music_filtered = Sxx_music[freq_idx_music, :]
    Sxx_seeg_filtered = Sxx_seeg[freq_idx_seeg, :]
    
    # 調整時間軸以匹配
    min_time_bins = min(Sxx_music_filtered.shape[1], Sxx_seeg_filtered.shape[1])
    Sxx_music_filtered = Sxx_music_filtered[:, :min_time_bins]
    Sxx_seeg_filtered = Sxx_seeg_filtered[:, :min_time_bins]
    t_music = t_music[:min_time_bins]
    t_seeg = t_seeg[:min_time_bins]
    
    # 計算頻率帶的相關性
    freq_bands = {
        'Delta (1-4 Hz)': (1, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-12 Hz)': (8, 12),
        'Beta (12-30 Hz)': (12, 30),
        'Gamma (30-50 Hz)': (30, 50)
    }
    
    band_correlations = {}
    
    for band_name, (low_freq, high_freq) in freq_bands.items():
        # 找到對應的頻率索引
        music_band_idx = (f_music_filtered >= low_freq) & (f_music_filtered <= high_freq)
        seeg_band_idx = (f_seeg_filtered >= low_freq) & (f_seeg_filtered <= high_freq)
        
        if np.any(music_band_idx) and np.any(seeg_band_idx):
            # 計算該頻段的平均功率
            music_band_power = np.mean(Sxx_music_filtered[music_band_idx, :], axis=0)
            seeg_band_power = np.mean(Sxx_seeg_filtered[seeg_band_idx, :], axis=0)
            
            # 計算相關係數
            if len(music_band_power) > 1 and len(seeg_band_power) > 1:
                corr, p_value = pearsonr(music_band_power, seeg_band_power)
                band_correlations[band_name] = {'correlation': corr, 'p_value': p_value}
    
    # 繪製結果
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])
    
    fig.suptitle(f'Spectrogram Correlation Analysis - Patient {patient_id} - {song_names[song_idx]}', 
                 fontsize=16, fontweight='bold')
    
    # 音樂 spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(t_music, f_music_filtered, 10*np.log10(Sxx_music_filtered + 1e-12), 
                         shading='gouraud', cmap='viridis')
    ax1.set_title('Music Spectrogram')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    # SEEG spectrogram
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(t_seeg, f_seeg_filtered, 10*np.log10(Sxx_seeg_filtered + 1e-12), 
                         shading='gouraud', cmap='plasma')
    ax2.set_title('SEEG Spectrogram')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=ax2, label='Power (dB)')
    
    # 頻段相關性條形圖
    ax3 = fig.add_subplot(gs[0, 2])
    band_names = list(band_correlations.keys())
    correlations = [band_correlations[band]['correlation'] for band in band_names]
    colors = ['red' if corr < 0 else 'blue' for corr in correlations]
    
    bars = ax3.bar(range(len(band_names)), correlations, color=colors, alpha=0.7)
    ax3.set_title('Frequency Band Correlations')
    ax3.set_xlabel('Frequency Bands')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_xticks(range(len(band_names)))
    ax3.set_xticklabels([name.split('(')[0].strip() for name in band_names], rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 添加數值標籤
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # 時域信號比較
    time_axis = np.arange(min_length) / fs
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time_axis, music_signal, alpha=0.7, linewidth=0.5, label='Music Signal')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_axis, env_signal, color='red', linewidth=1, label='Music Envelope')
    ax4.set_title('Music Signal & Envelope')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Music Amplitude', color='blue')
    ax4_twin.set_ylabel('Envelope Amplitude', color='red')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(time_axis, seeg_signal, alpha=0.7, linewidth=0.5, color='purple')
    ax5.set_title('SEEG Signal')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('SEEG Amplitude (μV)')
    ax5.grid(True, alpha=0.3)
    
    # 包絡線與 SEEG 相關性
    ax6 = fig.add_subplot(gs[1, 2])
    env_seeg_corr, env_seeg_p = pearsonr(env_signal, seeg_signal)
    ax6.scatter(env_signal, seeg_signal, alpha=0.1, s=1)
    ax6.set_xlabel('Music Envelope Amplitude')
    ax6.set_ylabel('SEEG Amplitude (μV)')
    ax6.set_title(f'Envelope-SEEG Correlation\n(r = {env_seeg_corr:.3f}, p = {env_seeg_p:.2e})')
    ax6.grid(True, alpha=0.3)
    
    # 添加回歸線
    z = np.polyfit(env_signal, seeg_signal, 1)
    p = np.poly1d(z)
    ax6.plot(env_signal, p(env_signal), "r--", alpha=0.8)
    
    # 詳細統計信息
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    stats_text = f"Correlation Analysis Summary for {song_names[song_idx]} - Patient {patient_id}:\n\n"
    stats_text += f"Envelope-SEEG Correlation: r = {env_seeg_corr:.4f}, p = {env_seeg_p:.2e}\n\n"
    stats_text += "Frequency Band Correlations:\n"
    
    for band_name, stats in band_correlations.items():
        corr = stats['correlation']
        p_val = stats['p_value']
        significance = " *" if p_val < 0.05 else "  " if p_val < 0.1 else ""
        stats_text += f"  {band_name}: r = {corr:.4f}, p = {p_val:.2e}{significance}\n"
    
    stats_text += "\n* p < 0.05"
    
    ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'spectrogram_correlation_{patient_id}_{song_idx}.png', dpi=300, bbox_inches='tight')
    # else:
    plt.show()
    
    return band_correlations

def spectral_analysis(seeg_data, music_data, patient_id='s011', song_idx=0, save_plots=True):
    """Perform spectral analysis"""
    song_names = ['Brahms Piano Concerto', 'Lost Stars', 'Doraemon']
    
    # 獲取數據
    seeg_epoch = seeg_data[patient_id][song_idx]['data']
    music_audio = music_data[song_idx]['audio']
    fs = seeg_data[patient_id][song_idx]['fs']
    
    # 確保時間長度一致
    min_length = min(seeg_epoch.shape[-2], len(music_audio))
    
    if len(seeg_epoch.shape) == 3:
        seeg_signal = np.mean(seeg_epoch, axis=0)[:min_length, 0]
    else:
        seeg_signal = seeg_epoch[:min_length, 0]
    
    music_signal = music_audio[:min_length]
    
    # 計算功率譜密度
    freqs_seeg, psd_seeg = signal.welch(seeg_signal, fs, nperseg=2048)
    freqs_music, psd_music = signal.welch(music_signal, fs, nperseg=2048)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Spectral Analysis - Patient {patient_id} - {song_names[song_idx]}', 
                 fontsize=14, fontweight='bold')
    
    # 時域信號
    time = np.arange(min_length) / fs
    axes[0, 0].plot(time, music_signal, alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Music Signal (Time Domain)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, seeg_signal, alpha=0.7, linewidth=0.5)
    axes[0, 1].set_title('SEEG Signal (Time Domain)')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 頻域分析
    axes[1, 0].semilogy(freqs_music, psd_music)
    axes[1, 0].set_title('Music Power Spectral Density')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_xlim(0, 100)  # 只看 100 Hz 以下
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(freqs_seeg, psd_seeg)
    axes[1, 1].set_title('SEEG Power Spectral Density')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power Spectral Density')
    axes[1, 1].set_xlim(0, 100)  # 只看 100 Hz 以下
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'spectral_analysis_{patient_id}_{song_idx}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def comprehensive_analysis():
    """執行完整的分析流程"""
    print("🔄 開始載入數據...")
    
    # 載入數據
    music_data = load_music_data()
    seeg_data = load_seeg_data()
    
    print(f"✅ 載入完成！音樂數據: {len(music_data)} 首歌，SEEG 數據: {len(seeg_data)} 位病人")
    
    # 1. 音樂數據總覽
    print("\n📊 繪製音樂數據總覽...")
    plot_music_overview(music_data)
    
    # 2. SEEG 數據總覽
    print("📊 繪製 SEEG 數據總覽...")
    plot_seeg_overview(seeg_data, music_data)
    
    # 3. 相關性分析
    print("🔗 進行相關性分析...")
    correlation_results = calculate_correlation_analysis(seeg_data, music_data)
    correlation_matrix = plot_correlation_heatmap(correlation_results)
    
    # 4. 詳細相關性分析 (以第一個病人的第一首歌為例)
    print("📊 繪製詳細相關性分析...")
    first_patient = list(seeg_data.keys())[0]
    plot_detailed_correlation(seeg_data, music_data, first_patient, 0)
    
    # 5. 頻譜分析
    print("🌊 進行頻譜分析...")
    spectral_analysis(seeg_data, music_data, first_patient, 0)
    
    # 6. Spectrogram 相關性分析
    print("🎵 進行 Spectrogram 相關性分析...")
    spectrogram_correlations = spectrogram_correlation_analysis(seeg_data, music_data, first_patient, 2)
    
    # 打印統計結果
    print("\n📈 相關性分析結果摘要:")
    song_names = ['Brahms', 'Lost Stars', 'Doraemon']
    
    for i, song_name in enumerate(song_names):
        print(f"\n{song_name}:")
        for j, patient_id in enumerate(seeg_data.keys()):
            corr_val = correlation_matrix[j, i]
            print(f"  Patient {patient_id}: r = {corr_val:.3f}")
    
    # 7. Spectrogram 頻段相關性摘要
    print(f"\n🎼 Spectrogram 頻段相關性結果 (Patient {first_patient}, {song_names[0]}):")
    for band_name, stats in spectrogram_correlations.items():
        corr = stats['correlation']
        p_val = stats['p_value']
        significance = " *" if p_val < 0.05 else ""
        print(f"  {band_name}: r = {corr:.4f}, p = {p_val:.2e}{significance}")
    
    print("\n🎉 分析完成！所有圖表已儲存。")

def analyze_object_array(obj_array, indent="       "):
    """分析 MATLAB object 陣列"""
    for i, obj in enumerate(obj_array.flat):
        print(f"{indent}Item {i}:")
        if hasattr(obj, 'shape'):
            print(f"{indent}  Shape: {obj.shape}")
        if hasattr(obj, 'dtype'):
            print(f"{indent}  Data Type: {obj.dtype}")
        if hasattr(obj, 'min') and hasattr(obj, 'max') and obj.size > 0:
            try:
                print(f"{indent}  Min/Max: {obj.min():.6f} / {obj.max():.6f}")
            except:
                print(f"{indent}  Min/Max: Cannot compute")

def analyze_structured_array(struct_array, indent="       "):
    """分析結構化陣列"""
    print(f"{indent}Fields: {list(struct_array.dtype.names)}")
    for field in struct_array.dtype.names:
        print(f"{indent}Field '{field}':")
        field_data = struct_array[field]
        if field_data.dtype == 'O':  # Object array
            for i, item in enumerate(field_data.flat):
                print(f"{indent}  Item {i}:")
                if hasattr(item, 'shape'):
                    print(f"{indent}    Shape: {item.shape}")
                if hasattr(item, 'dtype'):
                    print(f"{indent}    Data Type: {item.dtype}")
                if hasattr(item, 'min') and hasattr(item, 'max') and item.size > 0:
                    try:
                        print(f"{indent}    Min/Max: {item.min():.6f} / {item.max():.6f}")
                    except:
                        print(f"{indent}    Min/Max: Cannot compute")
        else:
            print(f"{indent}  Shape: {field_data.shape}")
            print(f"{indent}  Data Type: {field_data.dtype}")

def analyze_data_folder():
    """分析 data 資料夾中的所有檔案"""
    data_dir = Path("data")
    
    print("=== Data Folder Analysis ===\n")
    
    # 分析 WAV 檔案
    wav_files = list(data_dir.glob("*.wav"))
    print("🎵 Audio Files (WAV):")
    for wav_file in wav_files:
        try:
            y, sr = librosa.load(wav_file, sr=None)
            duration = len(y) / sr
            print(f"  📁 {wav_file.name}")
            print(f"     - Shape: {y.shape}")
            print(f"     - Sample Rate: {sr} Hz")
            print(f"     - Duration: {duration:.2f} seconds")
            print(f"     - Data Type: {y.dtype}")
            print(f"     - Min/Max: {y.min():.6f} / {y.max():.6f}")
            print()
        except Exception as e:
            print(f"  ❌ Error loading {wav_file.name}: {e}")
    
    # 分析 MAT 檔案
    mat_files = list(data_dir.glob("*.mat"))
    print("\nMATLAB Files (MAT):")
    for mat_file in mat_files:
        try:
            print(f"  📁 {mat_file.name}")
            mat_data = scipy.io.loadmat(mat_file)
            
            # 過濾掉 MATLAB 內建變數
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            
            for key in data_keys:
                data = mat_data[key]
                print(f"     - Variable: '{key}'")
                print(f"       Shape: {data.shape}")
                print(f"       Data Type: {data.dtype}")
                
                # 處理不同類型的資料
                if data.dtype == 'O':  # Object array
                    print(f"       Content: Object Array")
                    analyze_object_array(data)
                elif data.dtype.names:  # Structured array
                    print(f"       Content: Structured Array")
                    analyze_structured_array(data)
                else:  # Regular numeric array
                    if hasattr(data, 'min') and hasattr(data, 'max') and data.size > 0:
                        try:
                            print(f"       Min/Max: {data.min():.6f} / {data.max():.6f}")
                        except:
                            print(f"       Min/Max: Cannot compute")
                
                print()
        except Exception as e:
            print(f"  ❌ Error loading {mat_file.name}: {e}")
    
    print("\n=== Summary ===")
    print(f"Total WAV files: {len(wav_files)}")
    print(f"Total MAT files: {len(mat_files)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # 執行完整的 SEEG 音樂分析
        comprehensive_analysis()
    else:
        # 執行原始的數據結構分析
        analyze_data_folder()