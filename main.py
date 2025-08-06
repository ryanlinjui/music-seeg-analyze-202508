import sys
import numpy as np
import scipy.io
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QComboBox, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl
import pyqtgraph as pg
from pyqtgraph import PlotWidget, InfiniteLine
import librosa
import warnings
warnings.filterwarnings('ignore')

class MusicSEEGVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.music_data = {}
        self.seeg_data = {}
        self.current_song = 0
        self.current_patient = None
        self.current_channel = 0
        self.fs = 1024  # 採樣頻率
        
        # 音樂播放相關
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.update_play_position)
        
        # 播放進度線
        self.play_line = None
        self.seeg_play_line = None
        self.dragging_play_line = False
        
        # 滑鼠追蹤垂直線
        self.mouse_vline_envelope = None
        self.mouse_vline_seeg = None
        
        # 滑鼠追蹤標籤
        self.envelope_value_label = None
        self.seeg_value_label = None
        
        # 載入數據
        self.load_data()
        
        # 設置 UI
        self.init_ui()
        
        # 初始化顯示
        self.update_channel_info()  # 初始化聲道信息
        self.update_plots()
    
    def load_data(self):
        """載入音樂和 SEEG 數據"""
        # 先載入 SEEG 數據以獲取正確的採樣頻率
        self.load_seeg_data()
        # 然後載入音樂數據，使用 SEEG 的採樣頻率
        self.load_music_data()
    
    def load_music_data(self):
        """直接從 WAV 文件載入音樂數據"""
        # 從 SEEG 數據獲取採樣頻率
        target_fs = self.fs  # 預設使用 1024
        if self.seeg_data:
            # 使用第一個病人的採樣頻率
            first_patient = list(self.seeg_data.keys())[0]
            target_fs = self.seeg_data[first_patient][0]['fs']
            print(f"📊 從 SEEG 數據獲取採樣頻率: {target_fs} Hz")
            # 更新類別採樣頻率
            self.fs = target_fs
        
        song_names = ['Brahms Piano Concerto', 'Lost Stars', 'Doraemon']
        wav_files = ['BrahmsPianoConcerto.wav', 'LostStars.wav', 'Doraemon.wav']
        
        for i in range(3):
            wav_file = Path("data") / wav_files[i]
            
            if not wav_file.exists():
                print(f"❌ WAV 文件不存在: {wav_files[i]}")
                continue
                
            try:
                print(f"🎵 載入音樂文件: {wav_files[i]}")
                
                # 嘗試使用 librosa 載入音頻，強制轉換為 float32 避免整數溢出
                try:
                    audio_stereo, sr = librosa.load(wav_file, sr=None, mono=False, dtype=np.float32)
                except Exception as librosa_error:
                    print(f"   ❌ librosa 載入失敗: {librosa_error}")
                    print(f"   🔄 嘗試使用 scipy 載入...")
                    
                    # 備用方案：使用 scipy 載入
                    from scipy.io import wavfile
                    sr, audio_scipy = wavfile.read(wav_file)
                    
                    # 轉換為 float32 並正規化
                    if audio_scipy.dtype == np.int16:
                        audio_stereo = audio_scipy.astype(np.float32) / 32768.0
                    elif audio_scipy.dtype == np.int32:
                        audio_stereo = audio_scipy.astype(np.float32) / 2147483648.0
                    elif audio_scipy.dtype == np.uint8:
                        audio_stereo = (audio_scipy.astype(np.float32) - 128) / 128.0
                    else:
                        audio_stereo = audio_scipy.astype(np.float32)
                    
                    # 如果是立體聲，調整維度順序 (scipy 是 [samples, channels], librosa 是 [channels, samples])
                    if audio_stereo.ndim > 1 and audio_stereo.shape[1] == 2:
                        audio_stereo = audio_stereo.T  # 轉置
                
                print(f"   - 原始採樣率: {sr} Hz")
                print(f"   - 原始長度: {len(audio_stereo[0]) if audio_stereo.ndim > 1 else len(audio_stereo):.0f} 採樣點")
                
                # 處理立體聲/單聲道
                if audio_stereo.ndim == 1:
                    # 單聲道
                    audio_data = audio_stereo
                    selected_channel = "單聲道"
                    print(f"   - 類型: 單聲道")
                else:
                    # 立體聲，選擇左聲道（通常音樂製作時主要內容在左聲道或雙聲道平衡）
                    audio_left = audio_stereo[0]
                    audio_right = audio_stereo[1]
                    audio_mono = (audio_left + audio_right) / 2
                    
                    print(f"   - 類型: 立體聲")
                    print(f"   - 左聲道範圍: {audio_left.min():.4f} ~ {audio_left.max():.4f}")
                    print(f"   - 右聲道範圍: {audio_right.min():.4f} ~ {audio_right.max():.4f}")
                    print(f"   - 混合範圍: {audio_mono.min():.4f} ~ {audio_mono.max():.4f}")
                    
                    # 選擇混合單聲道作為主要信號（更平衡）
                    audio_data = audio_mono
                    selected_channel = "混合單聲道"
                
                # 重新採樣到目標頻率（與 SEEG 相同）
                if sr != target_fs:
                    print(f"   - 重新採樣: {sr} Hz -> {target_fs} Hz")
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_fs)
                
                print(f"   - 最終長度: {len(audio_data)} 採樣點")
                print(f"   - 最終範圍: {audio_data.min():.4f} ~ {audio_data.max():.4f}")
                print(f"   - 持續時間: {len(audio_data) / target_fs:.1f} 秒")
                
                # 根據對應的 SEEG 數據長度截取音樂數據
                if self.seeg_data:
                    # 獲取第一個病人對應歌曲的 SEEG 長度作為參考
                    first_patient = list(self.seeg_data.keys())[0]
                    seeg_length = self.seeg_data[first_patient][i]['data'].shape[1]
                    
                    print(f"   - 對應 SEEG 長度: {seeg_length} 採樣點 ({seeg_length/target_fs:.1f} 秒)")
                    
                    if len(audio_data) > seeg_length:
                        # 截取音樂數據到 SEEG 的長度
                        audio_data = audio_data[:seeg_length]
                        print(f"   - 截取後音樂長度: {len(audio_data)} 採樣點 ({len(audio_data)/target_fs:.1f} 秒)")
                    elif len(audio_data) < seeg_length:
                        # 如果音樂比 SEEG 短，進行零填充
                        padding = seeg_length - len(audio_data)
                        audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
                        print(f"   - 零填充後音樂長度: {len(audio_data)} 採樣點 ({len(audio_data)/target_fs:.1f} 秒)")
                
                
                # 計算多種特徵用於可視化
                # 1. 原始波形（適度下採樣用於顯示，保持足夠細節）
                downsample_factor = max(1, len(audio_data) // 500000)  # 增加顯示點數到50萬
                audio_display = audio_data[::downsample_factor]
                
                # 2. 包絡線（使用 RMS 方法，更平滑）
                try:
                    hop_length = target_fs // 50  # 增加到每秒 50 個點的包絡線
                    print(f"   - 計算包絡線，hop_length: {hop_length}")
                    
                    # 確保 hop_length 不會太小
                    hop_length = max(hop_length, 32)
                    frame_length = hop_length * 2
                    
                    rms_envelope = librosa.feature.rms(y=audio_data, hop_length=hop_length, frame_length=frame_length)[0]
                    print(f"   - 包絡線點數: {len(rms_envelope)}")
                    
                    # 調整包絡線時間軸以匹配音頻
                    envelope_time_points = len(rms_envelope)
                    if envelope_time_points != len(audio_data):
                        from scipy.interpolate import interp1d
                        envelope_time_orig = np.linspace(0, 1, envelope_time_points)
                        envelope_time_new = np.linspace(0, 1, len(audio_data))
                        interp_func = interp1d(envelope_time_orig, rms_envelope, kind='linear', bounds_error=False, fill_value='extrapolate')
                        rms_envelope = interp_func(envelope_time_new)
                        print(f"   - 包絡線插值後點數: {len(rms_envelope)}")
                    
                    # 3. 平滑包絡線（用於更好的視覺效果）
                    from scipy.ndimage import gaussian_filter1d
                    sigma = max(1, target_fs // 50)  # 確保 sigma 不會太小
                    smooth_envelope = gaussian_filter1d(rms_envelope, sigma=sigma)
                    print(f"   - 平滑包絡線計算完成，sigma: {sigma}")
                    
                except Exception as envelope_error:
                    print(f"   ❌ 包絡線計算失敗: {envelope_error}")
                    print(f"   🔄 使用最簡化包絡線代替")
                    
                    # 使用最簡單的方法：直接對音頻數據取絕對值並平滑
                    rms_envelope = np.abs(audio_data)
                    
                    # 簡單的移動平均平滑
                    window_size = min(1024, len(audio_data) // 100)  # 使用較小的窗口
                    if window_size > 1:
                        # 使用卷積實現移動平均
                        kernel = np.ones(window_size) / window_size
                        smooth_envelope = np.convolve(rms_envelope, kernel, mode='same')
                    else:
                        smooth_envelope = rms_envelope.copy()
                    
                    print(f"   ✅ 最簡化包絡線計算完成，窗口大小: {window_size}")
                
                self.music_data[i] = {
                    'name': song_names[i],
                    'audio': audio_data.astype(np.float32),           # 完整音頻數據，確保數據類型
                    'audio_display': audio_display.astype(np.float32), # 下採樣顯示數據
                    'envelope': rms_envelope.astype(np.float32),      # RMS 包絡線
                    'smooth_envelope': smooth_envelope.astype(np.float32), # 平滑包絡線
                    'selected_channel': selected_channel,
                    'fs': target_fs,  # 使用與 SEEG 相同的採樣頻率
                    'wav_file': wav_files[i],
                    'downsample_factor': downsample_factor
                }
                
                print(f"   ✅ 載入完成")
                
            except Exception as e:
                print(f"❌ 載入失敗 {wav_files[i]}: {e}")
                self.music_data[i] = {
                    'name': song_names[i],
                    'audio': None,
                    'envelope': None,
                    'selected_channel': "載入失敗",
                    'fs': target_fs,
                    'wav_file': wav_files[i]
                }
    
    def load_seeg_data(self):
        """載入所有病人的 SEEG 數據"""
        seeg_files = list(Path("data").glob("seeg_s*_5mm_110824.mat"))
        
        for seeg_file in seeg_files:
            patient_id = seeg_file.stem.split('_')[1]
            
            try:
                mat_data = scipy.io.loadmat(seeg_file)
                epoch_all = mat_data['epoch_all']
                patient_data = {}
                seeg_fs = mat_data['fs'][0, 0]
                
                print(f"🧠 載入 SEEG 數據: {seeg_file.name}")
                print(f"   - 病人 ID: {patient_id}")
                print(f"   - 採樣頻率: {seeg_fs} Hz")
                
                for song_idx in range(3):
                    epoch_data = epoch_all[0, song_idx]
                    patient_data[song_idx] = {
                        'data': epoch_data,
                        'channels': mat_data['ch_select'].flatten(),
                        'fs': seeg_fs
                    }
                    print(f"   - 歌曲 {song_idx+1}: {epoch_data.shape[0]} 通道, {epoch_data.shape[1]} 時間點, {epoch_data.shape[1]/seeg_fs:.1f} 秒")
                
                self.seeg_data[patient_id] = patient_data
                
            except Exception as e:
                print(f"❌ 載入 SEEG 數據失敗 {seeg_file}: {e}")
    
    def init_ui(self):
        """初始化使用者介面"""
        self.setWindowTitle('🧠🎵 SEEG Music Analysis Visualizer')
        self.setGeometry(100, 100, 1400, 900)
        
        # 主要 widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 音樂播放器
        music_player = self.create_music_player_panel()
        layout.addWidget(music_player)
        
        # 繪圖區域
        self.create_plot_area(layout)
        
        # 時間範圍控制
        time_control = self.create_time_control_panel()
        layout.addWidget(time_control)
    
    def create_control_panel(self):
        """創建控制面板"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # 歌曲選擇
        layout.addWidget(QLabel("歌曲:"))
        self.song_combo = QComboBox()
        self.song_combo.addItems([data['name'] for data in self.music_data.values()])
        self.song_combo.currentIndexChanged.connect(self.on_song_changed)
        layout.addWidget(self.song_combo)
        
        # 病人選擇
        layout.addWidget(QLabel("病人:"))
        self.patient_combo = QComboBox()
        patient_list = list(self.seeg_data.keys())
        self.patient_combo.addItems(patient_list)
        if patient_list:
            self.current_patient = patient_list[0]
        self.patient_combo.currentTextChanged.connect(self.on_patient_changed)
        layout.addWidget(self.patient_combo)
        
        # 通道選擇
        layout.addWidget(QLabel("通道:"))
        self.channel_combo = QComboBox()
        self.update_channel_combo()
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        layout.addWidget(self.channel_combo)
        
        # 重置視圖按鈕
        reset_btn = QPushButton("重置視圖")
        reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(reset_btn)
        
        # 音樂顯示模式選擇
        layout.addWidget(QLabel("音樂顯示:"))
        self.music_display_combo = QComboBox()
        self.music_display_combo.addItems(["波形", "包絡線", "平滑包絡線"])
        self.music_display_combo.setCurrentIndex(2)  # 預設使用平滑包絡線
        self.music_display_combo.currentIndexChanged.connect(self.on_display_mode_changed)
        layout.addWidget(self.music_display_combo)
        
        # 聲道信息顯示
        self.channel_info_label = QLabel("聲道: --")
        self.channel_info_label.setToolTip("顯示當前音樂使用的聲道信息")
        self.channel_info_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.channel_info_label)
        
        layout.addStretch()
        return panel
    
    def create_music_player_panel(self):
        """創建音樂播放器面板"""
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #d0d0d0;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
        """)
        layout = QHBoxLayout(panel)
        
        # 播放器標題
        title_label = QLabel("🎵 音樂播放器")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        layout.addWidget(QLabel("|"))  # 分隔符
        
        # 當前歌曲顯示
        layout.addWidget(QLabel("當前歌曲:"))
        self.current_song_label = QLabel("--")
        self.current_song_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        layout.addWidget(self.current_song_label)
        
        layout.addWidget(QLabel("|"))  # 分隔符
        
        # 播放控制按鈕
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setFixedWidth(80)
        layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ 暫停")
        self.pause_btn.clicked.connect(self.pause_playback)
        self.pause_btn.setFixedWidth(80)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setFixedWidth(80)
        layout.addWidget(self.stop_btn)
        
        layout.addWidget(QLabel("|"))  # 分隔符
        
        # 時間顯示
        layout.addWidget(QLabel("時間:"))
        self.time_display = QLabel("00:00 / 00:00")
        self.time_display.setStyleSheet("font-family: 'Courier New', 'Monaco', monospace; font-size: 16px; color: #2980b9; font-weight: bold;")
        self.time_display.setFixedWidth(120)
        layout.addWidget(self.time_display)
        
        layout.addWidget(QLabel("|"))  # 分隔符
        
        # 進度條
        layout.addWidget(QLabel("進度:"))
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setValue(0)
        self.progress_slider.setFixedWidth(200)
        self.progress_slider.sliderPressed.connect(self.on_progress_pressed)
        self.progress_slider.sliderReleased.connect(self.on_progress_released)
        self.progress_slider.valueChanged.connect(self.on_progress_changed)
        self.progress_dragging = False
        layout.addWidget(self.progress_slider)
        
        layout.addWidget(QLabel("|"))  # 分隔符
        
        # 音量控制
        layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("50%")
        self.volume_label.setFixedWidth(40)
        layout.addWidget(self.volume_label)
        
        layout.addStretch()
        return panel
    
    def create_plot_area(self, parent_layout):
        """創建繪圖區域"""
        # 設置 pyqtgraph 的全局選項
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        # 音樂信號圖
        self.envelope_plot = PlotWidget(title="🎵 音樂信號")
        self.envelope_plot.setLabel('left', '振幅')
        self.envelope_plot.setLabel('bottom', '時間 (秒)')
        self.envelope_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # 啟用滑鼠追蹤和十字線
        self.envelope_plot.scene().sigMouseMoved.connect(self.on_mouse_moved_envelope)
        self.envelope_plot.setMouseTracking(True)
        
        # 創建滑鼠追蹤垂直線
        self.mouse_vline_envelope = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='orange', width=1, style=Qt.PenStyle.DotLine))
        self.mouse_vline_envelope.setZValue(10)  # 設置較高的 Z 值確保在最上層
        self.envelope_plot.addItem(self.mouse_vline_envelope)
        
        parent_layout.addWidget(self.envelope_plot)
        
        # SEEG 腦波圖
        self.seeg_plot = PlotWidget(title="🧠 SEEG 腦波信號")
        self.seeg_plot.setLabel('left', '電壓 (μV)')
        self.seeg_plot.setLabel('bottom', '時間 (秒)')
        self.seeg_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # 啟用滑鼠追蹤
        self.seeg_plot.scene().sigMouseMoved.connect(self.on_mouse_moved_seeg)
        self.seeg_plot.setMouseTracking(True)
        
        # 創建滑鼠追蹤垂直線
        self.mouse_vline_seeg = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='orange', width=1, style=Qt.PenStyle.DotLine))
        self.mouse_vline_seeg.setZValue(10)  # 設置較高的 Z 值確保在最上層
        self.seeg_plot.addItem(self.mouse_vline_seeg)
        
        parent_layout.addWidget(self.seeg_plot)
        
        # 連結兩個圖的 X 軸，實現同步縮放和拖移
        self.seeg_plot.setXLink(self.envelope_plot)
        
        # 監聽視圖範圍變更，同步更新控制項
        self.envelope_plot.sigRangeChanged.connect(self.on_view_range_changed)
    
    def create_time_control_panel(self):
        """創建時間範圍控制面板"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        layout.addWidget(QLabel("時間範圍控制:"))
        
        # 開始時間
        layout.addWidget(QLabel("開始:"))
        self.start_time_spin = QSpinBox()
        self.start_time_spin.setSuffix(" 秒")
        self.start_time_spin.setMinimum(0)
        self.start_time_spin.valueChanged.connect(self.on_time_range_changed)
        layout.addWidget(self.start_time_spin)
        
        # 結束時間
        layout.addWidget(QLabel("結束:"))
        self.end_time_spin = QSpinBox()
        self.end_time_spin.setSuffix(" 秒")
        self.end_time_spin.setMinimum(1)
        self.end_time_spin.valueChanged.connect(self.on_time_range_changed)
        layout.addWidget(self.end_time_spin)
        
        # 時間窗口大小
        layout.addWidget(QLabel("視窗大小:"))
        self.window_size_spin = QDoubleSpinBox()
        self.window_size_spin.setSuffix(" 秒")
        self.window_size_spin.setMinimum(0.1)
        self.window_size_spin.setMaximum(60.0)
        self.window_size_spin.setValue(30.0)
        self.window_size_spin.setSingleStep(0.1)
        self.window_size_spin.setDecimals(1)
        self.window_size_spin.valueChanged.connect(self.on_window_size_changed)
        layout.addWidget(self.window_size_spin)
        
        # 滑鼠位置資訊顯示
        layout.addWidget(QLabel("滑鼠位置:"))
        self.mouse_info_label = QLabel("時間: -- 秒")
        self.mouse_info_label.setFixedWidth(120)
        layout.addWidget(self.mouse_info_label)
        
        self.envelope_value_label = QLabel("音樂: --")
        self.envelope_value_label.setFixedWidth(100)
        layout.addWidget(self.envelope_value_label)
        
        self.seeg_value_label = QLabel("SEEG: --")
        self.seeg_value_label.setFixedWidth(100)
        layout.addWidget(self.seeg_value_label)
        
        layout.addStretch()
        return panel
    
    def update_channel_combo(self):
        """更新通道選擇下拉選單"""
        self.channel_combo.clear()
        if self.current_patient and self.current_patient in self.seeg_data:
            patient_data = self.seeg_data[self.current_patient][self.current_song]
            n_channels = patient_data['data'].shape[0]
            channels = patient_data['channels']
            
            for i in range(n_channels):
                if i < len(channels):
                    self.channel_combo.addItem(f"通道 {i+1} (#{int(channels[i])})")
                else:
                    self.channel_combo.addItem(f"通道 {i+1}")
    
    def on_song_changed(self, index):
        """歌曲改變事件"""
        self.current_song = index
        self.update_channel_combo()
        self.update_time_controls()
        self.update_channel_info()  # 更新聲道信息
        # 停止當前播放並清除媒體源
        self.stop_playback()
        self.media_player.setSource(QUrl())  # 清除媒體源
        # 更新當前歌曲顯示
        if index in self.music_data:
            self.current_song_label.setText(self.music_data[index]['name'])
        self.update_plots()
    
    def on_patient_changed(self, patient_id):
        """病人改變事件"""
        self.current_patient = patient_id
        self.update_channel_combo()
        self.update_plots()
    
    def on_channel_changed(self, index):
        """通道改變事件"""
        self.current_channel = index
        self.update_plots()
    
    def on_display_mode_changed(self, index):
        """音樂顯示模式改變事件"""
        self.update_channel_info()
        self.update_plots()
    
    def update_channel_info(self):
        """更新聲道信息顯示"""
        if self.current_song in self.music_data:
            music_info = self.music_data[self.current_song]
            channel_info = music_info.get('selected_channel', '未知')
            display_modes = ["波形", "包絡線", "平滑包絡線"]
            current_mode = display_modes[self.music_display_combo.currentIndex()]
            self.channel_info_label.setText(f"聲道: {channel_info} | 顯示: {current_mode}")
        else:
            self.channel_info_label.setText("聲道: --")
    
    def on_mouse_moved_envelope(self, pos):
        """處理包絡線圖的滑鼠移動"""
        if self.envelope_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.envelope_plot.plotItem.vb.mapSceneToView(pos)
            x_pos = mouse_point.x()
            y_pos = mouse_point.y()
            
            # 更新滑鼠追蹤垂直線位置
            self.mouse_vline_envelope.setPos(x_pos)
            self.mouse_vline_seeg.setPos(x_pos)  # 同步更新 SEEG 圖的垂直線
            
            # 獲取對應時間點的音樂和 SEEG 數值
            music_val, seeg_val = self.get_values_at_time(x_pos)
            
            self.mouse_info_label.setText(f"時間: {x_pos:.2f} 秒")
            
            # 安全格式化數值
            if isinstance(music_val, str):
                self.envelope_value_label.setText(f"音樂: {music_val}")
            else:
                self.envelope_value_label.setText(f"音樂: {music_val:.4f}")
            
            if isinstance(seeg_val, str):
                self.seeg_value_label.setText(f"SEEG: {seeg_val}")
            else:
                self.seeg_value_label.setText(f"SEEG: {seeg_val:.2f} μV")
    
    def on_mouse_moved_seeg(self, pos):
        """處理SEEG圖的滑鼠移動"""
        if self.seeg_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.seeg_plot.plotItem.vb.mapSceneToView(pos)
            x_pos = mouse_point.x()
            y_pos = mouse_point.y()
            
            # 更新滑鼠追蹤垂直線位置
            self.mouse_vline_envelope.setPos(x_pos)  # 同步更新包絡線圖的垂直線
            self.mouse_vline_seeg.setPos(x_pos)
            
            # 獲取對應時間點的音樂和 SEEG 數值
            music_val, seeg_val = self.get_values_at_time(x_pos)
            
            self.mouse_info_label.setText(f"時間: {x_pos:.2f} 秒")
            
            # 安全格式化數值
            if isinstance(music_val, str):
                self.envelope_value_label.setText(f"音樂: {music_val}")
            else:
                self.envelope_value_label.setText(f"音樂: {music_val:.4f}")
            
            if isinstance(seeg_val, str):
                self.seeg_value_label.setText(f"SEEG: {seeg_val}")
            else:
                self.seeg_value_label.setText(f"SEEG: {seeg_val:.2f} μV")
    
    def get_values_at_time(self, time_pos):
        """獲取指定時間點的音樂和SEEG數值"""
        music_val = "--"
        seeg_val = "--"
        
        try:
            # 獲取音樂數值
            audio, display_data = self.get_current_music_data()
            if display_data is not None:
                display_mode = self.music_display_combo.currentIndex()
                
                if display_mode == 0:  # 波形模式，需要考慮下採樣因子
                    music_info = self.music_data[self.current_song]
                    downsample_factor = music_info.get('downsample_factor', 1)
                    sample_idx = int(time_pos * self.fs // downsample_factor)
                else:  # 包絡線模式
                    sample_idx = int(time_pos * self.fs)
                
                if 0 <= sample_idx < len(display_data):
                    music_val = display_data[sample_idx]
            
            # 獲取SEEG數值
            if (self.current_patient and self.current_patient in self.seeg_data and 
                self.current_song in self.seeg_data[self.current_patient]):
                
                patient_data = self.seeg_data[self.current_patient][self.current_song]
                seeg_data = patient_data['data']
                seeg_fs = patient_data['fs']  # 使用該病人/歌曲的特定採樣頻率
                
                if self.current_channel < seeg_data.shape[0]:
                    channel_data = seeg_data[self.current_channel, :, 0]
                    sample_idx = int(time_pos * seeg_fs)  # 使用正確的採樣頻率
                    if 0 <= sample_idx < len(channel_data):
                        seeg_val = channel_data[sample_idx]
        
        except Exception as e:
            pass  # 如果出錯就保持默認值
        
        return music_val, seeg_val
        
    def get_current_music_data(self):
        """獲取當前選擇的音樂數據和顯示模式"""
        if self.current_song not in self.music_data:
            return None, None
        
        music_info = self.music_data[self.current_song]
        if music_info['audio'] is None:
            return None, None
        
        # 根據顯示模式選擇數據
        display_mode = self.music_display_combo.currentIndex()
        
        if display_mode == 0:  # 波形
            # 波形模式使用下採樣的顯示數據
            return music_info['audio'], music_info['audio_display']
        elif display_mode == 1:  # 包絡線
            return music_info['audio'], music_info['envelope']
        else:  # 平滑包絡線
            return music_info['audio'], music_info['smooth_envelope']
    
    def on_view_range_changed(self):
        """視圖範圍改變時同步控制項"""
        # 使用 QTimer 來延遲執行，避免在縮放過程中頻繁更新
        if not hasattr(self, '_sync_timer'):
            from PyQt6.QtCore import QTimer
            self._sync_timer = QTimer()
            self._sync_timer.setSingleShot(True)
            self._sync_timer.timeout.connect(self.sync_view_controls)
        
        self._sync_timer.start(100)  # 100ms 延遲
    
    def toggle_playback(self):
        """切換播放/暫停"""
        if not self.is_playing:
            self.start_playback()
        else:
            self.pause_playback()
    
    def start_playback(self):
        """開始播放"""
        if self.current_song in self.music_data:
            # 如果媒體播放器已經有源，且不是停止狀態，直接播放
            if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                # 從暫停狀態恢復播放
                self.media_player.play()
            else:
                # 載入對應的WAV檔案
                wav_files = {
                    0: "BrahmsPianoConcerto.wav",
                    1: "LostStars.wav", 
                    2: "Doraemon.wav"
                }
                
                wav_file = Path("data") / wav_files[self.current_song]
                if wav_file.exists():
                    self.media_player.setSource(QUrl.fromLocalFile(str(wav_file.absolute())))
                    self.media_player.play()
            
            self.is_playing = True
            self.play_btn.setText("▶ 播放")
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.play_timer.start(50)  # 每50ms更新一次位置
            
            # 更新當前歌曲顯示
            self.current_song_label.setText(self.music_data[self.current_song]['name'])
    
    def pause_playback(self):
        """暫停播放"""
        self.media_player.pause()
        self.is_playing = False
        self.play_btn.setText("▶ 繼續")
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.play_timer.stop()
    
    def stop_playback(self):
        """停止播放"""
        self.media_player.stop()
        self.is_playing = False
        self.play_btn.setText("▶ 播放")
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.play_timer.stop()
        self.progress_slider.setValue(0)
        self.time_display.setText("00:00 / 00:00")
        if self.play_line:
            self.play_line.setPos(0)
            self.seeg_play_line.setPos(0)
    
    def set_volume(self, value):
        """設置音量"""
        self.audio_output.setVolume(value / 100.0)
        self.volume_label.setText(f"{value}%")
    
    def format_time(self, seconds):
        """格式化時間顯示"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def update_play_position(self):
        """更新播放位置線和時間顯示"""
        if self.media_player.duration() > 0:
            position_ms = self.media_player.position()
            duration_ms = self.media_player.duration()
            position_sec = position_ms / 1000.0
            duration_sec = duration_ms / 1000.0
            
            # 更新播放線位置
            if self.play_line and not self.dragging_play_line:
                self.play_line.setPos(position_sec)
                self.seeg_play_line.setPos(position_sec)
            
            # 更新時間顯示
            current_time = self.format_time(position_sec)
            total_time = self.format_time(duration_sec)
            self.time_display.setText(f"{current_time} / {total_time}")
            
            # 更新進度條
            if not self.progress_dragging:
                progress = int((position_ms / duration_ms) * 100)
                self.progress_slider.setValue(progress)
    
    def on_progress_pressed(self):
        """進度條按下"""
        self.progress_dragging = True
    
    def on_progress_released(self):
        """進度條釋放"""
        self.progress_dragging = False
        if self.media_player.duration() > 0:
            progress = self.progress_slider.value()
            new_position_ms = int((progress / 100.0) * self.media_player.duration())
            self.media_player.setPosition(new_position_ms)
            
            # 同步更新播放線位置
            position_sec = new_position_ms / 1000.0
            if self.play_line:
                self.play_line.setPos(position_sec)
                self.seeg_play_line.setPos(position_sec)
    
    def on_progress_changed(self, value):
        """進度條改變"""
        if self.progress_dragging and self.media_player.duration() > 0:
            position_sec = (value / 100.0) * (self.media_player.duration() / 1000.0)
            if self.play_line:
                self.play_line.setPos(position_sec)
                self.seeg_play_line.setPos(position_sec)
    
    def on_play_line_dragged(self, line):
        """處理播放線拖拽"""
        new_position = line.pos()[0]
        if self.media_player.duration() > 0:
            new_position_ms = int(new_position * 1000)
            # 確保位置在有效範圍內
            new_position_ms = max(0, min(new_position_ms, self.media_player.duration()))
            self.media_player.setPosition(new_position_ms)
            
            # 同步更新進度條
            progress = int((new_position_ms / self.media_player.duration()) * 100)
            self.progress_slider.setValue(progress)
    
    def on_play_line_drag_start(self):
        """開始拖拽播放線"""
        self.dragging_play_line = True
    
    def on_play_line_drag_end(self):
        """結束拖拽播放線"""
        self.dragging_play_line = False
    
    def on_time_range_changed(self):
        """時間範圍改變事件"""
        start_time = self.start_time_spin.value()
        end_time = self.end_time_spin.value()
        
        if start_time >= end_time:
            return
        
        # 設定視圖範圍
        self.envelope_plot.setXRange(start_time, end_time, padding=0)
        self.seeg_plot.setXRange(start_time, end_time, padding=0)
        
        # 更新視窗大小控制項
        window_size = end_time - start_time
        self.window_size_spin.setValue(window_size)
        
        # 禁用自動縮放以保持設定的範圍
        self.envelope_plot.enableAutoRange(axis='y', enable=False)
        self.seeg_plot.enableAutoRange(axis='y', enable=False)
    
    def on_window_size_changed(self):
        """視窗大小改變事件"""
        window_size = self.window_size_spin.value()
        
        # 獲取當前視圖的中心點
        current_range = self.envelope_plot.getViewBox().viewRange()[0]
        current_center = (current_range[0] + current_range[1]) / 2
        
        # 獲取數據的最大時間範圍
        max_time = 0
        if self.current_song in self.music_data:
            music_info = self.music_data[self.current_song]
            if music_info['audio'] is not None:
                max_time = len(music_info['audio']) / self.fs
        
        # 計算新的視圖範圍，確保不超出數據邊界
        new_start = max(0, current_center - window_size / 2)
        new_end = min(max_time, current_center + window_size / 2)
        
        # 如果調整後的範圍小於視窗大小，則調整中心點
        if new_end - new_start < window_size and max_time >= window_size:
            if new_start == 0:
                new_end = min(max_time, window_size)
            elif new_end == max_time:
                new_start = max(0, max_time - window_size)
        
        # 設定新的視圖範圍，禁用自動範圍
        self.envelope_plot.setXRange(new_start, new_end, padding=0)
        self.seeg_plot.setXRange(new_start, new_end, padding=0)
        
        # 禁用自動縮放，保持當前的 Y 軸範圍
        self.envelope_plot.enableAutoRange(axis='y', enable=False)
        self.seeg_plot.enableAutoRange(axis='y', enable=False)
    
    def update_time_controls(self):
        """更新時間控制範圍"""
        audio, envelope = self.get_current_music_data()
        if envelope is not None:
            max_time = len(envelope) / self.fs
            self.start_time_spin.setMaximum(int(max_time))
            self.end_time_spin.setMaximum(int(max_time))
            self.end_time_spin.setValue(min(60, int(max_time)))
            
            # 同時更新視窗大小控制的最大值
            self.window_size_spin.setMaximum(max_time)
    
    def sync_view_controls(self):
        """同步視圖控制項與當前視圖狀態"""
        current_range = self.envelope_plot.getViewBox().viewRange()[0]
        start_time = current_range[0]
        end_time = current_range[1]
        window_size = end_time - start_time
        
        # 暫時禁用信號以避免遞歸調用
        self.start_time_spin.blockSignals(True)
        self.end_time_spin.blockSignals(True)
        self.window_size_spin.blockSignals(True)
        
        # 更新控制項值
        self.start_time_spin.setValue(int(start_time))
        self.end_time_spin.setValue(int(end_time))
        self.window_size_spin.setValue(round(window_size, 1))
        
        # 重新啟用信號
        self.start_time_spin.blockSignals(False)
        self.end_time_spin.blockSignals(False)
        self.window_size_spin.blockSignals(False)
    
    def update_plots(self):
        """更新繪圖"""
        if not self.music_data or not self.current_patient:
            return
        
        # 清除舊的繪圖（但保留垂直線）
        self.envelope_plot.clear()
        self.seeg_plot.clear()
        
        # 初始化 max_time，優先使用音樂長度，否則使用 SEEG 長度
        max_time = 0
        
        # 繪製音樂信號
        audio, display_data = self.get_current_music_data()
        if display_data is not None:
            # 根據顯示模式調整時間軸
            display_mode = self.music_display_combo.currentIndex()
            
            if display_mode == 0:  # 波形模式，使用下採樣的數據
                music_info = self.music_data[self.current_song]
                downsample_factor = music_info.get('downsample_factor', 1)
                time_music = np.arange(len(display_data)) * downsample_factor / self.fs
                max_time = len(display_data) * downsample_factor / self.fs
            else:  # 包絡線模式，使用完整時間軸
                time_music = np.arange(len(display_data)) / self.fs
                max_time = len(display_data) / self.fs
            
            # 根據顯示模式選擇顏色和標籤
            display_names = ["原始波形", "包絡線", "平滑包絡線"]
            colors = ['darkblue', 'red', 'darkred']
            widths = [1, 2, 2]
            
            self.envelope_plot.plot(time_music, display_data, 
                                   pen=pg.mkPen(color=colors[display_mode], width=widths[display_mode]),
                                   name=f'🎵 音樂 - {display_names[display_mode]}')
            
            print(f"🔍 調試信息 - 音樂: 長度={len(display_data)}, 時間軸長度={len(time_music)}, 最大時間={time_music[-1]:.1f}秒")
        
        # 繪製 SEEG 數據
        if self.current_patient in self.seeg_data:
            patient_data = self.seeg_data[self.current_patient][self.current_song]
            seeg_data = patient_data['data']
            seeg_fs = patient_data['fs']  # 使用該病人/歌曲的特定採樣頻率
            
            if self.current_channel < seeg_data.shape[0]:
                # 取第一個條件的數據 (第三維度的第一個)
                channel_data = seeg_data[self.current_channel, :, 0]
                time_seeg = np.arange(len(channel_data)) / seeg_fs  # 使用正確的採樣頻率
                
                # 如果音樂數據無效，使用 SEEG 長度作為 max_time
                if max_time == 0:
                    max_time = len(channel_data) / seeg_fs  # 使用正確的採樣頻率
                
                self.seeg_plot.plot(time_seeg, channel_data,
                                   pen=pg.mkPen(color='darkblue', width=1),
                                   name=f'🧠 SEEG 通道 {self.current_channel+1}')
                                   
                print(f"🔍 調試信息 - SEEG: 長度={len(channel_data)}, 採樣頻率={seeg_fs}Hz, 時間軸長度={len(time_seeg)}, 最大時間={time_seeg[-1]:.1f}秒")
        
        # 添加播放位置線（只有當有有效的時間軸時）
        # 添加播放位置線（只有當有有效的時間軸時）
        if max_time > 0:
            if self.play_line is None:
                # 創建播放位置線（綠色）
                self.play_line = InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='green', width=3, style=Qt.PenStyle.DashLine))
                self.play_line.setMovable(True)
                self.play_line.setBounds([0, max_time])
                self.play_line.setZValue(5)  # 設置 Z 值使其在數據線之上，但在滑鼠線之下
                
                # 連接拖拽事件
                self.play_line.sigPositionChangeFinished.connect(self.on_play_line_dragged)
                self.play_line.sigDragged.connect(lambda: self.on_play_line_drag_start())
                
                self.envelope_plot.addItem(self.play_line)
                
                # 在SEEG圖中也添加相同的線
                self.seeg_play_line = InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='green', width=3, style=Qt.PenStyle.DashLine))
                self.seeg_play_line.setMovable(True)
                self.seeg_play_line.setBounds([0, max_time])
                self.seeg_play_line.setZValue(5)
                self.seeg_play_line.sigPositionChangeFinished.connect(self.on_play_line_dragged)
                self.seeg_plot.addItem(self.seeg_play_line)
                
                # 同步兩條線的位置
                self.play_line.sigPositionChanged.connect(lambda line: self.seeg_play_line.setPos(line.pos()))
                self.seeg_play_line.sigPositionChanged.connect(lambda line: self.play_line.setPos(line.pos()))
            else:
                # 更新現有線的邊界並重新添加到圖表
                self.play_line.setBounds([0, max_time])
                self.seeg_play_line.setBounds([0, max_time])
                self.play_line.setPos(0)
                self.seeg_play_line.setPos(0)
            
            # 重新添加播放線（因為 clear() 會移除它們）
            self.envelope_plot.addItem(self.play_line)
            self.seeg_plot.addItem(self.seeg_play_line)
        
        # 重新創建或重新添加滑鼠追蹤線
        if self.mouse_vline_envelope is None:
            self.mouse_vline_envelope = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='orange', width=1, style=Qt.PenStyle.DotLine))
            self.mouse_vline_envelope.setZValue(10)
        if self.mouse_vline_seeg is None:
            self.mouse_vline_seeg = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='orange', width=1, style=Qt.PenStyle.DotLine))
            self.mouse_vline_seeg.setZValue(10)
        
        # 重新添加滑鼠追蹤線（因為 clear() 會移除它們）
        self.envelope_plot.addItem(self.mouse_vline_envelope)
        self.seeg_plot.addItem(self.mouse_vline_seeg)
        
        # 設置初始視圖範圍
        self.reset_view()
    
    def reset_view(self):
        """重置視圖到全範圍"""
        if self.current_song in self.music_data:
            music_info = self.music_data[self.current_song]
            if music_info['audio'] is not None:
                max_time = len(music_info['audio']) / self.fs
                view_time = min(60, max_time)  # 預設顯示前60秒或全部
                
                # 設定 X 軸範圍
                self.envelope_plot.setXRange(0, view_time, padding=0.02)
                self.seeg_plot.setXRange(0, view_time, padding=0.02)
                
                # 重新啟用並執行 Y 軸自動調整
                self.envelope_plot.enableAutoRange(axis='y', enable=True)
                self.seeg_plot.enableAutoRange(axis='y', enable=True)
                
                # 手動同步控制項（不等待定時器）
                self.sync_view_controls()

def main():
    app = QApplication(sys.argv)
    
    # 檢查數據文件是否存在
    data_path = Path("data")
    if not data_path.exists():
        print("Error: data 資料夾不存在!")
        return
    
    # 檢查 WAV 音樂文件
    wav_files = ['BrahmsPianoConcerto.wav', 'LostStars.wav', 'Doraemon.wav']
    missing_wav = []
    for wav_file in wav_files:
        if not (data_path / wav_file).exists():
            missing_wav.append(wav_file)
    
    if missing_wav:
        print(f"Warning: 缺少音樂文件: {', '.join(missing_wav)}")
    
    # 檢查 SEEG 數據文件
    seeg_files = list(data_path.glob("seeg_s*_5mm_110824.mat"))
    if not seeg_files:
        print("Error: 找不到 SEEG 數據文件!")
        return
    
    # 創建並顯示視窗
    visualizer = MusicSEEGVisualizer()
    visualizer.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
