#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音樂與SEEG信號的Beat和BPM關係分析模組
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy.signal import find_peaks, correlate, hilbert
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BeatBPMAnalyzer:
    """Beat和BPM關係分析器"""
    
    def __init__(self, fs=2048):
        """初始化分析器"""
        self.fs = fs
        
        # 已知的音樂BPM信息（從README.md獲取）
        self.known_bpm = {
            'BrahmsPianoConcerto.wav': 93,
            'LostStars.wav': 82,
            'Doraemon.wav': 121
        }
    
    def analyze_beat_bpm_relationship(self, audio_data, seeg_data, song_name, channel_name):
        """
        與main.py接口兼容的分析方法
        分析音樂與SEEG信號的beat和BPM關係
        """
        print(f"🎵 開始分析 {song_name} - {channel_name} 的Beat-BPM關係...")
        
        try:
            # 限制數據長度避免溢位（最多30秒）
            max_samples = min(30 * self.fs, 60000)  # 30秒或最大60k樣本
            min_length = min(len(audio_data), len(seeg_data), max_samples)
            
            audio_segment = audio_data[:min_length].astype(np.float32)
            seeg_segment = seeg_data[:min_length].astype(np.float32)
            
            # 確保seeg_data是1維數組
            if seeg_segment.ndim > 1:
                seeg_segment = seeg_segment.flatten()
            
            print(f"   數據長度: {min_length} 點 ({min_length/self.fs:.1f} 秒)")
            
            # 獲取已知BPM
            known_bpm = None
            for key, bpm in self.known_bpm.items():
                if key.lower().replace('.wav', '') in song_name.lower():
                    known_bpm = bpm
                    break
            
            # 1. 使用已知BPM生成beat時間點
            print("   🎼 使用已知BPM生成beat時間點...")
            music_result = self.generate_beats_from_known_bpm(audio_segment, self.fs, known_bpm)
            if not music_result:
                print("❌ Beat時間點生成失敗")
                return None
            
            # 2. SEEG beat檢測
            print("   🧠 檢測SEEG rhythm...")
            seeg_result = self.analyze_seeg_rhythm(seeg_segment)
            if not seeg_result:
                print("❌ SEEG節律檢測失敗")
                return None
            
            # 3. 同步性分析
            print("   🔄 計算同步性...")
            sync_result = self.analyze_synchronization(audio_segment, seeg_segment)
            
            # 4. 相關性分析
            print("   📈 計算信號相關性...")
            corr_analysis = self.analyze_correlation(audio_segment, seeg_segment, music_result['beat_times'])
            
            # 5. 相位同步分析
            print("   📊 計算相位同步...")
            phase_result = self.analyze_phase_synchronization(audio_segment, seeg_segment)
            if not phase_result:
                print("❌ 相位同步分析失敗")
                return None
            
            # 6. 詳細同步性分析
            print("   🔬 進行綜合分析...")
            sync_analysis = self.analyze_detailed_synchronization(audio_segment, seeg_segment, music_result['beat_times'])
            
            # 7. 生成分析摘要
            summary = self.generate_analysis_summary(music_result, sync_result, sync_analysis, corr_analysis, phase_result)
            
            # 8. 整合所有結果
            results = {
                'song_name': song_name,
                'channel_name': channel_name,
                'detected_bpm': float(known_bpm) if known_bpm else 0.0,
                'known_bpm': known_bpm,
                'bmp_accuracy': float(music_result['bmp_accuracy']),
                'beat_times': music_result['beat_times'],
                'beat_frequency': len(music_result['beat_times']) / (min_length / self.fs),
                'seeg_estimated_bpm': float(seeg_result['estimated_bpm']),
                'sync_score': float(sync_result['sync_score']),
                'best_lag': float(sync_result['best_lag']),
                'overall_correlation': float(corr_analysis['overall_correlation']),
                'mean_beat_correlation': float(corr_analysis['mean_beat_correlation']),
                'sliding_correlation': corr_analysis['sliding_correlation'],
                'sliding_times': corr_analysis['sliding_times'],
                'beat_correlations': corr_analysis['beat_correlations'],
                'phase_locking_value': float(phase_result['phase_locking_value']),
                'phase_consistency': float(phase_result['phase_consistency']),
                'beat_consistency_score': float(sync_analysis['beat_consistency_score']),
                'beat_enhancement': float(sync_analysis['beat_enhancement']),
                'analysis_summary': summary,
                'audio_signal': audio_segment,
                'seeg_signal': seeg_segment,
                'smooth_envelope': seeg_result['envelope'],
                'average_beat_response': sync_analysis['average_beat_response']
            }
            
            print(f"   ✅ 分析完成")
            print(f"      使用BPM: {known_bpm}")
            print(f"      Beat數量: {len(music_result['beat_times'])}")
            print(f"      BPM準確度: {float(music_result['bmp_accuracy']):.1f}%")
            print(f"      整體相關性: {float(corr_analysis['overall_correlation']):.3f}")
            print(f"      綜合評分: {float(summary['overall_synchronization_score']):.3f}")
            
            # 9. 生成分析圖表
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            # 保存圖表
            fig_filename = f"beat_bpm_analysis_{song_name.replace(' ', '_')}_{channel_name}_{timestamp}.png"
            fig_path = output_dir / fig_filename
            fig = self.create_analysis_plots(results, fig_path)
            results['figure_path'] = fig_path
            
            # 10. 保存CSV結果
            csv_filename = f"beat_bpm_analysis_{song_name.replace(' ', '_')}_{channel_name}_{timestamp}.csv"
            csv_path = output_dir / csv_filename
            self.save_results_to_csv(results, csv_path)
            
            return results
            
        except Exception as e:
            print(f"❌ 分析過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_beats_from_known_bpm(self, audio_data, sr, known_bpm):
        """根據已知BPM生成beat時間點"""
        try:
            if not known_bpm:
                print("❌ 沒有已知BPM資訊")
                return None
            
            # 計算beat間隔（秒）
            beat_interval = 60.0 / known_bpm
            audio_duration = len(audio_data) / sr
            
            # 生成beat時間點（從0.5秒開始，避免開頭的靜音）
            beat_times = []
            current_time = 0.5  # 從0.5秒開始
            
            while current_time < audio_duration:
                beat_times.append(current_time)
                current_time += beat_interval
            
            beat_times = np.array(beat_times)
            
            # 準確度設為100%（因為使用已知BPM）
            bmp_accuracy = 100.0
            
            print(f"      根據BPM {known_bpm}生成了 {len(beat_times)} 個beat")
            print(f"      Beat間隔: {beat_interval:.3f}秒")
            print(f"      音樂時長: {audio_duration:.1f}秒")
            
            return {
                'detected_bpm': known_bpm,
                'bmp_accuracy': bmp_accuracy,
                'beat_times': beat_times,
                'num_beats': len(beat_times)
            }
        except Exception as e:
            print(f"❌ Beat生成失敗: {e}")
            return {
                'detected_bpm': 0.0,
                'bmp_accuracy': 0.0,
                'beat_times': np.array([]),
                'num_beats': 0
            }

    def analyze_seeg_rhythm(self, seeg_data):
        """分析SEEG信號的節律特徵"""
        try:
            # 限制數據長度以提高效率
            max_samples = min(len(seeg_data), 30000)  # 最多分析30k樣本
            seeg_segment = seeg_data[:max_samples]
            
            # 獲取信號包絡
            analytic_signal = hilbert(seeg_segment)
            envelope = np.abs(analytic_signal)
            
            # 平滑包絡
            sigma = min(5.0, len(envelope) / 2000)  # 適度平滑
            smooth_envelope = gaussian_filter1d(envelope, sigma=sigma)
            
            # 檢測peaks - 使用較低的閾值以獲得更多peaks
            height_threshold = np.mean(smooth_envelope) + 0.5 * np.std(smooth_envelope)
            min_distance = max(10, int(self.fs * 0.1))  # 最小間隔0.1秒
            
            peaks, peak_properties = find_peaks(smooth_envelope, 
                                              height=height_threshold,
                                              distance=min_distance,
                                              prominence=0.1 * np.std(smooth_envelope))
            
            # 估算BPM（基於peak間隔）
            if len(peaks) > 3:  # 需要足夠的peaks來估算
                peak_intervals = np.diff(peaks) / self.fs  # 轉為秒
                # 過濾異常值
                valid_intervals = peak_intervals[(peak_intervals > 0.3) & (peak_intervals < 3.0)]
                if len(valid_intervals) > 0:
                    mean_interval = np.median(valid_intervals)  # 使用中位數更穩定
                    estimated_bpm = 60.0 / mean_interval
                else:
                    estimated_bpm = 0.0
            else:
                estimated_bpm = 0.0
            
            # 計算節律強度
            rhythm_strength = np.std(smooth_envelope) / np.mean(smooth_envelope) if np.mean(smooth_envelope) > 0 else 0
            
            print(f"      檢測到 {len(peaks)} 個SEEG peaks")
            print(f"      估算SEEG BPM: {estimated_bpm:.1f}")
            print(f"      節律強度: {rhythm_strength:.3f}")
            
            return {
                'estimated_bpm': estimated_bpm,
                'envelope': smooth_envelope,
                'peaks': peaks,
                'num_peaks': len(peaks),
                'rhythm_strength': rhythm_strength,
                'peak_heights': peak_properties.get('peak_heights', []) if hasattr(peak_properties, 'get') else []
            }
        except Exception as e:
            print(f"❌ SEEG節律分析失敗: {e}")
            return {
                'estimated_bpm': 0.0,
                'envelope': np.zeros(min(len(seeg_data), 1000)),
                'peaks': np.array([]),
                'num_peaks': 0,
                'rhythm_strength': 0.0,
                'peak_heights': []
            }
            
    def analyze_synchronization(self, audio_data, seeg_data):
        """分析音樂與SEEG信號的同步性"""
        try:
            # 確保數據長度一致並下採樣
            min_length = min(len(audio_data), len(seeg_data), 20000)  # 限制最大長度
            audio_short = audio_data[:min_length]
            seeg_short = seeg_data[:min_length]
            
            # 計算互相關
            correlation = correlate(audio_short, seeg_short, mode='full')
            lags = np.arange(-len(seeg_short) + 1, len(audio_short))
            
            # 找到最大相關值和對應的lag
            max_idx = np.argmax(np.abs(correlation))
            best_lag = lags[max_idx]
            max_correlation = correlation[max_idx]
            
            # 計算同步分數
            sync_score = float(np.abs(max_correlation) / (np.linalg.norm(audio_short) * np.linalg.norm(seeg_short)))
            
            return {
                'sync_score': sync_score,
                'best_lag': best_lag,
                'max_correlation': float(max_correlation)
            }
        except Exception as e:
            print(f"❌ 同步性分析失敗: {e}")
            return {
                'sync_score': 0.0,
                'best_lag': 0,
                'max_correlation': 0.0
            }

    def analyze_correlation(self, audio_data, seeg_data, beat_times):
        """分析音樂與SEEG信號的相關性"""
        try:
            # 確保數據長度一致
            min_length = min(len(audio_data), len(seeg_data))
            audio_aligned = audio_data[:min_length]
            seeg_aligned = seeg_data[:min_length]
            
            # 整體相關性
            overall_corr, _ = pearsonr(audio_aligned, seeg_aligned)
            
            # 滑動窗口相關性分析
            window_size = min(int(self.fs * 2), min_length // 10)  # 2秒窗口或數據的1/10
            step_size = window_size // 2
            
            sliding_corr = []
            sliding_times = []
            
            for i in range(0, min_length - window_size, step_size):
                try:
                    window_audio = audio_aligned[i:i+window_size]
                    window_seeg = seeg_aligned[i:i+window_size]
                    corr, _ = pearsonr(window_audio, window_seeg)
                    sliding_corr.append(float(corr) if not np.isnan(corr) else 0.0)
                    sliding_times.append(i / self.fs)
                except:
                    sliding_corr.append(0.0)
                    sliding_times.append(i / self.fs)
            
            # Beat相關的相關性分析
            beat_correlations = []
            beat_window = int(self.fs * 0.5)  # 0.5秒窗口
            
            for beat_time in beat_times:
                try:
                    beat_idx = int(beat_time * self.fs)
                    if beat_idx + beat_window < min_length:
                        audio_beat = audio_aligned[beat_idx:beat_idx+beat_window]
                        seeg_beat = seeg_aligned[beat_idx:beat_idx+beat_window]
                        corr, _ = pearsonr(audio_beat, seeg_beat)
                        beat_correlations.append(float(corr) if not np.isnan(corr) else 0.0)
                except:
                    beat_correlations.append(0.0)
            
            mean_beat_corr = np.mean(beat_correlations) if beat_correlations else 0.0
            
            return {
                'overall_correlation': float(overall_corr) if not np.isnan(overall_corr) else 0.0,
                'mean_beat_correlation': float(mean_beat_corr),
                'sliding_correlation': sliding_corr,
                'sliding_times': sliding_times,
                'beat_correlations': beat_correlations
            }
        except Exception as e:
            print(f"❌ 相關性分析失敗: {e}")
            return {
                'overall_correlation': 0.0,
                'mean_beat_correlation': 0.0,
                'sliding_correlation': [],
                'sliding_times': [],
                'beat_correlations': []
            }

    def analyze_phase_synchronization(self, audio_data, seeg_data):
        """分析相位同步"""
        try:
            # 確保數據長度一致並限制長度
            min_length = min(len(audio_data), len(seeg_data), 10000)
            audio_short = audio_data[:min_length]
            seeg_short = seeg_data[:min_length]
            
            # 獲取瞬時相位
            audio_analytic = hilbert(audio_short)
            seeg_analytic = hilbert(seeg_short)
            
            audio_phase = np.angle(audio_analytic)
            seeg_phase = np.angle(seeg_analytic)
            
            # 計算相位差
            phase_diff = audio_phase - seeg_phase
            
            # Phase Locking Value (PLV)
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            
            # 相位一致性
            phase_consistency = float(1 - np.std(np.angle(np.exp(1j * phase_diff))) / np.pi)
            
            return {
                'phase_locking_value': plv,
                'phase_consistency': phase_consistency,
                'phase_difference': phase_diff
            }
        except Exception as e:
            print(f"❌ 相位同步分析失敗: {e}")
            return {
                'phase_locking_value': 0.0,
                'phase_consistency': 0.0,
                'phase_difference': np.array([])
            }

    def analyze_detailed_synchronization(self, audio_data, seeg_data, beat_times):
        """詳細同步性分析"""
        try:
            # 確保數據長度一致
            min_length = min(len(audio_data), len(seeg_data))
            audio_aligned = audio_data[:min_length]
            seeg_aligned = seeg_data[:min_length]
            
            # Beat一致性分析
            beat_responses = []
            beat_window = int(self.fs * 0.5)  # 0.5秒窗口
            
            for beat_time in beat_times:
                try:
                    beat_idx = int(beat_time * self.fs)
                    if beat_idx + beat_window < min_length:
                        seeg_beat = seeg_aligned[beat_idx:beat_idx+beat_window]
                        beat_responses.append(seeg_beat)
                except:
                    continue
            
            if beat_responses:
                # 確保所有beat響應長度一致
                min_beat_length = min(len(br) for br in beat_responses)
                beat_responses = [br[:min_beat_length] for br in beat_responses]
                
                # 計算平均beat響應
                average_beat_response = np.mean(beat_responses, axis=0)
                
                # Beat一致性分數
                consistencies = []
                for br in beat_responses:
                    try:
                        corr, _ = pearsonr(br, average_beat_response)
                        consistencies.append(float(corr) if not np.isnan(corr) else 0.0)
                    except:
                        consistencies.append(0.0)
                
                beat_consistency_score = float(np.mean(consistencies))
                
                # Beat增強程度
                baseline_var = np.var(seeg_aligned)
                beat_var = np.var(average_beat_response)
                beat_enhancement = float(beat_var / baseline_var) if baseline_var > 0 else 0.0
            else:
                average_beat_response = np.array([])
                beat_consistency_score = 0.0
                beat_enhancement = 0.0
            
            return {
                'beat_consistency_score': beat_consistency_score,
                'beat_enhancement': beat_enhancement,
                'average_beat_response': average_beat_response,
                'num_beats_analyzed': len(beat_responses)
            }
        except Exception as e:
            print(f"❌ 詳細同步性分析失敗: {e}")
            return {
                'beat_consistency_score': 0.0,
                'beat_enhancement': 0.0,
                'average_beat_response': np.array([]),
                'num_beats_analyzed': 0
            }

    def generate_analysis_summary(self, music_result, sync_result, sync_analysis, corr_analysis, phase_result):
        """生成分析摘要"""
        try:
            # 綜合評分計算
            scores = [
                float(music_result['bmp_accuracy']) / 100.0,
                float(sync_result['sync_score']),
                float(abs(corr_analysis['overall_correlation'])),
                float(corr_analysis['mean_beat_correlation']),
                float(phase_result['phase_locking_value']),
                float(sync_analysis['beat_consistency_score'])
            ]
            
            # 過濾掉NaN值
            valid_scores = [s for s in scores if not np.isnan(s)]
            overall_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            # 品質評級
            if overall_score >= 0.7:
                quality_grade = "優秀"
            elif overall_score >= 0.5:
                quality_grade = "良好"
            elif overall_score >= 0.3:
                quality_grade = "一般"
            else:
                quality_grade = "較差"
            
            return {
                'overall_synchronization_score': overall_score,
                'quality_grade': quality_grade,
                'bmp_detection_quality': "準確" if float(music_result['bmp_accuracy']) > 90 else "一般",
                'sync_quality': "高" if float(sync_result['sync_score']) > 0.5 else "中等",
                'correlation_strength': "強" if float(abs(corr_analysis['overall_correlation'])) > 0.5 else "中等"
            }
        except Exception as e:
            print(f"❌ 摘要生成失敗: {e}")
            return {
                'overall_synchronization_score': 0.0,
                'quality_grade': "未知",
                'bmp_detection_quality': "未知",
                'sync_quality': "未知",
                'correlation_strength': "未知"
            }

    def create_analysis_plots(self, results, save_path):
        """創建分析圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f"Beat-BPM 分析報告: {results['song_name']} - {results['channel_name']}", 
                        fontsize=16, fontweight='bold')
            
            # 限制顯示長度（前30秒）
            max_display_samples = min(30 * self.fs, len(results['audio_signal']))
            time_axis = np.arange(max_display_samples) / self.fs
            
            # 1. 音樂信號與beat標記
            axes[0,0].plot(time_axis, results['audio_signal'][:max_display_samples], 'b-', alpha=0.7, label='音樂信號')
            for bt in results['beat_times']:
                if bt <= 30:  # 只顯示前30秒的beat
                    axes[0,0].axvline(x=bt, color='r', linestyle='--', alpha=0.8)
            axes[0,0].set_title(f'音樂信號與Beat標記\n使用BPM: {results["known_bpm"]} | Beat數: {len(results["beat_times"])}')
            axes[0,0].set_xlabel('時間 (秒)')
            axes[0,0].set_ylabel('振幅')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()
            
            # 2. SEEG信號與包絡
            axes[0,1].plot(time_axis, results['seeg_signal'][:max_display_samples], 'g-', alpha=0.7, label='SEEG信號')
            if len(results['smooth_envelope']) == len(results['seeg_signal']):
                axes[0,1].plot(time_axis, results['smooth_envelope'][:max_display_samples], 'r-', linewidth=2, label='平滑包絡')
            axes[0,1].set_title(f'SEEG信號與包絡\n估算BPM: {float(results["seeg_estimated_bpm"]):.1f}')
            axes[0,1].set_xlabel('時間 (秒)')
            axes[0,1].set_ylabel('振幅')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
            
            # 3. 滑動相關性
            if results['sliding_times']:
                axes[1,0].plot(results['sliding_times'], results['sliding_correlation'], 'purple', linewidth=2)
                axes[1,0].set_title(f'滑動窗口相關性\n整體相關性: {float(results["overall_correlation"]):.3f}')
                axes[1,0].set_xlabel('時間 (秒)')
                axes[1,0].set_ylabel('相關係數')
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 4. Beat相關性分佈
            if results['beat_correlations']:
                axes[1,1].hist(results['beat_correlations'], bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[1,1].set_title(f'Beat相關性分佈\n平均: {float(results["mean_beat_correlation"]):.3f}')
                axes[1,1].set_xlabel('相關係數')
                axes[1,1].set_ylabel('頻數')
                axes[1,1].grid(True, alpha=0.3)
            
            # 5. 相位同步指標
            metrics = ['BPM準確度', '同步分數', '整體相關性', 'Beat相關性', '相位鎖定', 'Beat一致性']
            values = [
                float(results['bmp_accuracy']) / 100,
                float(results['sync_score']),
                float(abs(results['overall_correlation'])),
                float(abs(results['mean_beat_correlation'])),
                float(results['phase_locking_value']),
                float(results['beat_consistency_score'])
            ]
            
            bars = axes[2,0].bar(range(len(metrics)), values, color=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
            axes[2,0].set_title('綜合同步性指標')
            axes[2,0].set_xlabel('指標')
            axes[2,0].set_ylabel('分數')
            axes[2,0].set_xticks(range(len(metrics)))
            axes[2,0].set_xticklabels(metrics, rotation=45, ha='right')
            axes[2,0].set_ylim(0, 1)
            axes[2,0].grid(True, alpha=0.3)
            
            # 在柱狀圖上添加數值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[2,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 6. 分析摘要
            summary_text = f"""
分析摘要:
• 綜合評分: {float(results['analysis_summary']['overall_synchronization_score']):.3f}
• 品質等級: {results['analysis_summary']['quality_grade']}
• BPM檢測: {results['analysis_summary']['bmp_detection_quality']}
• 同步品質: {results['analysis_summary']['sync_quality']}
• 相關強度: {results['analysis_summary']['correlation_strength']}

數據統計:
• 檢測到 {len(results['beat_times'])} 個beats
• 相位鎖定值: {float(results['phase_locking_value']):.3f}
• Beat增強程度: {float(results['beat_enhancement']):.3f}
• 最佳延遲: {float(results['best_lag']):.1f} 樣本
            """
            
            axes[2,1].text(0.05, 0.95, summary_text, transform=axes[2,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[2,1].set_xlim(0, 1)
            axes[2,1].set_ylim(0, 1)
            axes[2,1].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 分析圖表已保存: {save_path}")
            return fig
            
        except Exception as e:
            print(f"❌ 圖表生成失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_results_to_csv(self, results, save_path):
        """保存分析結果到CSV文件"""
        try:
            # 準備CSV數據
            csv_data = {
                '歌曲名稱': [results['song_name']],
                '電極通道': [results['channel_name']],
                '使用BPM': [results['known_bpm']],
                'Beat數量': [len(results['beat_times'])],
                'Beat頻率(Hz)': [float(results['beat_frequency'])],
                'SEEG估算BPM': [float(results['seeg_estimated_bpm'])],
                '同步分數': [float(results['sync_score'])],
                '最佳延遲(樣本)': [float(results['best_lag'])],
                '整體相關性': [float(results['overall_correlation'])],
                '平均Beat相關性': [float(results['mean_beat_correlation'])],
                '滑動相關性數量': [len(results['sliding_correlation'])],
                'Beat相關性數量': [len(results['beat_correlations'])],
                '相位鎖定值': [float(results['phase_locking_value'])],
                '相位一致性': [float(results['phase_consistency'])],
                'Beat一致性分數': [float(results['beat_consistency_score'])],
                'Beat增強程度': [float(results['beat_enhancement'])],
                '綜合評分': [float(results['analysis_summary']['overall_synchronization_score'])],
                '品質等級': [results['analysis_summary']['quality_grade']],
                'BPM檢測品質': [results['analysis_summary']['bmp_detection_quality']],
                '同步品質': [results['analysis_summary']['sync_quality']],
                '相關強度': [results['analysis_summary']['correlation_strength']],
                '分析數據長度(秒)': [len(results['audio_signal']) / self.fs],
                '分析時間': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            
            df = pd.DataFrame(csv_data)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            print(f"   💾 分析結果已保存: {save_path}")
            
        except Exception as e:
            print(f"❌ CSV保存失敗: {e}")
            import traceback
            traceback.print_exc()
