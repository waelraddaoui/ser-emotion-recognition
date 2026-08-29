import { Component, OnInit, ChangeDetectorRef, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { TranslateService, TranslatePipe, TranslateDirective } from '@ngx-translate/core';
import { ApiService, Example, EmotionResult } from '../api.service';
import { API_BASE_URL } from '../api.config';
import { RecorderV2Service, RecorderState } from './recorder-v2.service';

const EMOTION_NAMES = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'] as const;
const EMOTION_EMOJIS = ['😐', '😌', '😊', '😢', '😠', '😨', '🤢', '😲'] as const;
const EMOTION_COLORS = ['#607D8B', '#9C27B0', '#FFC107', '#2196F3', '#F44336', '#673AB7', '#4CAF50', '#FF6F00'] as const;

type ConfidenceTier = 'high' | 'medium' | 'low';

interface RankedEmotion {
  rank: number;
  emotion: string;
  emoji: string;
  color: string;
  confidence: number;
}

interface AudioQualityIssue {
  severity: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
}

interface AudioQualityReport {
  score: number;
  verdict: 'good' | 'warning' | 'poor';
  duration: number;
  rms: number;
  peak: number;
  clippingRatio: number;
  silenceRatio: number;
  issues: AudioQualityIssue[];
  recommendations: string[];
}

@Component({
  selector: 'app-ser',
  standalone: true,
  imports: [CommonModule, FormsModule, TranslatePipe, TranslateDirective],
  templateUrl: './ser.component.html',
  styleUrls: ['./ser.component.css']
})
export class SerComponent implements OnInit, OnDestroy {
  activeTab: 'upload' | 'example' | 'record' = 'upload';
  examples: Example[] = [];
  selectedExample: Example | null = null;
  selectedFile: File | null = null;
  recordedBlob: Blob | null = null;
  currentResult: EmotionResult | null = null;
  currentPredictionId: number | null = null;
  isLoading: boolean = false;
  errorMsg: string | null = null;
  audioUrl: string | null = null;
  reportReason: string = 'wrong_emotion';
  reportComment: string = '';
  isReporting: boolean = false;
  reportMessage: string | null = null;
  reportError: string | null = null;
  audioQuality: AudioQualityReport | null = null;
  qualityScanState: 'idle' | 'scanning' | 'ready' | 'error' = 'idle';
  qualityScanError: string | null = null;
  topPredictions: RankedEmotion[] = [];
  decisionGap = 0;
  confidenceTier: ConfidenceTier = 'low';
  explainabilityNote = '';

  // Studio does not display global model performance (moved to Admin)

  private qualityScanPromise: Promise<void> | null = null;
  private qualityScanToken = 0;
  
  recorderState: RecorderState = {
    isRecording: false,
    isPaused: false,
    duration: 0,
    volume: 0,
  };
  recordingError: string | null = null;
  volume: number = 0;

  private destroy$ = new Subject<void>();

  constructor(
    private apiService: ApiService,
    private cdr: ChangeDetectorRef,
    private recorderService: RecorderV2Service,
    private translate: TranslateService
  ) {}

  ngOnInit() {
    this.loadExamples();
    // performance metrics are shown on Admin only
    this.recorderService.state$
      .pipe(takeUntil(this.destroy$))
      .subscribe((state) => {
        this.recorderState = state;
        this.cdr.detectChanges();
      });

    this.recorderService.volume$
      .pipe(takeUntil(this.destroy$))
      .subscribe((vol) => {
        this.volume = vol;
        this.cdr.detectChanges();
      });
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
    this.recorderService.cancelRecording();
  }

  loadExamples() {
    this.apiService.getExamples().subscribe({
      next: (res) => {
        this.examples = res;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading examples', err);
      }
    });
  }

  setActiveTab(tab: 'upload' | 'example' | 'record') {
    this.activeTab = tab;
    this.currentResult = null;
    this.currentPredictionId = null;
    this.resetExplainability();
    this.resetAudioQuality();
    this.errorMsg = null;
    this.recordingError = null;
    this.reportMessage = null;
    this.reportError = null;
    if (tab !== 'record' && this.recorderState.isRecording) {
      this.recorderService.cancelRecording();
    }
    this.cdr.detectChanges();
  }

  selectExample(ex: Example) {
    this.selectedExample = ex;
    this.currentResult = null;
    this.currentPredictionId = null;
    this.errorMsg = null;
    this.reportMessage = null;
    this.reportError = null;

    if (this.audioUrl && this.selectedFile) {
      URL.revokeObjectURL(this.audioUrl);
    }
    this.audioUrl = `${API_BASE_URL}${ex.audio_url}`;
    this.selectedFile = null;
    this.resetAudioQuality();
    this.cdr.detectChanges();
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      if (this.audioUrl) {
        URL.revokeObjectURL(this.audioUrl);
      }
      this.audioUrl = URL.createObjectURL(file);
      this.selectedFile = file;

      this.selectedExample = null;
      this.currentResult = null;
      this.currentPredictionId = null;
      this.errorMsg = null;
      this.reportMessage = null;
      this.reportError = null;
      this.resetAudioQuality();
      void this.queueAudioQualityScan(file);
      this.cdr.detectChanges();
    }
  }

  async analyze() {
    this.errorMsg = null;
    this.currentResult = null;
    this.resetExplainability();

    await this.waitForAudioQualityScan();

    if (this.activeTab === 'upload') {
      if (!this.selectedFile) {
        return;
      }
      this.isLoading = true;
      this.apiService.predictUpload(this.selectedFile).subscribe({
        next: (res) => {
          this.isLoading = false;
          if (res.success) {
            this.currentResult = res.result;
            this.applyExplainability(res.result);
            this.currentPredictionId = res.prediction_id;
            this.reportMessage = null;
            this.reportError = null;
          } else {
            this.errorMsg = 'Analysis failed.';
          }
          this.cdr.detectChanges();
        },
        error: (err: HttpErrorResponse) => {
          this.isLoading = false;
          if (err.status === 0) {
            this.errorMsg = 'Server unreachable. Please start the backend.';
          } else {
            const detail = (err.error as { detail?: string } | null)?.detail;
            this.errorMsg = detail || 'AI error: invalid or corrupted audio file.';
          }
          console.error(err);
          this.cdr.detectChanges();
        }
      });
    } else if (this.activeTab === 'example') {
      if (!this.selectedExample) {
        return;
      }
      this.isLoading = true;
      this.apiService.predictExample(this.selectedExample.id).subscribe({
        next: (res) => {
          this.isLoading = false;
          if (res.success) {
            this.currentResult = res.result;
            this.applyExplainability(res.result);
            this.currentPredictionId = res.prediction_id;
            this.reportMessage = null;
            this.reportError = null;
          } else {
            this.errorMsg = 'Analysis failed.';
          }
          this.cdr.detectChanges();
        },
        error: (err) => {
          this.isLoading = false;
          this.errorMsg = 'Server error: cannot reach AI service.';
          console.error(err);
          this.cdr.detectChanges();
        }
      });
    } else if (this.activeTab === 'record') {
      if (!this.recordedBlob) {
        return;
      }
      this.isLoading = true;
      
      // Déterminer l'extension selon le MIME type
      let ext = '.wav';
      if (this.recordedBlob.type.includes('webm')) {
        ext = '.webm';
      } else if (this.recordedBlob.type.includes('ogg')) {
        ext = '.ogg';
      } else if (this.recordedBlob.type.includes('mp4')) {
        ext = '.mp4';
      }
      
      const filename = `recording-${Date.now()}${ext}`;
      const file = new File([this.recordedBlob], filename, {
        type: this.recordedBlob.type,
      });
      
      this.apiService.predictUpload(file).subscribe({
        next: (res) => {
          this.isLoading = false;
          if (res.success) {
            this.currentResult = res.result;
            this.applyExplainability(res.result);
            this.currentPredictionId = res.prediction_id;
            this.reportMessage = null;
            this.reportError = null;
          } else {
            this.errorMsg = 'Analysis failed.';
          }
          this.cdr.detectChanges();
        },
        error: (err: HttpErrorResponse) => {
          this.isLoading = false;
          if (err.status === 0) {
            this.errorMsg = 'Server unreachable. Please start the backend.';
          } else {
            const detail = (err.error as { detail?: string } | null)?.detail;
            this.errorMsg = detail || 'AI error: invalid or corrupted audio file.';
          }
          console.error(err);
          this.cdr.detectChanges();
        }
      });
    }
  }

  async startRecording() {
    this.recordingError = null;
    this.recordedBlob = null;
    this.currentResult = null;
    this.currentPredictionId = null;
    this.errorMsg = null;
    this.reportMessage = null;
    this.reportError = null;
    try {
      await this.recorderService.startRecording();
    } catch (error: any) {
      this.recordingError = error.message || 'Failed to start recording';
      this.cdr.detectChanges();
    }
  }

  pauseRecording() {
    this.recorderService.pauseRecording();
  }

  resumeRecording() {
    this.recorderService.resumeRecording();
  }

  async stopRecording() {
    try {
      this.recordedBlob = await this.recorderService.stopRecording();
      if (this.audioUrl && !this.selectedFile) {
        URL.revokeObjectURL(this.audioUrl);
      }
      this.audioUrl = URL.createObjectURL(this.recordedBlob);
      this.selectedFile = null;
      this.selectedExample = null;
      this.currentPredictionId = null;
      this.reportMessage = null;
      this.reportError = null;
      this.resetAudioQuality();
      void this.queueAudioQualityScan(this.recordedBlob);
      this.cdr.detectChanges();
    } catch (error: any) {
      this.recordingError = error.message || 'Failed to stop recording';
      this.cdr.detectChanges();
    }
  }

  cancelRecording() {
    this.recorderService.cancelRecording();
    this.recordedBlob = null;
    this.recordingError = null;
    this.currentPredictionId = null;
    this.reportMessage = null;
    this.reportError = null;
    this.resetAudioQuality();
    if (this.audioUrl) {
      URL.revokeObjectURL(this.audioUrl);
      this.audioUrl = null;
    }
    this.cdr.detectChanges();
  }

  reportPrediction() {
    if (!this.currentPredictionId || this.activeTab === 'example') {
      return;
    }

    this.reportMessage = null;
    this.reportError = null;
    this.isReporting = true;

    this.apiService.reportPrediction(this.currentPredictionId, {
      reason: this.reportReason,
      comment: this.reportComment?.trim() || null,
    }).subscribe({
      next: () => {
        this.isReporting = false;
        this.reportMessage = this.translate.instant('STUDIO.REPORT_SENT');
        this.reportComment = '';
        this.cdr.detectChanges();
      },
      error: (err: HttpErrorResponse) => {
        this.isReporting = false;
        const detail = (err.error as { detail?: string } | null)?.detail;
        this.reportError = detail || 'Could not submit the report.';
        this.cdr.detectChanges();
      }
    });
  }

  getConfidenceTierLabel(): string {
    switch (this.confidenceTier) {
      case 'high':
        return this.translate.instant('STUDIO.CONFIDENCE_HIGH');
      case 'medium':
        return this.translate.instant('STUDIO.CONFIDENCE_MEDIUM');
      default:
        return this.translate.instant('STUDIO.CONFIDENCE_LOW');
    }
  }

  getQualityTierLabel(): string {
    if (!this.audioQuality) {
      return this.translate.instant('STUDIO.QUALITY_PENDING');
    }

    switch (this.audioQuality.verdict) {
      case 'good':
        return this.translate.instant('STUDIO.QUALITY_READY');
      case 'warning':
        return this.translate.instant('STUDIO.QUALITY_REVIEW');
      default:
        return this.translate.instant('STUDIO.QUALITY_POOR');
    }
  }

  getQualitySummary(): string {
    if (!this.audioQuality) {
      return '';
    }

    const report = this.audioQuality;
    return `Duration ${report.duration.toFixed(1)}s · RMS ${(report.rms * 100).toFixed(1)}% · Peak ${(report.peak * 100).toFixed(1)}%`;
  }

  getQualityActionHint(): string {
    if (!this.audioQuality) {
      return this.translate.instant('STUDIO.QUALITY_HINTS.PENDING');
    }

    if (this.audioQuality.verdict === 'good') {
      return this.translate.instant('STUDIO.QUALITY_HINTS.READY');
    }

    return this.translate.instant('STUDIO.QUALITY_HINTS.FIX');
  }

  private resetExplainability() {
    this.topPredictions = [];
    this.decisionGap = 0;
    this.confidenceTier = 'low';
    this.explainabilityNote = '';
  }

  private resetAudioQuality() {
    this.audioQuality = null;
    this.qualityScanState = 'idle';
    this.qualityScanError = null;
    this.qualityScanPromise = null;
    this.qualityScanToken += 1;
  }

  private async waitForAudioQualityScan() {
    if (this.qualityScanPromise) {
      await this.qualityScanPromise;
    }
  }

  private queueAudioQualityScan(source: Blob) {
    const token = this.qualityScanToken;
    this.qualityScanState = 'scanning';
    this.qualityScanError = null;

    const scanPromise = this.inspectAudioQuality(source)
      .then((report) => {
        if (token !== this.qualityScanToken) {
          return;
        }
        this.audioQuality = report;
        this.qualityScanState = 'ready';
      })
      .catch((error) => {
        if (token !== this.qualityScanToken) {
          return;
        }
        console.error('Audio quality scan failed', error);
        this.audioQuality = null;
        this.qualityScanState = 'error';
        this.qualityScanError = 'Could not inspect audio quality before prediction.';
      })
      .finally(() => {
        if (this.qualityScanPromise === scanPromise) {
          this.qualityScanPromise = null;
        }
        if (token === this.qualityScanToken) {
          this.cdr.detectChanges();
        }
      });

    this.qualityScanPromise = scanPromise;
  }

  private async inspectAudioQuality(source: Blob): Promise<AudioQualityReport> {
    const audioBuffer = await this.decodeAudioBlob(source);
    return this.buildAudioQualityReport(audioBuffer);
  }

  private async decodeAudioBlob(source: Blob): Promise<AudioBuffer> {
    const audioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!audioContextClass) {
      throw new Error('Web Audio API is not available in this browser');
    }

    const arrayBuffer = await source.arrayBuffer();
    const audioContext = new audioContextClass();

    try {
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      return decoded;
    } finally {
      await audioContext.close();
    }
  }

  private buildAudioQualityReport(audioBuffer: AudioBuffer): AudioQualityReport {
    const samples = audioBuffer.getChannelData(0);
    const windowSize = Math.max(1024, Math.floor(audioBuffer.sampleRate * 0.025));

    let totalEnergy = 0;
    let peak = 0;
    let clippedSamples = 0;
    let silentWindows = 0;
    let processedWindows = 0;

    for (let offset = 0; offset < samples.length; offset += windowSize) {
      const end = Math.min(offset + windowSize, samples.length);
      let windowEnergy = 0;
      let windowPeak = 0;

      for (let index = offset; index < end; index += 1) {
        const sample = samples[index];
        const absSample = Math.abs(sample);
        totalEnergy += sample * sample;
        windowEnergy += sample * sample;
        windowPeak = Math.max(windowPeak, absSample);
        peak = Math.max(peak, absSample);
        if (absSample >= 0.98) {
          clippedSamples += 1;
        }
      }

      const windowLength = Math.max(1, end - offset);
      const windowRms = Math.sqrt(windowEnergy / windowLength);
      if (windowRms < 0.015 || windowPeak < 0.03) {
        silentWindows += 1;
      }
      processedWindows += 1;
    }

    const duration = audioBuffer.duration;
    const rms = Math.sqrt(totalEnergy / Math.max(1, samples.length));
    const clippingRatio = clippedSamples / Math.max(1, samples.length);
    const silenceRatio = silentWindows / Math.max(1, processedWindows);

    const issues: AudioQualityIssue[] = [];
    const recommendations: string[] = [];
    let score = 100;

    if (duration < 1.5) {
      score -= 35;
      issues.push({
        severity: 'critical',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.TOO_SHORT_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.TOO_SHORT_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.RECORD_LONGER'));
    } else if (duration < 2.5) {
      score -= 10;
      issues.push({
        severity: 'warning',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.SHORT_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.SHORT_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.EXTEND_UTTERANCE'));
    }

    if (rms < 0.01) {
      score -= 30;
      issues.push({
        severity: 'critical',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.LOW_SIGNAL_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.LOW_SIGNAL_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.SPEAK_LOUDER'));
    } else if (rms < 0.02) {
      score -= 15;
      issues.push({
        severity: 'warning',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.QUIET_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.QUIET_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.INCREASE_GAIN'));
    }

    if (silenceRatio > 0.45) {
      score -= 25;
      issues.push({
        severity: 'warning',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.SILENCE_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.SILENCE_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.TRIM_PAUSES'));
    }

    if (clippingRatio > 0.02) {
      score -= 30;
      issues.push({
        severity: 'critical',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.CLIPPING_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.CLIPPING_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.REDUCE_GAIN'));
    } else if (clippingRatio > 0.005) {
      score -= 10;
      issues.push({
        severity: 'warning',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.MILD_CLIPPING_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.MILD_CLIPPING_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.AVOID_LOUD'));
    }

    if (peak < 0.08) {
      score -= 10;
      issues.push({
        severity: 'info',
        title: this.translate.instant('STUDIO.QUALITY_ISSUES.WEAK_PEAKS_TITLE'),
        message: this.translate.instant('STUDIO.QUALITY_ISSUES.WEAK_PEAKS_MSG')
      });
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.MIC_CLOSER'));
    }

    if (issues.length === 0) {
      recommendations.push(this.translate.instant('STUDIO.QUALITY_SUGGESTIONS.READY'));
    }

    const finalScore = Math.max(0, Math.min(100, score));
    let verdict: AudioQualityReport['verdict'] = 'good';
    if (finalScore < 55 || duration < 1.5 || rms < 0.01 || clippingRatio > 0.02) {
      verdict = 'poor';
    } else if (finalScore < 80 || issues.some((issue) => issue.severity !== 'info')) {
      verdict = 'warning';
    }

    return {
      score: finalScore,
      verdict,
      duration,
      rms,
      peak,
      clippingRatio,
      silenceRatio,
      issues,
      recommendations,
    };
  }

  private applyExplainability(result: EmotionResult) {
    if (!result?.probabilities?.length) {
      this.resetExplainability();
      return;
    }

    const ranked = result.probabilities
      .map((confidence, index) => ({ index, confidence }))
      .sort((left, right) => right.confidence - left.confidence)
      .slice(0, 3);

    this.topPredictions = ranked.map((item, position) => ({
      rank: position + 1,
      emotion: EMOTION_NAMES[item.index] ?? `Class ${item.index + 1}`,
      emoji: EMOTION_EMOJIS[item.index] ?? '🎙️',
      color: EMOTION_COLORS[item.index] ?? '#94a3b8',
      confidence: item.confidence,
    }));

    const topConfidence = this.topPredictions[0]?.confidence ?? 0;
    const runnerUpConfidence = this.topPredictions[1]?.confidence ?? 0;
    this.decisionGap = Math.max(topConfidence - runnerUpConfidence, 0);

    if (topConfidence >= 0.8 && this.decisionGap >= 0.15) {
      this.confidenceTier = 'high';
      this.explainabilityNote = this.translate.instant('STUDIO.EXPLAIN_HIGH', {
        emotion1: this.topPredictions[0].emotion,
        emotion2: this.topPredictions[1]?.emotion ?? ''
      });
      return;
    }

    if (topConfidence >= 0.6) {
      this.confidenceTier = 'medium';
      this.explainabilityNote = this.translate.instant('STUDIO.EXPLAIN_MEDIUM', {
        emotion1: this.topPredictions[0].emotion,
        emotion2: this.topPredictions[1]?.emotion ?? ''
      });
      return;
    }

    this.confidenceTier = 'low';
    this.explainabilityNote = this.translate.instant('STUDIO.EXPLAIN_LOW');
  }

  formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      if (this.audioUrl) {
        URL.revokeObjectURL(this.audioUrl);
      }
      this.audioUrl = URL.createObjectURL(file);
      this.selectedFile = file;
      this.selectedExample = null;
      this.currentResult = null;
      this.currentPredictionId = null;
      this.errorMsg = null;
      this.reportMessage = null;
      this.reportError = null;
      this.resetAudioQuality();
      void this.queueAudioQualityScan(file);
      this.cdr.detectChanges();
    }
  }
}

