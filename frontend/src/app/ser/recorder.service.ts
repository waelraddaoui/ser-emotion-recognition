import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export interface RecorderState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  volume: number;
}

@Injectable({
  providedIn: 'root'
})
export class RecorderService {
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private stream: MediaStream | null = null;
  private audioChunks: Blob[] = [];
  private startTime: number = 0;
  private pauseTime: number = 0;
  private recordedTime: number = 0;
  private dataRequestInterval: number | null = null;

  private stateSubject = new BehaviorSubject<RecorderState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    volume: 0,
  });

  private volumeSubject = new BehaviorSubject<number>(0);
  private timerInterval: number | null = null;

  state$ = this.stateSubject.asObservable();
  volume$ = this.volumeSubject.asObservable();

  constructor() {}

  async startRecording(): Promise<void> {
    try {
      this.audioChunks = [];
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });

      // Set up audio context for volume monitoring
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();
      const source = this.audioContext.createMediaStreamSource(this.stream);
      source.connect(this.analyser);

      // Create MediaRecorder with best format
      const options = this.getMediaRecorderOptions();
      this.mediaRecorder = new MediaRecorder(this.stream, options);

      this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onerror = (event: Event) => {
        console.error('MediaRecorder error:', event);
      };

      this.mediaRecorder.start();
      
      // Request data every 100ms to ensure we capture all audio chunks
      this.dataRequestInterval = window.setInterval(() => {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
          this.mediaRecorder.requestData();
        }
      }, 100);

      this.startTime = Date.now() - this.recordedTime;
      this.pauseTime = 0;

      this.updateState({ isRecording: true, isPaused: false, duration: this.recordedTime });
      this.startVolumeMonitoring();
      this.startTimer();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      throw new Error('Microphone access denied or not available');
    }
  }

  pauseRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.pause();
      this.pauseTime = Date.now();
      this.updateState({ isPaused: true });
      this.stopVolumeMonitoring();
    }
  }

  resumeRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
      this.mediaRecorder.resume();
      this.recordedTime += Date.now() - this.pauseTime;
      this.startTime = Date.now() - this.recordedTime;
      this.updateState({ isPaused: false });
      this.startVolumeMonitoring();
    }
  }

  stopRecording(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('No recording in progress'));
        return;
      }

      if (this.dataRequestInterval !== null) {
        clearInterval(this.dataRequestInterval);
        this.dataRequestInterval = null;
      }

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder?.mimeType || 'audio/wav';
        const blob = new Blob(this.audioChunks, { type: mimeType });
        this.cleanup();
        this.updateState({ isRecording: false, isPaused: false, duration: 0 });
        resolve(blob);
      };

      // Request any remaining data before stopping
      if (this.mediaRecorder.state === 'recording') {
        this.mediaRecorder.requestData();
      }
      this.mediaRecorder.stop();
      this.stopVolumeMonitoring();
      this.stopTimer();
    });
  }

  cancelRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    if (this.dataRequestInterval !== null) {
      clearInterval(this.dataRequestInterval);
      this.dataRequestInterval = null;
    }
    this.cleanup();
    this.audioChunks = [];
    this.recordedTime = 0;
    this.updateState({ isRecording: false, isPaused: false, duration: 0 });
    this.stopTimer();
    this.stopVolumeMonitoring();
  }

  private getMediaRecorderOptions(): MediaRecorderOptions {
    // Try formats in order of preference
    const mimeTypes = [
      'audio/wav',
      'audio/webm',
      'audio/mp4',
      'audio/ogg',
    ];

    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        return { mimeType };
      }
    }

    // Fallback to default
    return {};
  }

  private startVolumeMonitoring(): void {
    if (!this.analyser) return;

    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    const volumeCheck = () => {
      this.analyser?.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      const normalized = Math.min(average / 255, 1);
      this.volumeSubject.next(normalized);

      if (this.stateSubject.value.isRecording && !this.stateSubject.value.isPaused) {
        requestAnimationFrame(volumeCheck);
      }
    };
    volumeCheck();
  }

  private stopVolumeMonitoring(): void {
    this.volumeSubject.next(0);
  }

  private startTimer(): void {
    this.timerInterval = window.setInterval(() => {
      const state = this.stateSubject.value;
      if (state.isRecording && !state.isPaused) {
        const duration = Math.floor((Date.now() - this.startTime) / 1000);
        this.updateState({ duration });
      }
    }, 100);
  }

  private stopTimer(): void {
    if (this.timerInterval !== null) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }

  private cleanup(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    this.mediaRecorder = null;
    this.audioContext = null;
    this.analyser = null;
    this.stream = null;
  }

  private updateState(partial: Partial<RecorderState>): void {
    const current = this.stateSubject.value;
    this.stateSubject.next({ ...current, ...partial });
  }
}
