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
export class RecorderV2Service {
  private audioContext: AudioContext | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private analyser: AnalyserNode | null = null;
  private stream: MediaStream | null = null;
  private audioBuffers: Float32Array[] = [];
  private startTime: number = 0;
  private pauseTime: number = 0;
  private recordedTime: number = 0;
  private isPaused: boolean = false;

  private readonly SAMPLE_RATE = 44100;
  private readonly BUFFER_SIZE = 4096;

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
      this.audioBuffers = [];
      this.recordedTime = 0;
      this.isPaused = false;

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

      // Create audio context with specific sample rate
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: this.SAMPLE_RATE
      });

      // Create analyser for volume monitoring
      this.analyser = this.audioContext.createAnalyser();
      const source = this.audioContext.createMediaStreamSource(this.stream);
      source.connect(this.analyser);

      // Create script processor to capture raw PCM
      this.scriptProcessor = this.audioContext.createScriptProcessor(
        this.BUFFER_SIZE,
        1, // mono input
        1  // mono output
      );

      this.scriptProcessor.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!this.isPaused) {
          const inputData = event.inputBuffer.getChannelData(0);
          const bufferCopy = new Float32Array(inputData);
          this.audioBuffers.push(bufferCopy);
        }
      };

      source.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioContext.destination);

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
    this.isPaused = true;
    this.pauseTime = Date.now();
    this.updateState({ isPaused: true });
    this.stopVolumeMonitoring();
  }

  resumeRecording(): void {
    this.isPaused = false;
    this.recordedTime += Date.now() - this.pauseTime;
    this.startTime = Date.now() - this.recordedTime;
    this.updateState({ isPaused: false });
    this.startVolumeMonitoring();
  }

  stopRecording(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      try {
        this.stopVolumeMonitoring();
        this.stopTimer();

        if (this.scriptProcessor) {
          this.scriptProcessor.disconnect();
          this.scriptProcessor.onaudioprocess = null;
        }

        if (this.stream) {
          this.stream.getTracks().forEach((track) => track.stop());
        }

        // Merge all audio buffers
        const totalLength = this.audioBuffers.reduce((acc, buf) => acc + buf.length, 0);
        const mergedAudio = new Float32Array(totalLength);
        let offset = 0;
        for (const buf of this.audioBuffers) {
          mergedAudio.set(buf, offset);
          offset += buf.length;
        }

        // Encode to WAV
        const wavBlob = this.encodeWAV(mergedAudio);
        this.cleanup();
        this.updateState({ isRecording: false, isPaused: false, duration: 0 });
        resolve(wavBlob);
      } catch (error) {
        reject(error);
      }
    });
  }

  cancelRecording(): void {
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor.onaudioprocess = null;
      this.scriptProcessor = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }

    this.cleanup();
    this.audioBuffers = [];
    this.recordedTime = 0;
    this.updateState({ isRecording: false, isPaused: false, duration: 0 });
    this.stopTimer();
    this.stopVolumeMonitoring();
  }

  private encodeWAV(pcmAudio: Float32Array): Blob {
    const dataLength = pcmAudio.length * 2;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    // RIFF chunk
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, 'WAVE');

    // fmt sub-chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // subchunk1Size
    view.setUint16(20, 1, true); // audioFormat (1 = PCM)
    view.setUint16(22, 1, true); // numChannels (1 = mono)
    view.setUint32(24, this.SAMPLE_RATE, true); // sampleRate
    view.setUint32(28, this.SAMPLE_RATE * 2, true); // byteRate
    view.setUint16(32, 2, true); // blockAlign
    view.setUint16(34, 16, true); // bitsPerSample

    // data sub-chunk
    writeString(36, 'data');
    view.setUint32(40, dataLength, true);

    // Write PCM samples as 16-bit signed integers
    let offset = 44;
    const volume = 0.8; // scale to prevent clipping
    for (let i = 0; i < pcmAudio.length; i++) {
      const s = Math.max(-1, Math.min(1, pcmAudio[i] * volume));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }

  private startVolumeMonitoring(): void {
    if (!this.analyser) return;

    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    const volumeCheck = () => {
      this.analyser?.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      const normalized = Math.min(average / 255, 1);
      this.volumeSubject.next(normalized);

      if (this.stateSubject.value.isRecording) {
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
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    this.audioContext = null;
    this.scriptProcessor = null;
    this.analyser = null;
    this.stream = null;
  }

  private updateState(partial: Partial<RecorderState>): void {
    const current = this.stateSubject.value;
    this.stateSubject.next({ ...current, ...partial });
  }
}
