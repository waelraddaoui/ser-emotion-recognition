import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from './api.config';

export interface EmotionResult {
  class_id: number;
  emotion: string;
  emoji: string;
  color: string;
  confidence: number;
  description: string;
  probabilities: number[];
  img_prob: string;
  img_mfcc: string;
  img_wave: string;
}

export interface Example {
  id: number;
  filename: string;
  emotion: string;
  emoji: string;
  audio_url: string;
}

export interface PredictionResponse {
  success: boolean;
  prediction_id: number;
  audio_id?: number | null;
  audio_url?: string | null;
  original_name?: string | null;
  result: EmotionResult;
}

export interface PredictionReportCreate {
  reason: string;
  comment?: string | null;
}

export interface PredictionReport {
  id: number;
  prediction_id: number;
  reason: string;
  comment?: string | null;
  status: string;
  admin_review_note?: string | null;
  reviewed_at?: string | null;
  created_at: string;
  reporter_email: string;
  prediction_emotion: string;
  prediction_confidence: number;
  audio_id?: number | null;
  audio_url?: string | null;
  original_name?: string | null;
  source?: string | null;
  reviewed_by_email?: string | null;
}

export interface PredictionReportListResponse {
  items: PredictionReport[];
  count: number;
}


export interface PredictionHistoryItem {
  id: number;
  emotion: string;
  emoji?: string | null;
  confidence: number;
  probabilities: number[];
  created_at: string;
  audio_id?: number | null;
  audio_url?: string | null;
  original_name?: string | null;
  source?: string | null;
}

export interface PredictionsListResponse {
  items: PredictionHistoryItem[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number;
  class_f1?: { emotion: string; f1: number }[];
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = API_BASE_URL;

  constructor(private http: HttpClient) {}

  getExamples(): Observable<Example[]> {
    return this.http.get<Example[]>(`${this.apiUrl}/examples`);
  }

  predictUpload(file: File): Observable<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<PredictionResponse>(`${this.apiUrl}/predict/upload`, formData);
  }

  predictExample(exampleId: number): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(`${this.apiUrl}/predict/example`, { example_id: exampleId });
  }

  /**
   * Fetch aggregated model evaluation metrics (precision, recall, f1, accuracy, per-class f1).
   * Endpoint expected: GET ${API_BASE_URL}/model/metrics
   */
  getModelMetrics(): Observable<ModelMetrics> {
    return this.http.get<ModelMetrics>(`${this.apiUrl}/model/metrics`);
  }

  reportPrediction(predictionId: number, payload: PredictionReportCreate): Observable<PredictionReport> {
    return this.http.post<PredictionReport>(`${this.apiUrl}/predictions/${predictionId}/report`, payload);
  }

  getMyPredictions(): Observable<PredictionsListResponse> {
    return this.http.get<PredictionsListResponse>(`${this.apiUrl}/me/predictions`);
  }

  fetchPredictionAudio(audioId: number): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/audio/${audioId}`, { responseType: 'blob' });
  }
}
