import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from './api.config';

export interface AdminUser {
  id: number;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface AdminUsersList {
  items: AdminUser[];
}

export interface AdminPrediction {
  id: number;
  emotion: string;
  confidence: number;
  probabilities: number[];
  created_at: string;
  audio_id?: number | null;
  audio_url?: string | null;
  user_email: string;
}

export interface AdminPredictionsList {
  items: AdminPrediction[];
}

@Injectable({
  providedIn: 'root'
})
export class AdminService {
  private apiUrl = API_BASE_URL;

  constructor(private http: HttpClient) {}

  getUsers(): Observable<AdminUsersList> {
    return this.http.get<AdminUsersList>(`${this.apiUrl}/admin/users`);
  }

  createUser(email: string, password: string, role: string): Observable<AdminUser> {
    return this.http.post<AdminUser>(`${this.apiUrl}/admin/users`, { email, password, role });
  }

  updateUser(userId: number, payload: { role?: string; is_active?: boolean; password?: string }): Observable<AdminUser> {
    return this.http.patch<AdminUser>(`${this.apiUrl}/admin/users/${userId}`, payload);
  }

  deleteUser(userId: number): Observable<{ success: boolean }> {
    return this.http.delete<{ success: boolean }>(`${this.apiUrl}/admin/users/${userId}`);
  }

  getPredictions(): Observable<AdminPredictionsList> {
    return this.http.get<AdminPredictionsList>(`${this.apiUrl}/admin/predictions`);
  }

  deletePrediction(predictionId: number): Observable<{ success: boolean }> {
    return this.http.delete<{ success: boolean }>(`${this.apiUrl}/admin/predictions/${predictionId}`);
  }
}
