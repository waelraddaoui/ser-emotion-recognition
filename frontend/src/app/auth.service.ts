import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap, of } from 'rxjs';
import { API_BASE_URL } from './api.config';

export interface UserProfile {
  id: number;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: UserProfile;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = API_BASE_URL;
  private accessKey = 'access_token';
  private refreshKey = 'refresh_token';
  private userKey = 'user_profile';

  constructor(private http: HttpClient) {}

  login(email: string, password: string): Observable<TokenPair> {
    return this.http.post<TokenPair>(`${this.apiUrl}/auth/login`, { email, password }).pipe(
      tap((res) => this.storeSession(res))
    );
  }

  register(email: string, password: string): Observable<TokenPair> {
    return this.http.post<TokenPair>(`${this.apiUrl}/auth/register`, { email, password }).pipe(
      tap((res) => this.storeSession(res))
    );
  }

  refresh(): Observable<TokenPair> {
    const refreshToken = this.getRefreshToken();
    return this.http.post<TokenPair>(`${this.apiUrl}/auth/refresh`, { refresh_token: refreshToken }).pipe(
      tap((res) => this.storeSession(res))
    );
  }

  logout(): Observable<{ success: boolean }> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      this.clearSession();
      return of({ success: true });
    }
    return this.http.post<{ success: boolean }>(`${this.apiUrl}/auth/logout`, { refresh_token: refreshToken }).pipe(
      tap(() => this.clearSession())
    );
  }

  storeSession(tokens: TokenPair) {
    localStorage.setItem(this.accessKey, tokens.access_token);
    localStorage.setItem(this.refreshKey, tokens.refresh_token);
    localStorage.setItem(this.userKey, JSON.stringify(tokens.user));
  }

  clearSession() {
    localStorage.removeItem(this.accessKey);
    localStorage.removeItem(this.refreshKey);
    localStorage.removeItem(this.userKey);
  }

  getAccessToken(): string | null {
    return localStorage.getItem(this.accessKey);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(this.refreshKey);
  }

  getUser(): UserProfile | null {
    const raw = localStorage.getItem(this.userKey);
    return raw ? (JSON.parse(raw) as UserProfile) : null;
  }

  isLoggedIn(): boolean {
    return !!this.getAccessToken();
  }

  isAdmin(): boolean {
    const user = this.getUser();
    return !!user && user.role === 'ADMIN';
  }
}
