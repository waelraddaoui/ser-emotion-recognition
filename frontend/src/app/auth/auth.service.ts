import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { tap } from 'rxjs/operators';
import { API_BASE_URL } from '../api.config';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: {
    id: number;
    email: string;
    role: string;
    is_active: boolean;
    created_at: string;
  };
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private accessTokenSubject = new BehaviorSubject<string | null>(this.getStoredAccessToken());
  private refreshTokenSubject = new BehaviorSubject<string | null>(this.getStoredRefreshToken());

  public accessToken$ = this.accessTokenSubject.asObservable();
  public refreshToken$ = this.refreshTokenSubject.asObservable();

  constructor(private http: HttpClient) {}

  login(email: string, password: string): Observable<TokenPair> {
    return this.http.post<TokenPair>(`${API_BASE_URL}/auth/login`, {
      email,
      password
    }).pipe(
      tap(response => {
        this.setTokens(response.access_token, response.refresh_token);
      })
    );
  }

  register(email: string, password: string): Observable<TokenPair> {
    return this.http.post<TokenPair>(`${API_BASE_URL}/auth/register`, {
      email,
      password
    }).pipe(
      tap(response => {
        this.setTokens(response.access_token, response.refresh_token);
      })
    );
  }

  refresh(): Observable<TokenPair> {
    const refreshToken = this.getStoredRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }
    return this.http.post<TokenPair>(`${API_BASE_URL}/auth/refresh`, {
      refresh_token: refreshToken
    }).pipe(
      tap(response => {
        this.setTokens(response.access_token, response.refresh_token);
      })
    );
  }

  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    this.accessTokenSubject.next(null);
    this.refreshTokenSubject.next(null);
  }

  clearSession(): void {
    this.logout();
  }

  getAccessToken(): string | null {
    return this.getStoredAccessToken();
  }

  getRefreshToken(): string | null {
    return this.getStoredRefreshToken();
  }

  isAuthenticated(): boolean {
    return !!this.getStoredAccessToken();
  }

  private setTokens(accessToken: string, refreshToken: string): void {
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('refresh_token', refreshToken);
    this.accessTokenSubject.next(accessToken);
    this.refreshTokenSubject.next(refreshToken);
  }

  private getStoredAccessToken(): string | null {
    return localStorage.getItem('access_token');
  }

  private getStoredRefreshToken(): string | null {
    return localStorage.getItem('refresh_token');
  }
}
