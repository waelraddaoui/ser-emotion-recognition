import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { BehaviorSubject, catchError, filter, switchMap, take, throwError } from 'rxjs';
import { AuthService } from './auth.service';
import { API_BASE_URL } from './api.config';

let isRefreshing = false;
const refreshSubject = new BehaviorSubject<string | null>(null);

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const accessToken = auth.getAccessToken();
  const tokenFromStorage = localStorage.getItem('access_token');
  const tokenToUse = accessToken || tokenFromStorage;

  let authReq = req;
  if (tokenToUse && req.url.startsWith(API_BASE_URL)) {
    authReq = req.clone({
      setHeaders: { Authorization: `Bearer ${tokenToUse}` },
    });
  }

  if (req.url.includes('/auth/')) {
    return next(authReq);
  }

  return next(authReq).pipe(
    catchError((error: HttpErrorResponse) => {
      if (error.status !== 401 || !auth.getRefreshToken()) {
        return throwError(() => error);
      }

      if (!isRefreshing) {
        isRefreshing = true;
        refreshSubject.next(null);

        return auth.refresh().pipe(
          switchMap((tokens) => {
            isRefreshing = false;
            refreshSubject.next(tokens.access_token);
            const retryReq = req.clone({
              setHeaders: { Authorization: `Bearer ${tokens.access_token}` },
            });
            return next(retryReq);
          }),
          catchError((refreshError) => {
            isRefreshing = false;
            auth.clearSession();
            return throwError(() => refreshError);
          })
        );
      }

      return refreshSubject.pipe(
        filter((token) => token !== null),
        take(1),
        switchMap((token) => {
          const retryReq = req.clone({
            setHeaders: { Authorization: `Bearer ${token}` },
          });
          return next(retryReq);
        })
      );
    })
  );
};
