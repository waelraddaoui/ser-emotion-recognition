import { Component, OnDestroy, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NavigationEnd, Router } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { filter, Subscription } from 'rxjs';
import { ApiService, PredictionHistoryItem } from '../api.service';
import { AuthService, UserProfile } from '../auth.service';
import { TranslateService, TranslatePipe, TranslateDirective } from '@ngx-translate/core';

export interface PredictionHistoryEntry extends PredictionHistoryItem {
  playbackUrl?: string | null;
}

@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [CommonModule, TranslatePipe, TranslateDirective],
  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.css']
})
export class ProfileComponent implements OnInit, OnDestroy {
  user: UserProfile | null = null;
  predictions: PredictionHistoryEntry[] = [];
  isLoading = false;
  errorMsg: string | null = null;

  private routerSub?: Subscription;
  private audioObjectUrls: string[] = [];

  constructor(
    private auth: AuthService,
    private api: ApiService,
    private router: Router,
    private cdr: ChangeDetectorRef,
    private translate: TranslateService
  ) {}

  ngOnInit() {
    this.user = this.auth.getUser();
    this.loadPredictions();

    this.routerSub = this.router.events
      .pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd))
      .subscribe((event) => {
        if (event.urlAfterRedirects.startsWith('/profile')) {
          this.loadPredictions();
        }
      });
  }

  ngOnDestroy() {
    this.routerSub?.unsubscribe();
    this.revokeAudioUrls();
  }

  loadPredictions() {
    if (!this.auth.isLoggedIn()) {
      this.predictions = [];
      this.errorMsg = this.translate.instant('PROFILE.PLEASE_LOGIN');
      return;
    }

    this.isLoading = true;
    this.errorMsg = null;
    this.revokeAudioUrls();
    this.predictions = [];

    this.api.getMyPredictions().subscribe({
      next: (res) => {
        this.predictions = res.items ?? [];
        this.isLoading = false;
        this.loadAudioForPredictions();
        this.cdr.detectChanges();
      },
      error: (err: HttpErrorResponse) => {
        this.isLoading = false;
        if (err.status === 0) {
          this.errorMsg = this.translate.instant('PROFILE.SERVER_UNREACHABLE');
        } else if (err.status === 401) {
          this.errorMsg = this.translate.instant('PROFILE.SESSION_EXPIRED');
        } else {
          const detail = (err.error as { detail?: string } | null)?.detail;
          this.errorMsg = detail || this.translate.instant('PROFILE.LOAD_ERROR');
        }
        this.cdr.detectChanges();
      }
    });
  }

  private loadAudioForPredictions() {
    for (const item of this.predictions) {
      if (!item.audio_id) {
        continue;
      }
      this.api.fetchPredictionAudio(item.audio_id).subscribe({
        next: (blob) => {
          const url = URL.createObjectURL(blob);
          this.audioObjectUrls.push(url);
          item.playbackUrl = url;
          this.cdr.detectChanges();
        },
        error: () => {
          item.playbackUrl = null;
          this.cdr.detectChanges();
        }
      });
    }
  }

  private revokeAudioUrls() {
    for (const url of this.audioObjectUrls) {
      URL.revokeObjectURL(url);
    }
    this.audioObjectUrls = [];
  }

  sourceLabel(source?: string | null): string {
    return source === 'upload'
      ? this.translate.instant('PROFILE.SOURCE_UPLOAD')
      : this.translate.instant('PROFILE.SOURCE_RAVDESS');
  }
}
