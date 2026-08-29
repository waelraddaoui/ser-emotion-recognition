import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { TranslateService, TranslatePipe, TranslateDirective } from '@ngx-translate/core';
import { API_BASE_URL } from '../api.config';
import { ApiService, ModelMetrics } from '../api.service';

@Component({
  selector: 'app-admin-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, TranslatePipe, TranslateDirective],
  templateUrl: './admin-dashboard.component.html',
  styleUrls: ['./admin-dashboard.component.css']
})
export class AdminDashboardComponent implements OnInit, OnDestroy {
  apiBaseUrl = API_BASE_URL;
  activeSection: 'dashboard' | 'explorer' | 'performance' | 'reports' | 'users' | 'batch' = 'dashboard';
  readonly emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'];
  // Dashboard data
  dashboardData: any = null;
  dashboardLoading = false;

  // Explorer data
  allPredictions: any[] = [];
  filteredPredictions: any[] = [];
  explorerLoading = false;
  selectedEmotion = 'all';
  minConfidence = 0;
  selectedPrediction: any = null;
  selectedPredictionAudioUrl: string | null = null;
  selectedPredictionAudioLoading = false;
  selectedPredictionAudioError: string | null = null;
  isPredictionAudioPlaying = false;
  curationEmotion = 'Happy';
  curationResult: any = null;
  curationError: string | null = null;
  curatedDatasetStats: any = null;

  // Performance data
  performanceData: any = null;
  performanceLoading = false;

  // Reports data
  reports: any[] = [];
  reportsLoading = false;
  selectedReport: any = null;
  reportStatusFilter = 'all';
  reportReview = { status: 'CONFIRMED', admin_review_note: '' };
  correctedEmotion = 'Happy';
  learningResult: any = null;
  learningError: string | null = null;
  retestResult: any = null;
  selectedReportAudioUrl: string | null = null;
  selectedReportAudioLoading = false;
  selectedReportAudioError: string | null = null;

  // UI states
  actionLoading = false;
  showConfirmModal = false;
  confirmAction: { type: 'REVIEW' | 'RETEST' | null, payload?: any } = { type: null };
  toastMessage: string | null = null;
  toastVisible = false;
  isAudioPlaying = false;
  showTrainingLog = true;

  toggleTrainingLog() {
    this.showTrainingLog = !this.showTrainingLog;
  }


  // Users data
  usersStats: any[] = [];
  usersLoading = false;

  // Training data
  trainingStatus: any = null;
  trainingLoading = false;
  trainingError: string | null = null;
  private trainingPollId: number | null = null;

  // Training step status & parsed metrics
  trainingStep: 'idle' | 'preparing' | 'training' | 'reloading' | 'completed' | 'failed' = 'idle';
  parsedEpochs: any[] = [];
  currentEpoch = 0;
  maxEpochs = 100;
  currentLoss = 0;
  currentAcc = 0;
  currentValLoss = 0;
  currentValAcc = 0;

  // Real SVG chart paths
  realLossPath: string | null = null;
  realAccPath: string | null = null;
  realLossAreaPath: string | null = null;
  realAccAreaPath: string | null = null;
  realValLossPath: string | null = null;
  realValAccPath: string | null = null;

  // Mockup Chart Properties
  lossPath: string | null = null;
  accuracyPath: string | null = null;
  metricsAvailable = false;
  totalSamples = 0;
  testAccuracy = 0;
  valAccuracyPath: string | null = null;
  valLossPath: string | null = null;
  testLoss = 0;

  // User Management
  showUserModal = false;
  editingUserId: number | null = null;
  userForm = { email: '', password: '', role: 'USER', is_active: true };

  // Performance metrics display
  metrics: ModelMetrics | null = null;
  // animated display values
  displayAccuracy = 0;
  displayPrecision = 0;
  displayRecall = 0;
  displayF1 = 0;

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef, private api: ApiService, private translate: TranslateService) {}

  ngOnDestroy() {
    this.revokeSelectedReportAudioUrl();
    this.revokeSelectedPredictionAudioUrl();
    this.stopTrainingPolling();
  }

  ngOnInit() {
    this.loadDashboard();
  }

  loadDashboard() {
    this.dashboardLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get<any>(`${API_BASE_URL}/admin/analytics/dashboard`, { headers }).subscribe({
      next: (data) => {
        this.dashboardData = data;
        this.dashboardLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load dashboard:', err);
        this.dashboardLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  loadExplorer() {
    this.explorerLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.loadCuratedDatasetStats();
    this.http.get<any>(`${API_BASE_URL}/admin/analytics/predictions-list`, {
      headers,
      params: {
        emotion: this.selectedEmotion,
        min_confidence: this.minConfidence / 100
      }
    }).subscribe({
      next: (data) => {
        this.allPredictions = data.items;
        this.filteredPredictions = data.items;
        this.explorerLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load predictions:', err);
        this.explorerLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  loadPerformance() {
    this.performanceLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get<any>(`${API_BASE_URL}/admin/analytics/low-confidence`, { headers }).subscribe({
      next: (data) => {
        this.performanceData = data || { items: [], count: 0 };
        this.performanceLoading = false;
        // fetch aggregated model metrics to display in the performance panel
        this.fetchModelMetrics();
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load performance data:', err);
        this.performanceData = { items: [], count: 0 };
        this.performanceLoading = false;
        this.fetchModelMetrics();
        this.cdr.detectChanges();
      }
    });
  }

  fetchModelMetrics() {
    this.api.getModelMetrics().subscribe({
      next: (m) => {
        this.metrics = m;
        // animate counters
        this.animateValue('accuracy', Math.round((m.accuracy || 0) * 100));
        this.animateValue('precision', Math.round((m.precision || 0) * 100));
        this.animateValue('recall', Math.round((m.recall || 0) * 100));
        this.animateValue('f1', Math.round((m.f1 || 0) * 100));
        this.cdr.detectChanges();
      },
      error: (err) => {
        // silently ignore if endpoint unavailable
        console.warn('Model metrics not available:', err);
      }
    });
  }

  animateValue(key: 'accuracy' | 'precision' | 'recall' | 'f1', target: number) {
    const duration = 800;
    const start = performance.now();
    const from = this[`display${key[0].toUpperCase() + key.slice(1)}` as keyof this] as number || 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const value = Math.round(from + (target - from) * t);
      this[`display${key[0].toUpperCase() + key.slice(1)}` as keyof this] = value as any;
      this.cdr.detectChanges();
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  loadReports() {
    this.reportsLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get<any>(`${API_BASE_URL}/admin/reports`, {
      headers,
      params: {
        status: this.reportStatusFilter
      }
    }).subscribe({
      next: (data) => {
        this.reports = data.items || [];
        this.selectedReport = this.selectedReport ? this.reports.find(r => r.id === this.selectedReport.id) || this.reports[0] || null : this.reports[0] || null;
        this.reportsLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load reports:', err);
        this.reports = [];
        this.selectedReport = null;
        this.reportsLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  loadUsers() {
    this.usersLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get<any>(`${API_BASE_URL}/admin/analytics/users-stats`, { headers }).subscribe({
      next: (data) => {
        this.usersStats = data.items;
        this.usersLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load users:', err);
        this.usersLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  setActiveSection(section: 'dashboard' | 'explorer' | 'performance' | 'reports' | 'users' | 'batch') {
    this.activeSection = section;
    
    if (section === 'dashboard' && !this.dashboardData) {
      this.loadDashboard();
    } else if (section === 'explorer' && this.allPredictions.length === 0) {
      this.loadExplorer();
    } else if (section === 'performance' && !this.performanceData) {
      this.loadPerformance();
    } else if (section === 'reports' && this.reports.length === 0) {
      this.loadReports();
    } else if (section === 'users' && this.usersStats.length === 0) {
      this.loadUsers();
    } else if (section === 'batch') {
      this.loadCuratedDatasetStats();
      this.loadTrainingStatus();
    }
  }

  filterReports() {
    this.loadReports();
  }

  filterPredictions() {
    this.loadExplorer();
  }

  selectPrediction(pred: any) {
    this.revokeSelectedPredictionAudioUrl();
    this.selectedPrediction = pred;
    this.curationEmotion = pred.emotion || 'Happy';
    this.curationResult = null;
    this.curationError = null;
    this.selectedPredictionAudioLoading = false;
    this.selectedPredictionAudioError = null;
    if (pred.audio_id) {
      this.loadSelectedPredictionAudio();
    }
  }

  loadCuratedDatasetStats() {
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get<any>(`${API_BASE_URL}/admin/curated-dataset`, { headers }).subscribe({
      next: (data) => {
        this.curatedDatasetStats = data;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load curated dataset stats:', err);
      }
    });
  }

  loadSelectedPredictionAudio() {
    if (!this.selectedPrediction || !this.selectedPrediction.audio_id) {
      this.selectedPredictionAudioUrl = null;
      this.selectedPredictionAudioError = this.translate.instant('ADMIN.NO_AUDIO_ATTACHED');
      return;
    }

    this.selectedPredictionAudioLoading = true;
    this.selectedPredictionAudioError = null;
    this.revokeSelectedPredictionAudioUrl();

    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get(`${API_BASE_URL}/audio/${this.selectedPrediction.audio_id}`, { headers, responseType: 'blob' }).subscribe({
      next: (blob) => {
        this.selectedPredictionAudioUrl = window.URL.createObjectURL(blob);
        this.selectedPredictionAudioLoading = false;
        this.isPredictionAudioPlaying = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load prediction audio:', err);
        this.selectedPredictionAudioUrl = null;
        this.selectedPredictionAudioLoading = false;
        this.selectedPredictionAudioError = this.translate.instant('ADMIN.AUDIO_LOAD_ERROR');
        this.cdr.detectChanges();
      }
    });
  }

  togglePlayPredictionAudio() {
    const el = document.getElementById('prediction-audio') as HTMLAudioElement | null;
    if (!el) return;
    if (el.paused) {
      el.play().catch(err => console.error('Play failed', err));
    } else {
      el.pause();
    }
  }

  onPredictionAudioPlay() {
    this.isPredictionAudioPlaying = true;
  }

  onPredictionAudioPause() {
    this.isPredictionAudioPlaying = false;
  }

  curateSelectedPrediction() {
    if (!this.selectedPrediction || !this.selectedPrediction.audio_id || !this.curationEmotion) return;

    this.actionLoading = true;
    this.curationError = null;
    this.curationResult = null;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.post<any>(`${API_BASE_URL}/admin/predictions/${this.selectedPrediction.id}/curate`, {
      corrected_emotion: this.curationEmotion
    }, { headers }).subscribe({
      next: (data) => {
        this.curationResult = data;
        this.showToast(this.translate.instant('ADMIN.TOAST_SAVED_TO_FOLDER', { emotion: data.emotion }));
        this.loadCuratedDatasetStats();
        this.actionLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to curate prediction:', err);
        this.curationError = err?.error?.detail || this.translate.instant('ADMIN.TOAST_CANNOT_SAVE');
        this.showToast(this.translate.instant('ADMIN.TOAST_CANNOT_SAVE'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  selectReport(report: any) {
    this.revokeSelectedReportAudioUrl();
    this.selectedReport = report;
    this.selectedReportAudioLoading = false;
    this.selectedReportAudioError = null;
    this.reportReview = {
      status: report.status === 'PENDING' ? 'CONFIRMED' : report.status,
      admin_review_note: report.admin_review_note || ''
    };
    this.correctedEmotion = report.corrected_emotion || report.prediction_emotion || 'Happy';
    this.learningResult = null;
    this.learningError = null;
    this.curationResult = null;
    this.retestResult = null;
    this.loadSelectedReportAudio();
  }

  loadSelectedReportAudio() {
    if (!this.selectedReport || !this.selectedReport.audio_id) {
      this.selectedReportAudioUrl = null;
      this.selectedReportAudioError = this.translate.instant('ADMIN.NO_AUDIO_ATTACHED');
      return;
    }

    this.selectedReportAudioLoading = true;
    this.selectedReportAudioError = null;
    this.revokeSelectedReportAudioUrl();

    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get(`${API_BASE_URL}/audio/${this.selectedReport.audio_id}`, { headers, responseType: 'blob' }).subscribe({
      next: (blob) => {
        this.selectedReportAudioUrl = window.URL.createObjectURL(blob);
        this.selectedReportAudioLoading = false;
        this.showToast(this.translate.instant('ADMIN.TOAST_AUDIO_LOADED'));
        // ensure playing flag false until user plays
        this.isAudioPlaying = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load reported audio:', err);
        this.selectedReportAudioUrl = null;
        this.selectedReportAudioLoading = false;
        this.selectedReportAudioError = this.translate.instant('ADMIN.AUDIO_LOAD_ERROR');
        this.showToast(this.translate.instant('ADMIN.TOAST_AUDIO_LOAD_FAILED'));
        this.cdr.detectChanges();
      }
    });
  }

  togglePlayAudio() {
    const el = document.getElementById('report-audio') as HTMLAudioElement | null;
    if (!el) return;
    if (el.paused) {
      el.play().catch(err => console.error('Play failed', err));
    } else {
      el.pause();
    }
  }

  onAudioPlay() {
    this.isAudioPlaying = true;
  }

  onAudioPause() {
    this.isAudioPlaying = false;
  }

  reviewReport(status: string) {
    // Open confirmation modal for review actions
    if (!this.selectedReport) return;
    this.confirmAction = { type: 'REVIEW', payload: { status } };
    this.showConfirmModal = true;
  }

  // Called after admin confirms action in modal
  confirmReviewAction() {
    if (!this.selectedReport || this.confirmAction.type !== 'REVIEW') return;
    const status = this.confirmAction.payload.status;
    this.showConfirmModal = false;
    this.actionLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.patch<any>(`${API_BASE_URL}/admin/reports/${this.selectedReport.id}`, {
      status,
      admin_review_note: this.reportReview.admin_review_note || null
    }, { headers }).subscribe({
      next: (updated) => {
        this.selectedReport = updated;
        this.reports = this.reports.map(report => report.id === updated.id ? updated : report);
        this.loadReports();
        this.showToast(this.translate.instant('ADMIN.TOAST_REPORT_UPDATED'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to review report:', err);
        this.showToast(this.translate.instant('ADMIN.TOAST_REPORT_UPDATE_FAILED'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  retestReport() {
    if (!this.selectedReport) return;
    // Confirm retest
    this.confirmAction = { type: 'RETEST' };
    this.showConfirmModal = true;
  }

  // perform retest after confirmation
  confirmRetest() {
    if (!this.selectedReport) return;
    this.showConfirmModal = false;
    this.actionLoading = true;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.post<any>(`${API_BASE_URL}/admin/reports/${this.selectedReport.id}/retest`, {}, { headers }).subscribe({
      next: (data) => {
        this.retestResult = data;
        this.showToast(this.translate.instant('ADMIN.TOAST_RETEST_DONE'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to retest report:', err);
        this.showToast(this.translate.instant('ADMIN.TOAST_RETEST_FAILED'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  learnFromReport() {
    if (!this.selectedReport || !this.selectedReport.audio_id || !this.correctedEmotion) return;

    this.actionLoading = true;
    this.learningError = null;
    this.learningResult = null;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.post<any>(`${API_BASE_URL}/admin/reports/${this.selectedReport.id}/learn`, {
      corrected_emotion: this.correctedEmotion,
      epochs: 80,
      learning_rate: 0.001
    }, { headers }).subscribe({
      next: (data) => {
        this.learningResult = data;
        this.selectedReport = {
          ...this.selectedReport,
          status: 'CONFIRMED',
          corrected_emotion: data.corrected_emotion,
          learning_status: data.training_status,
          learned_at: data.learned_at
        };
        this.reports = this.reports.map(report => report.id === this.selectedReport.id ? this.selectedReport : report);
        this.showToast(data.training_status === 'LEARNED'
          ? this.translate.instant('ADMIN.TOAST_MODEL_LEARNED')
          : this.translate.instant('ADMIN.TOAST_CORRECTION_SAVED'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to teach model:', err);
        this.learningError = err?.error?.detail || this.translate.instant('ADMIN.LEARNING_ERROR');
        this.showToast(this.translate.instant('ADMIN.TOAST_LEARNING_FAILED'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  curateSelectedReport() {
    if (!this.selectedReport || !this.selectedReport.audio_id || !this.correctedEmotion) return;

    this.actionLoading = true;
    this.learningError = null;
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.post<any>(`${API_BASE_URL}/admin/reports/${this.selectedReport.id}/curate`, {
      corrected_emotion: this.correctedEmotion
    }, { headers }).subscribe({
      next: (data) => {
        this.selectedReport = {
          ...this.selectedReport,
          status: 'CONFIRMED',
          corrected_emotion: data.emotion,
          learning_status: 'CURATED'
        };
        this.reports = this.reports.map(report => report.id === this.selectedReport.id ? this.selectedReport : report);
        this.curationResult = data;
        this.showToast(this.translate.instant('ADMIN.TOAST_SAVED_TO_FOLDER', { emotion: data.emotion }));
        this.loadCuratedDatasetStats();
        this.actionLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to curate report:', err);
        this.learningError = err?.error?.detail || this.translate.instant('ADMIN.TOAST_CANNOT_SAVE');
        this.showToast(this.translate.instant('ADMIN.TOAST_CANNOT_SAVE'));
        this.actionLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  loadTrainingStatus() {
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.get<any>(`${API_BASE_URL}/admin/training/status`, { headers }).subscribe({
      next: (data) => {
        this.trainingStatus = data;
        this.trainingError = null;

        // Parse log if available
        if (data.log_tail) {
          this.parseTrainingLog(data.log_tail);
        } else {
          this.resetTrainingSteps();
        }

        if (data.running) {
          this.showTrainingLog = true;
          this.startTrainingPolling();
        } else {
          this.stopTrainingPolling();
          // Reset or refresh stats if transition from running to complete
          if (this.activeSection === 'batch') {
            this.loadCuratedDatasetStats();
          }
        }
        this.cdr.detectChanges();
        this.autoScrollLog();
      },
      error: (err) => {
        console.error('Failed to load training status:', err);
        this.trainingError = err?.error?.detail || 'Could not load training status.';
        this.cdr.detectChanges();
      }
    });
  }

  startTraining() {
    this.trainingLoading = true;
    this.trainingError = null;
    this.resetTrainingSteps();
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    this.http.post<any>(`${API_BASE_URL}/admin/training/start`, {}, { headers }).subscribe({
      next: (data) => {
        this.trainingStatus = data;
        this.trainingLoading = false;
        this.showToast('Training started');
        this.startTrainingPolling();
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to start training:', err);
        this.trainingError = err?.error?.detail || 'Could not start training.';
        this.trainingLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  private startTrainingPolling() {
    // Poll faster (every 2 seconds) during training for real-time console logs
    if (this.trainingPollId !== null) {
      window.clearInterval(this.trainingPollId);
    }
    this.trainingPollId = window.setInterval(() => this.loadTrainingStatus(), 2000);
  }

  private stopTrainingPolling() {
    if (this.trainingPollId !== null) {
      window.clearInterval(this.trainingPollId);
      this.trainingPollId = null;
    }
  }

  parseTrainingLog(log: string) {
    if (!log) {
      this.resetTrainingSteps();
      return;
    }

    // Determine training step
    if (log.includes('=== Running:') && log.includes('prepare_data.py')) {
      this.trainingStep = 'preparing';
    }
    if (log.includes('=== Running:') && log.includes('train_model_gpu.py')) {
      this.trainingStep = 'training';
    }
    if (log.includes('Reloading model') || log.includes('EXTRACTION TERMINEE') && !log.includes('=== Running: train_model_gpu.py')) {
      this.trainingStep = 'reloading';
    }

    if (this.trainingStatus) {
      if (!this.trainingStatus.running) {
        if (this.trainingStatus.status === 'COMPLETED') {
          this.trainingStep = 'completed';
        } else if (this.trainingStatus.status === 'FAILED') {
          this.trainingStep = 'failed';
        } else {
          this.trainingStep = 'idle';
        }
      }
    }

    // Extract epoch stats using regex: Epoch   3/100 -- loss: 0.1234, acc: 0.5678  |  val_loss: 0.2345, val_acc: 0.6789
    const epochRegex = /Epoch\s+(\d+)\s*\/\s*(\d+)\s*--\s*loss:\s*([\d.]+),\s*acc:\s*([\d.]+)\s*\|\s*val_loss:\s*([\d.]+),\s*val_acc:\s*([\d.]+)/g;
    let match;
    const epochs: any[] = [];

    while ((match = epochRegex.exec(log)) !== null) {
      const epochNum = parseInt(match[1], 10);
      const totalEp = parseInt(match[2], 10);
      const loss = parseFloat(match[3]);
      const acc = parseFloat(match[4]);
      const valLoss = parseFloat(match[5]);
      const valAcc = parseFloat(match[6]);

      epochs.push({
        epoch: epochNum,
        loss: loss,
        acc: acc,
        valLoss: valLoss,
        valAcc: valAcc
      });

      this.maxEpochs = totalEp;
    }

    if (epochs.length > 0) {
      this.parsedEpochs = epochs;
      const latest = epochs[epochs.length - 1];
      this.currentEpoch = latest.epoch;
      this.currentLoss = latest.loss;
      this.currentAcc = latest.acc;
      this.currentValLoss = latest.valLoss;
      this.currentValAcc = latest.valAcc;

      // Generate SVG path coordinate strings
      this.generateSvgPaths(epochs);
    } else {
      this.clearSvgPaths();
    }
  }

  resetTrainingSteps() {
    this.trainingStep = 'idle';
    this.parsedEpochs = [];
    this.currentEpoch = 0;
    this.currentLoss = 0;
    this.currentAcc = 0;
    this.currentValLoss = 0;
    this.currentValAcc = 0;
    this.clearSvgPaths();
  }

  clearSvgPaths() {
    this.realLossPath = null;
    this.realAccPath = null;
    this.realLossAreaPath = null;
    this.realAccAreaPath = null;
    this.realValLossPath = null;
    this.realValAccPath = null;
  }

  generateSvgPaths(epochs: any[]) {
    if (!epochs || epochs.length === 0) {
      this.clearSvgPaths();
      return;
    }

    const startX = 40;
    const width = 740;
    const startY = 270;
    const height = 240;
    const totalEp = Math.max(this.maxEpochs, epochs.length);

    const accPoints: string[] = [];
    const lossPoints: string[] = [];
    const valAccPoints: string[] = [];
    const valLossPoints: string[] = [];

    // Scale loss based on max loss value in dataset to prevent clipping
    const maxLossVal = Math.max(...epochs.map(e => e.loss), ...epochs.map(e => e.valLoss), 1.0);
    const lossScaleMax = maxLossVal * 1.15;

    epochs.forEach((e) => {
      // Map epoch number to X coord (0 to totalEp-1)
      const x = startX + ((e.epoch - 1) / (totalEp - 1 || 1)) * width;

      // Accuracy y coord (0.0 to 1.0)
      const yAcc = startY - e.acc * height;
      const yValAcc = startY - e.valAcc * height;

      // Loss y coord (scaled)
      const yLoss = startY - (e.loss / lossScaleMax) * height;
      const yValLoss = startY - (e.valLoss / lossScaleMax) * height;

      accPoints.push(`${x.toFixed(1)} ${yAcc.toFixed(1)}`);
      valAccPoints.push(`${x.toFixed(1)} ${yValAcc.toFixed(1)}`);
      lossPoints.push(`${x.toFixed(1)} ${yLoss.toFixed(1)}`);
      valLossPoints.push(`${x.toFixed(1)} ${yValLoss.toFixed(1)}`);
    });

    this.realAccPath = `M ${accPoints.join(' L ')}`;
    this.realValAccPath = `M ${valAccPoints.join(' L ')}`;
    this.realLossPath = `M ${lossPoints.join(' L ')}`;
    this.realValLossPath = `M ${valLossPoints.join(' L ')}`;

    // Closed areas for gradient backgrounds
    const lastX = startX + ((epochs[epochs.length - 1].epoch - 1) / (totalEp - 1 || 1)) * width;
    this.realAccAreaPath = `${this.realAccPath} L ${lastX.toFixed(1)} ${startY} L ${startX} ${startY} Z`;
    this.realLossAreaPath = `${this.realLossPath} L ${lastX.toFixed(1)} ${startY} L ${startX} ${startY} Z`;
  }

  autoScrollLog() {
    setTimeout(() => {
      const container = document.querySelector('.training-log pre');
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    }, 50);
  }

  // Toast helpers
  showToast(msg: string, timeout = 3500) {
    this.toastMessage = msg;
    this.toastVisible = true;
    setTimeout(() => {
      this.toastVisible = false;
    }, timeout);
  }

  private revokeSelectedReportAudioUrl() {
    if (this.selectedReportAudioUrl) {
      window.URL.revokeObjectURL(this.selectedReportAudioUrl);
      this.selectedReportAudioUrl = null;
    }
  }

  private revokeSelectedPredictionAudioUrl() {
    if (this.selectedPredictionAudioUrl) {
      window.URL.revokeObjectURL(this.selectedPredictionAudioUrl);
      this.selectedPredictionAudioUrl = null;
    }
  }

  exportData() {
    const csv = [
      ['Emotion', 'Confidence', 'User', 'Date'],
      ...this.filteredPredictions.map(p => [
        p.emotion,
        p.confidence,
        p.user_email,
        new Date(p.created_at).toLocaleString()
      ])
    ];
    
    const csvContent = csv.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  getEmotionColor(emotion: string): string {
    const colors: { [key: string]: string } = {
      'Neutral': '#607D8B',
      'Calm': '#9C27B0',
      'Happy': '#FFC107',
      'Sad': '#2196F3',
      'Angry': '#F44336',
      'Fearful': '#673AB7',
      'Disgust': '#4CAF50',
      'Surprised': '#FF6F00'
    };
    return colors[emotion] || '#999';
  }

  getEmotionEmoji(emotion: string): string {
    const emojis: { [key: string]: string } = {
      'Neutral': '😐',
      'Calm': '😌',
      'Happy': '😊',
      'Sad': '😢',
      'Angry': '😠',
      'Fearful': '😨',
      'Disgust': '🤢',
      'Surprised': '😲'
    };
    return emojis[emotion] || '🎙️';
  }

  getConfidenceClass(confidence: number): string {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
  }

  // --- USER CRUD METHODS ---

  openAddUserModal() {
    this.editingUserId = null;
    this.userForm = { email: '', password: '', role: 'USER', is_active: true };
    this.showUserModal = true;
  }

  openEditUserModal(user: any) {
    this.editingUserId = user.id;
    this.userForm = { 
      email: user.email, 
      password: '', // Leave blank, only send if changed
      role: user.role || 'USER', 
      is_active: user.is_active !== false 
    };
    this.showUserModal = true;
  }

  closeUserModal() {
    this.showUserModal = false;
  }

  saveUser() {
    const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
    const payload: any = { ...this.userForm };
    if (this.editingUserId) {
      if (!payload.password) {
        delete payload.password; // don't send empty password
      }
      this.http.patch<any>(`${API_BASE_URL}/admin/users/${this.editingUserId}`, payload, { headers }).subscribe({
        next: () => {
          this.closeUserModal();
          this.loadUsers();
        },
        error: (err) => console.error('Failed to update user', err)
      });
    } else {
      this.http.post<any>(`${API_BASE_URL}/admin/users`, payload, { headers }).subscribe({
        next: () => {
          this.closeUserModal();
          this.loadUsers();
        },
        error: (err) => console.error('Failed to create user', err)
      });
    }
  }

  deleteUser(id: number) {
    if (confirm('Are you sure you want to delete this user?')) {
      const headers = { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` };
      this.http.delete<any>(`${API_BASE_URL}/admin/users/${id}`, { headers }).subscribe({
        next: () => this.loadUsers(),
        error: (err) => console.error('Failed to delete user', err)
      });
    }
  }
}
