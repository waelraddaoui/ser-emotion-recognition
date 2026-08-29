import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from './auth.service';
import { TranslateService, TranslatePipe, TranslateDirective } from '@ngx-translate/core';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, TranslatePipe, TranslateDirective],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  isDarkMode = true;
  currentLanguage = 'fr';

  constructor(
    private auth: AuthService, 
    private router: Router,
    private translate: TranslateService
  ) {
    this.initTheme();
    this.initLanguage();
  }

  initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
      this.isDarkMode = false;
      document.documentElement.classList.add('light-theme');
    }
  }

  initLanguage() {
    const savedLang = localStorage.getItem('language') || 'fr';
    this.currentLanguage = savedLang;
    this.translate.setFallbackLang('fr');
    this.translate.use(savedLang);
  }

  toggleTheme() {
    this.isDarkMode = !this.isDarkMode;
    if (this.isDarkMode) {
      document.documentElement.classList.remove('light-theme');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.add('light-theme');
      localStorage.setItem('theme', 'light');
    }
  }

  toggleLanguage() {
    this.currentLanguage = this.currentLanguage === 'fr' ? 'en' : 'fr';
    this.translate.use(this.currentLanguage);
    localStorage.setItem('language', this.currentLanguage);
  }

  get isLoggedIn(): boolean {
    return this.auth.isLoggedIn();
  }

  get isAdmin(): boolean {
    return this.auth.isAdmin();
  }

  get userEmail(): string | null {
    return this.auth.getUser()?.email || null;
  }

  logout() {
    const refresh = this.auth.getRefreshToken();
    if (!refresh) {
      this.auth.clearSession();
      this.router.navigate(['/login']);
      return;
    }
    this.auth.logout().subscribe({
      next: () => {
        this.router.navigate(['/login']);
      },
      error: () => {
        this.auth.clearSession();
        this.router.navigate(['/login']);
      }
    });
  }
}
