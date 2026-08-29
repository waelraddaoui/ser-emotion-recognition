import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { AuthService } from '../auth.service';
import { TranslatePipe, TranslateDirective } from '@ngx-translate/core';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink, TranslatePipe, TranslateDirective],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  errorMsg: string | null = null;
  isLoading = false;

  form: FormGroup;

  constructor(private fb: FormBuilder, private auth: AuthService, private router: Router) {
    this.form = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]]
    });
  }

  submit() {
    this.errorMsg = null;
    if (this.form.invalid) {
      this.errorMsg = 'Please check your input.';
      return;
    }

    const { email, password } = this.form.getRawValue();
    this.isLoading = true;
    this.auth.login(email!, password!).subscribe({
      next: () => {
        this.isLoading = false;
        this.router.navigate(['/studio']);
      },
      error: (err: HttpErrorResponse) => {
        this.isLoading = false;
        if (err.status === 0) {
          this.errorMsg = 'Server unreachable. Please start the backend.';
          return;
        }
        if (err.status === 401) {
          this.errorMsg = 'Invalid email or password.';
          return;
        }
        this.errorMsg = 'Login failed. Please try again.';
      }
    });
  }
}
