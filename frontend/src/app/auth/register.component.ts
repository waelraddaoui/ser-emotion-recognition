import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../auth.service';
import { TranslatePipe, TranslateDirective } from '@ngx-translate/core';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink, TranslatePipe, TranslateDirective],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {
  errorMsg: string | null = null;
  isLoading = false;

  form: FormGroup;

  constructor(private fb: FormBuilder, private auth: AuthService, private router: Router) {
    this.form = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
      confirmPassword: ['', [Validators.required, Validators.minLength(6)]]
    });
  }

  submit() {
    this.errorMsg = null;
    const { email, password, confirmPassword } = this.form.getRawValue();
    if (!email || !password || !confirmPassword) {
      this.errorMsg = 'Please check your input.';
      return;
    }
    if (password !== confirmPassword) {
      this.errorMsg = 'Passwords do not match.';
      return;
    }

    this.isLoading = true;
    this.auth.register(email, password).subscribe({
      next: () => {
        this.isLoading = false;
        this.router.navigate(['/studio']);
      },
      error: () => {
        this.isLoading = false;
        this.errorMsg = 'Cannot create account. Email may exist.';
      }
    });
  }
}
