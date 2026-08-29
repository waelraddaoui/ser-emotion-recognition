import { Routes } from '@angular/router';
import { SerComponent } from './ser/ser.component';
import { LoginComponent } from './auth/login.component';
import { RegisterComponent } from './auth/register.component';
import { ProfileComponent } from './profile/profile.component';
import { AdminComponent } from './admin/admin.component';
import { authGuard } from './auth.guard';
import { adminGuard } from './admin.guard';

export const routes: Routes = [
	{ path: '', redirectTo: 'studio', pathMatch: 'full' },
	{ path: 'studio', component: SerComponent, canActivate: [authGuard] },
	{ path: 'login', component: LoginComponent },
	{ path: 'register', component: RegisterComponent },
	{ path: 'profile', component: ProfileComponent, canActivate: [authGuard] },
	{ path: 'admin', component: AdminComponent, canActivate: [adminGuard] },
	{ path: '**', redirectTo: 'studio' }
];
