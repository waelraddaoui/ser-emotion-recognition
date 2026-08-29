import { Component } from '@angular/core';
import { AdminDashboardComponent } from './admin-dashboard.component';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [AdminDashboardComponent],
  template: '<app-admin-dashboard></app-admin-dashboard>',
  styles: []
})
export class AdminComponent {}
