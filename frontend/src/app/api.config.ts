import { Capacitor } from '@capacitor/core';
import { environment } from '../environments/environment';

// When running as a native app, the device can't reach 127.0.0.1 (that's the phone itself).
// Set nativeApiUrl to your PC's local network IP for development,
// or to your production server URL for release.
const NATIVE_API_URL = environment.nativeApiUrl;
const WEB_API_URL = environment.webApiUrl;

export const API_BASE_URL = Capacitor.isNativePlatform() ? NATIVE_API_URL : WEB_API_URL;
