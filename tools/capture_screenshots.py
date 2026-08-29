"""
Capture UI screenshots for the PFE report using Playwright.
Requires: frontend on :4200, backend on :8001
"""
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(BASE, "figures", "screenshots")
FRONTEND = "http://localhost:4200"
API = "http://localhost:8001"

ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123"


def wait_server(url: str, timeout: int = 90) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def capture():
    from playwright.sync_api import sync_playwright

    os.makedirs(SHOTS, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            device_scale_factor=1.5,
        )
        page = context.new_page()

        def shot(name: str, full_page: bool = True):
            path = os.path.join(SHOTS, f"{name}.png")
            page.screenshot(path=path, full_page=full_page)
            print(f"  -> {path}")

        # Login page
        page.goto(f"{FRONTEND}/login", wait_until="networkidle", timeout=60000)
        time.sleep(1)
        shot("01_login")

        # Login as admin
        page.fill('input[type="email"]', ADMIN_EMAIL)
        page.fill('input[type="password"]', ADMIN_PASSWORD)
        page.click('button[type="submit"]')
        page.wait_for_url("**/studio**", timeout=30000)
        time.sleep(2)
        shot("02_studio")

        # Profile / history
        page.goto(f"{FRONTEND}/profile", wait_until="networkidle")
        time.sleep(2)
        shot("03_profile")

        # Admin
        page.goto(f"{FRONTEND}/admin", wait_until="networkidle")
        time.sleep(2)
        try:
            buttons = page.locator("button.nav-btn")

            users_btn = buttons.filter(has_text="Users")
            if users_btn.count() > 0:
                users_btn.first.click()
                time.sleep(1.5)
                shot("04_admin_users")

            explorer_btn = buttons.filter(has_text="Explorer")
            if explorer_btn.count() > 0:
                explorer_btn.first.click()
                time.sleep(1.5)
                shot("05_admin_predictions")
        except Exception:
            pass

        # Admin reports tab and detail view
        try:
            buttons = page.locator("button.nav-btn")
            reports_btn = buttons.filter(has_text="Reports")
            if reports_btn.count() > 0:
                reports_btn.first.click()
                time.sleep(2)
                shot("07_admin_reports")

                try:
                    page.wait_for_selector(".report-card", timeout=10000)
                except Exception:
                    pass

                report_cards = page.locator(".report-card")
                if report_cards.count() > 0:
                    report_cards.first.click()
                    time.sleep(2)
                    shot("08_admin_report_detail")
        except Exception as e:
            print(f"  admin reports capture: {e}")

        # Studio — example tab if visible
        page.goto(f"{FRONTEND}/studio", wait_until="networkidle")
        time.sleep(1)
        try:
            page.locator("button.tab-btn").filter(has_text="RAVDESS").first.click()
            time.sleep(1)
            examples = page.locator("button").filter(has_text="😊")
            if examples.count() == 0:
                examples = page.locator(".examples-grid button")
            if examples.count() > 0:
                examples.first.click()
                time.sleep(0.5)
                analyze = page.locator("button").filter(has_text="analyse")
                if analyze.count() == 0:
                    analyze = page.locator("button.btn-primary")
                if analyze.count() > 0:
                    analyze.first.click()
                    time.sleep(8)
                    shot("06_studio_result")
        except Exception as e:
            print(f"  studio result capture: {e}")

        browser.close()


def main():
    print("Waiting for frontend...")
    if not wait_server(FRONTEND):
        print("ERROR: Frontend not running on port 4200. Start with: cd frontend && npm start")
        sys.exit(1)
    if not wait_server(f"{API}/"):
        print("WARNING: Backend not on 8001 — some pages may fail")

    print("Capturing screenshots...")
    capture()
    print("Screenshots done.")


if __name__ == "__main__":
    main()
