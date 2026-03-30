"""
Personalized Disease Risk Manager — Launcher
=============================================
This launcher:
  1. Checks Python is available
  2. Creates/activates a virtual environment in the project folder
  3. Installs all dependencies from requirements.txt
  4. Trains the FL model if global_model.pth doesn't exist
  5. Starts the Streamlit app
  6. Opens the browser automatically
  7. Shows a console window with status so the user knows what's happening
"""

import sys
import os
import subprocess
import webbrowser
import time
import threading

# ── Resolve the project directory ─────────────────────────────────────────────
# Works both when run as a plain .py and when frozen by PyInstaller
if getattr(sys, "frozen", False):
    # Running as PyInstaller .exe — project files are next to the .exe
    PROJECT_DIR = os.path.dirname(sys.executable)
else:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

VENV_DIR      = os.path.join(PROJECT_DIR, ".venv")
REQUIREMENTS  = os.path.join(PROJECT_DIR, "requirements.txt")
APP_PY        = os.path.join(PROJECT_DIR, "app.py")
MODEL_PTH     = os.path.join(PROJECT_DIR, "global_model.pth")
RUN_SIM_PY    = os.path.join(PROJECT_DIR, "run_simulation.py")
PORT          = 8501


def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def get_python() -> str:
    """Return python executable path — prefer venv, fallback to system."""
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")   # Windows
    if os.path.exists(venv_python):
        return venv_python
    venv_python_unix = os.path.join(VENV_DIR, "bin", "python")      # Mac/Linux
    if os.path.exists(venv_python_unix):
        return venv_python_unix
    return sys.executable  # fallback to system python


def step1_check_python():
    banner("Step 1/4 — Checking Python installation")
    ver = sys.version_info
    print(f"  Python {ver.major}.{ver.minor}.{ver.micro} — OK")
    if ver.major < 3 or (ver.major == 3 and ver.minor < 10):
        print("\n  ❌ ERROR: Python 3.10 or newer is required.")
        print("  Download from: https://www.python.org/downloads/")
        input("\n  Press Enter to exit...")
        sys.exit(1)


def step2_setup_venv():
    banner("Step 2/4 — Setting up virtual environment")
    if not os.path.exists(VENV_DIR):
        print("  Creating virtual environment (.venv) ...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        print("  ✓ Virtual environment created")
    else:
        print("  ✓ Virtual environment already exists")

    python = get_python()
    print(f"\n  Installing / updating dependencies from requirements.txt ...")
    print("  (This may take a few minutes on first run)\n")
    subprocess.run(
        [python, "-m", "pip", "install", "-q", "--upgrade", "pip"],
        check=True,
    )
    result = subprocess.run(
        [python, "-m", "pip", "install", "-q", "-r", REQUIREMENTS],
        cwd=PROJECT_DIR,
    )
    if result.returncode != 0:
        print("\n  ❌ Dependency installation failed.")
        print("  Try running: pip install -r requirements.txt manually.")
        input("\n  Press Enter to exit...")
        sys.exit(1)
    print("\n  ✓ All dependencies installed")


def step3_train_model():
    banner("Step 3/4 — Checking trained model")
    if os.path.exists(MODEL_PTH):
        size_mb = os.path.getsize(MODEL_PTH) / (1024 * 1024)
        print(f"  ✓ global_model.pth found ({size_mb:.1f} MB) — skipping training")
        return

    print("  No trained model found. Running Federated Learning simulation...")
    print("  This trains the AI model across 3 hospital nodes — takes ~2–3 minutes.\n")
    python = get_python()
    result = subprocess.run(
        [python, RUN_SIM_PY],
        cwd=PROJECT_DIR,
    )
    if result.returncode != 0 or not os.path.exists(MODEL_PTH):
        print("\n  ❌ Model training failed.")
        print("  Try running: python run_simulation.py")
        input("\n  Press Enter to exit...")
        sys.exit(1)
    print("\n  ✓ Model trained and saved → global_model.pth")


def open_browser_delayed(url: str, delay: float = 4.0):
    """Open browser after a short delay so Streamlit has time to start."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def step4_launch_app():
    banner("Step 4/4 — Launching the Web App")
    url = f"http://localhost:{PORT}"
    print(f"  Starting Streamlit on {url} ...")
    print("  Your browser will open automatically in a few seconds.")
    print("\n  ⚠  Keep this window open while using the app.")
    print("  ⚠  Close this window to shut down the app.\n")

    python = get_python()
    open_browser_delayed(url, delay=5.0)

    try:
        subprocess.run(
            [
                python, "-m", "streamlit", "run", APP_PY,
                "--server.port", str(PORT),
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false",
            ],
            cwd=PROJECT_DIR,
        )
    except KeyboardInterrupt:
        print("\n\n  App stopped by user. Goodbye!")


def main():
    os.chdir(PROJECT_DIR)
    print("\n  🫀  Personalized Disease Risk Manager")
    print("  Powered by Federated Learning + Differential Privacy")

    step1_check_python()
    step2_setup_venv()
    step3_train_model()
    step4_launch_app()


if __name__ == "__main__":
    main()
