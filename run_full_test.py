import subprocess
import time
import os
import pandas as pd
import numpy as np
import csv

# === Script Dosyaları ===
HEADPOSE_SCRIPT = "markerTracking.py"
APRILTAG_SCRIPT = "markerTracking_apriltag.py"
COMPARE_SCRIPT = "compare_versions.py"

# === Log Dosyaları ===
HEADPOSE_LOG = "headpose_log.csv"
APRILTAG_LOG = "apriltag_log.csv"

# === Çalıştırma Süresi (saniye)
RUN_DURATION = 10

def run_script(script_name, duration, log_file):
    print(f"\n▶️ Çalıştırılıyor: {script_name} ({duration} saniye)")
    
    # Önceki log varsa sil
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Eski log silindi: {log_file}")

    # Script'i başlat
    process = subprocess.Popen(["python", script_name])
    time.sleep(duration)
    process.terminate()
    print(f"⏹️ Tamamlandı: {script_name}")

def main():
    print("=== Otomatik Test Başlatılıyor ===")

    run_script(HEADPOSE_SCRIPT, RUN_DURATION, HEADPOSE_LOG)
    run_script(APRILTAG_SCRIPT, RUN_DURATION, APRILTAG_LOG)

    print("\n📝 Test raporu oluşturuluyor...")
    subprocess.run(["python", "generateTestReport.py"])
    

    print("\n📊 Karşılaştırma başlatılıyor...")
    subprocess.run(["python", COMPARE_SCRIPT])

    
    print("\n✅ Tüm testler tamamlandı.")


if __name__ == "__main__":
    main()
