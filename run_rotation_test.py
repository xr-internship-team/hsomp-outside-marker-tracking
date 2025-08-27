
import os
import subprocess
import time
import shutil

# === Kullanıcıdan dönme açısı al
rotation = input("🔄 Dönme açısını girin (örnek: 30): ").strip()
folder_name = f"rotation_{rotation}deg"
target_dir = os.path.join("tests", folder_name)

# === Klasörü oluştur
os.makedirs(target_dir, exist_ok=True)
print(f"\n📁 Test klasörü oluşturuldu: {target_dir}")

# === Önceki logları sil
for log_file in ["headpose_log.csv", "apriltag_log.csv"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Eski log silindi: {log_file}")

# === HeadPose çalıştır
print("\n▶️ HeadPose testi başlatılıyor...")
process_hp = subprocess.Popen(["python", "markerTracking.py"])
time.sleep(10)
process_hp.terminate()
print("⏹️ HeadPose tamamlandı.")

# === AprilTag çalıştır
print("\n▶️ AprilTag testi başlatılıyor...")
process_april = subprocess.Popen(["python", "markerTracking_apriltag.py"])
time.sleep(10)
process_april.terminate()
print("⏹️ AprilTag tamamlandı.")

# === Logları hedef klasöre taşı
for log_file in ["headpose_log.csv", "apriltag_log.csv"]:
    if os.path.exists(log_file):
        shutil.move(log_file, os.path.join(target_dir, log_file))
        print(f"📦 Log taşındı: {log_file} → {target_dir}")

print(f"\n✅ {rotation}° açısı için test tamamlandı ve loglar kaydedildi.")
