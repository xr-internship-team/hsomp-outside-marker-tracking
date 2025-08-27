import os
import subprocess
import time
import shutil

# === Kullanıcıdan mesafe al
distance = input("📏 Test mesafesini girin (örnek: 1.0m): ").strip()
folder_name = f"distance_{distance}m"
target_dir = os.path.join("tests", folder_name)

# === Klasörü oluştur
os.makedirs(target_dir, exist_ok=True)
print(f"\n📁 Test klasörü oluşturuldu: {target_dir}")

# === Log dosyalarını sil (önceki testten kalan varsa)
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

print(f"\n✅ {distance}m mesafesi için test tamamlandı ve loglar kaydedildi.")

