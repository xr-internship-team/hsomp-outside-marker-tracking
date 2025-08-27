import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test_report.csv")

# Kararlılık grafiği
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(df["Versiyon"] + " " + df["Filtre"], df["Tz Std Sapma"], color='skyblue')
plt.title("Pozisyon Kararlılığı (Tz Std Sapma)")
plt.ylabel("Std Sapma")
plt.xticks(rotation=45)

# Güvenlik grafiği
plt.subplot(1, 2, 2)
plt.bar(df["Versiyon"] + " " + df["Filtre"], df["DM Ort."], color='salmon')
plt.title("Karar Marjı Ortalaması (DM)")
plt.ylabel("DM Ort.")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

