UDP_IP = "127.0.0.1"
UDP_PORT = 12345

FAMILY = "tag25h9"

FRONT_TAG_ID = 3
EXTRINSICS_FILE = "head_extrinsics.json"

# CSV yolu
CSV_PATH = "logs/apriltag_log.csv"

# ID->boyut (m)
TAG_SIZES = {
    1: 0.0375, 2: 0.05, 3: 0.0643,
    4: 0.09,   5: 0.07, 6: 0.11, 7: 0.08
}

# Kalibrasyon yumuşatma katsayısı
EMA_ALPHA = 0.15
