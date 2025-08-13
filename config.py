UDP_IP = "127.0.0.1"
UDP_PORT = 12345

FAMILY = "tag25h9"

FRONT_TAG_ID = 1
EXTRINSICS_FILE = "head_extrinsics.json"

# CSV yolu
CSV_PATH = "logs/apriltag_log.csv"

# ID->boyut (m)
TAG_SIZES = {
    1: 0.0375, 2: 0.0375, 3: 0.0375,
    4: 0.0375, 5: 0.0375, 6: 0.0375,
    7: 0.0375, 8: 0.0375
}

# Kalibrasyon yumuşatma katsayısı
EMA_ALPHA = 0.15
# Bu DM eşiğinin altındaki ölçümleri füzyona sokma
DM_MIN = 20.0
# Marker ağırlıklarında histerezis / EMA: w_t = β*w_{t-1}  (1-β)*w_raw
WEIGHT_BETA = 0.7