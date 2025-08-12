import cv2
import numpy as np
import glob

# Checkerboard dimensions (number of inner corners, e.g., 9x6)
CHECKERBOARD = (9, 6)
square_size = 0.025  # Square size in meters (e.g., 2.5cm)

# Prepare 3D points for the checkerboard pattern (z=0 plane)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Path to calibration images (e.g., 'calib_images/*.jpg')
images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
if not imgpoints:
    print("No checkerboard corners found in any image. Calibration aborted.")
    exit()
cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Save calibration results
np.savez('calib_params.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("Calibration complete. Parameters saved to 'calib_params.npz'.")