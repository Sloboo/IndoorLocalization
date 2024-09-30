####calibration code using fisheye function

import cv2
import numpy as np
import glob
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the images directory in Google Drive
#image_dir = '/content/drive/My Drive/images'

# Use glob to find all .jpg images in the specified directory
images = glob.glob('/content/drive/My Drive/final project/new try/*.jpg')

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

for fname in images:
    img = cv2.imread(fname)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")


###code for undistored images
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Replace these lines with your actual calibration output values
DIM = (2592, 1944)  # Example dimensions
K = np.array([[1339.9745918897838, 0.0, 1317.3023813008786],
              [0.0, 1370.9666252290544, 995.0038732251304],
              [0.0, 0.0, 1.0]])  # Example intrinsic matrix
D = np.array([[-0.5434811438886831],
              [5.202993933017546],
              [-15.673150244931035],
              [14.866338930660284]])  # Example distortion coefficients

def undistort(img_path, output_dir):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Save the undistorted image
    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, base_name)
    cv2.imwrite(output_path, undistorted_img)
    print(f"Saved undistorted image: {output_path}")

    # Convert BGR to RGB for matplotlib
    undistorted_img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title('Undistorted Image')
    plt.imshow(undistorted_img_rgb)
    plt.show()

# Example usage
image_dir = '/content/drive/My Drive/final project/images for calib demand'
output_dir = '/content/drive/My Drive/final project/images for calib demand/undistorted_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

images = glob.glob(os.path.join(image_dir, '*.jpg'))

if __name__ == '__main__':
    for img_path in images:
        undistort(img_path, output_dir)
