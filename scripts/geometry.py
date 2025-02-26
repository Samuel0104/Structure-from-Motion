import cv2 as cv
import numpy as np

def get_matrices(kp1: np.ndarray, kp2: np.ndarray, K: np.ndarray):
    min_points = min(kp1.shape[0], kp2.shape[0])
    kp1 = kp1[:min_points]
    kp2 = kp2[:min_points]

    F, _ = cv.findFundamentalMat(kp1, kp2, cv.RANSAC)
    E, _ = cv.findEssentialMat(kp1, kp2, K, cv.RANSAC)
    _, R, t, _ = cv.recoverPose(E, kp1, kp2, K)
    return F, E, R, t

if __name__ == "__main__":
    from calibration import calibrate_camera
    from features import detect_features
    from glob import glob
    
    chessboard_images = glob("../images/chess/*.jpg")
    K = calibrate_camera(chessboard_images)
    object_images = glob("../images/object/*.jpg")
    kp1, desc1 = detect_features(object_images[0], coords_only=True)
    kp2, desc2 = detect_features(object_images[1], coords_only=True)
    F, E, R, t = get_matrices(kp1, kp2, K)
    print("Fundamental matrix:\n", F)
    print("\nEssential matrix:\n", E)
    print("\nRotation matrix:\n", R)
    print("\nTranslation vector:\n", t)