import cv2 as cv
import numpy as np

def get_fundamental_matrix():
    pass

def get_essential_matrix(kp1: tuple[cv.KeyPoint], kp2: tuple[cv.KeyPoint], K: np.ndarray):
    pts1 = np.float32([kp.pt for kp in kp1])
    pts2 = np.float32([kp.pt for kp in kp2])
    
    min_points = min(pts1.shape[0], pts2.shape[0])
    pts1 = pts1[:min_points]
    pts2 = pts2[:min_points]
    
    E, _ = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC)
    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)
    return R, t

if __name__ == "__main__":
    from calibration import calibrate_camera
    from features import detect_features
    from glob import glob
    
    chessboard_images = glob("../images/chess/*.jpg")
    K = calibrate_camera(chessboard_images)
    object_images = glob("../images/object/*.jpg")
    kp1, desc1 = detect_features(object_images[0])
    kp2, desc2 = detect_features(object_images[1])
    R, t = get_essential_matrix(kp1, kp2, K)
    print("Rotation matrix:\n", R)
    print("Translation vector:\n", t)