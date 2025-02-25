import cv2 as cv
import numpy as np

def calibrate_camera(images: list[str]):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    
    for fname in images:
        img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
        ret, corners = cv.findChessboardCorners(img, (8, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        cv.waitKey(500)
        
    cv.destroyAllWindows()
    _, mtx, _, _, _ = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    return mtx

if __name__ == "__main__":
    from glob import glob
    
    images = glob("../images/chess/*.jpg")
    K = calibrate_camera(images)
    print("Calibration matrix:\n", K)