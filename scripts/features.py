import cv2 as cv
import numpy as np

def detect_features(image: str, return_image=False):
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    if return_image:
        return kp, desc, img
    return kp, desc

def match_features(desc1: np.ndarray, desc2: np.ndarray):
    bf = cv.BFMatcher(crossCheck=True)
    matches = bf.match(desc1, desc2)
    return matches

if __name__ == "__main__":
    from glob import glob
    import matplotlib.pyplot as plt
    
    images = glob("../images/object/*.jpg")
    kp1, desc1, img2 = detect_features(images[0], True)
    kp2, desc2, img1 = detect_features(images[1], True)
    matches = match_features(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    image = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                           matchesThickness=10)
    plt.imshow(image)
    plt.show()