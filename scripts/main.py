from calibration import calibrate_camera
from features import detect_features, match_features
from geometry import get_matrices
from glob import glob
from pickle import dump

chessboard_images = glob("../images/chess/*.jpg")
object_images = glob("../images/object/*.jpg")
num = len(object_images)

K = calibrate_camera(chessboard_images)
with open("../assets/calibration_matrix.pkl", "wb") as file:
    dump(K, file)

keypoints = dict()
for i, fname in enumerate(object_images):
    kp, desc = detect_features(fname)
    keypoints[f"img{i + 1}"] = {"points": kp,
                                "descriptor": desc}
with open("../assets/keypoints.pkl", "wb") as file:
    dump(keypoints, file)

matches = dict()
for i in range(1, num):
    for j in range(i + 1, num + 1):
        desc1 = keypoints[f"img{i}"]["descriptor"]
        desc2 = keypoints[f"img{j}"]["descriptor"]
        match = match_features(desc1, desc2)
        matches[f"{i}_{j}"] = match
with open("../assets/matches.pkl", "wb") as file:
    dump(matches, file)

matrices = dict()
for i in range(1, num):
    for j in range(i + 1, num + 1):
        kp1 = keypoints[f"img{i}"]["points"]
        kp2 = keypoints[f"img{j}"]["points"]
        F, E, R, t = get_matrices(kp1, kp2, K)
        matrices[f"{i}_{j}"] = {"Fundamental": F,
                                "Essential": E,
                                "Rotation": R,
                                "Translation": t}
with open("../assets/matrices.pkl", "wb") as file:
    dump(matrices, file)