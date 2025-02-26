from calibration import calibrate_camera
from features import detect_features, match_features
from geometry import get_fundamental_matrix, get_essential_matrix
from glob import glob
from pickle import dump

chessboard_images = glob("../images/chess/*.jpg")
object_images = glob("../images/object/*.jpg")

K = calibrate_camera(chessboard_images)
with open("../assets/calibration_matrix.pkl", "wb") as file:
    dump(K, file)

keypoints = dict()
for i, fname in enumerate(object_images):
    kp, desc = detect_features(fname)
    keypoints[f"img{i + 1}"] = {"kp": kp, "desc": desc}
with open("../assets/keypoints.pkl", "wb") as file:
    dump(keypoints, file)

matches = dict()
for i in range(1, len(object_images)):
    desc1 = keypoints[f"img{i}"]["desc"]
    desc2 = keypoints[f"img{i + 1}"]["desc"]
    match = match_features(desc1, desc2)
    matches[f"{i}_{i + 1}"] = match
with open("../assets/matches.pkl", "wb") as file:
    dump(matches, file)

R, t = get_essential_matrix(kp1, kp2, K)







