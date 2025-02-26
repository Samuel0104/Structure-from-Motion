from calibration import calibrate_camera
from features import detect_features, match_features
from geometry import get_matrices
from glob import glob
from pickle import dump
# from visualization import draw_model

chessboard_images = glob("../images/chess/*.jpg")
object_images = glob("../images/object/*.jpg")
num = len(object_images)

keypoints = dict()
for i in range(num):
    kp, desc = detect_features(object_images[i], coords_only=True)
    keypoints[f"img{i + 1}"] = {"points": kp,
                                "descriptor": desc}
with open("../assets/keypoints.pkl", "wb") as file:
    dump(keypoints, file)

matches = dict()
for i in range(1, num):
    desc1 = keypoints[f"img{i}"]["descriptor"]
    for j in range(i + 1, num + 1):
        desc2 = keypoints[f"img{j}"]["descriptor"]
        match = match_features(desc1, desc2)
        matches[f"{i}_{j}"] = [(m.distance, m.imgIdx, m.queryIdx, m.trainIdx) for m in match]
        print(f"{i}_{j}")
with open("../assets/matches.pkl", "wb") as file:
    dump(matches, file)

K = calibrate_camera(chessboard_images)
with open("../assets/calibration_matrix.pkl", "wb") as file:
    dump(K, file)
    
matrices = dict()
for i in range(1, num):
    kp1 = keypoints[f"img{i}"]["points"]
    for j in range(i + 1, num + 1):
        kp2 = keypoints[f"img{j}"]["points"]
        F, E, R, t = get_matrices(kp1, kp2, K)
        matrices[f"{i}_{j}"] = {"Fundamental": F,
                                "Essential": E,
                                "Rotation": R,
                                "Translation": t}
        print(f"{i}_{j}")
with open("../assets/matrices.pkl", "wb") as file:
    dump(matrices, file)
    
# path = "../assets/reconstruction/dense/0/fused.ply"
# draw_model(path)