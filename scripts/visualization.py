import open3d as o3d

def draw_model(path: str):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd], zoom=0.008,
                                      front = [-1.8, -0.1, -0.9],
                                      lookat=[-0.3, 0.8, 0.5],
                                      up=[0.9, -1.5, -0.1])

if __name__ == "__main__":
    path = "../assets/reconstruction.ply"
    draw_model(path)