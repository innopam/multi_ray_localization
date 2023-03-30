import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from geometry import find_closest_points
# from visualization import visualize_points_with_plotly
from reconstruction import reconstruction_from_json
from numpy import VisibleDeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

# def icp_objective(x, lines):
#     distances = np.sqrt(np.sum((lines - x)**2, axis=2))
#     return np.sum(distances**2)

def icp_objective(x, lines):
    distances = np.abs(np.linalg.norm(np.cross(lines[:, 1] - lines[:, 0], lines[:, 0] - x), axis=1)
                       / np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1))
    return np.sum(distances ** 2)
                       

def save_to_obj(vertices, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            vertex_str = ' '.join([f'{x:.6f}' for x in v])
            f.write(f"v {vertex_str}\n")

if __name__ == "__main__":
    image_data = [
        ('DJI_0010.JPG', (2100, 1332)),
        ('DJI_0029.JPG', (1881, 647)),
        ('DJI_0095.JPG', (2062, 538))
    ]

    # image_data = [
    #     ('DJI_0010.JPG', (2100, 1332)),
    #     ('DJI_0029.JPG', (1881, 647)),
    #     ('DJI_0095.JPG', (2062, 538)),
    #     ('DJI_0120.JPG', (2158, 1256)),
    #     ('DJI_0879.JPG', (2150, 806)),
    #     ('DJI_0886.JPG', (1240, 1448)),
    #     ('DJI_0898.JPG', (1544, 1336)),
    #     ('DJI_0942.JPG', (1728, 923)),
    #     ('DJI_0946.JPG', (1712, 945)),
    #     ('DJI_0991.JPG', (2492, 419)),
    # ]

    scale = 10

    reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
    rec = reconstructions[0]

    cam_positions = []
    rel_positions = []

    for image_name, pixel_coordinates in image_data:
        pose = rec.shots[image_name].pose
        camera = rec.shots[image_name].camera

        rel_pix = camera.pixel_to_normalized_coordinates(pixel_coordinates)
        bearing = camera.pixel_bearing(rel_pix)
        bearing /= bearing[2]

        cam_pos = pose.transform_inverse((0, 0, 0))
        rel_pos = pose.transform_inverse(bearing * scale)

        cam_positions.append(cam_pos)
        rel_positions.append(rel_pos)

    lines = np.array(list(zip(cam_positions, rel_positions)))

    x0 = np.mean(lines, axis=(0, 1))
    print(x0)
    result = minimize(icp_objective, x0, args=(lines,), method='Powell')
    print(result)
    intersection = result.x
    print("Intersection:", intersection)
    x, y, z = intersection[0], intersection[1], intersection[2]
    print(f"Intersection: x={x:.8f}, y={y:.8f}, z={z:.8f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], line[:, 2])
    ax.scatter(intersection[0], intersection[1], intersection[2], c='r', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('intersection_plot.png')
    # plt.show()

    save_to_obj([intersection], "/code/save_obj/average_intersection2.obj")