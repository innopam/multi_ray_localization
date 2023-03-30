from visualization import visualize_points_with_plotly
from reconstruction import reconstruction_from_json
from numpy import VisibleDeprecationWarning
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

def read_multiple_image_data_from_txt(filename):
    image_data_groups = []
    with open(filename, 'r') as f:
        current_group = []
        for line in f:
            line = line.strip()
            if not line:
                if current_group:
                    image_data_groups.append(current_group)
                    current_group = []
                continue
            parts = line.split()
            image_name = parts[0]
            pixel_coordinates = tuple(map(int, parts[1:]))
            current_group.append((image_name, pixel_coordinates))
        if current_group:
            image_data_groups.append(current_group)
    return image_data_groups

def save_to_obj(vertices, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            vertex_str = ' '.join([f'{x:.6f}' for x in v])
            f.write(f"v {vertex_str}\n")

def save_all_intersections_to_obj(intersections, filename):
    with open(filename, 'w') as f:
        for v in intersections:
            vertex_str = ' '.join([f'{x:.6f}' for x in v])
            f.write(f"v {vertex_str}\n")
            
# Define the objective function for ICP
def icp_objective(lines):
    x0 = np.mean(lines[:, 0], axis=0)
    res = minimize(lambda x: np.sum(np.abs(np.linalg.norm(np.cross(lines[:, 1] - lines[:, 0], lines[:, 0] - x), axis=1) 
                                           / np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)) ** 2), x0)

    return res.x

def find_optimal_intersection_point(cam_positions, rel_positions):
    def distance_to_segments(x, cam_positions, rel_positions):
        total_distance = 0
        for i in range(len(cam_positions)):
            a = cam_positions[i]
            b = rel_positions[i]
            ab = b - a
            t = max(0, min(1, np.dot(x - a, ab) / np.dot(ab, ab)))
            projection = a + t * ab
            total_distance += np.linalg.norm(x - projection)
        return total_distance

    x0 = np.mean(cam_positions, axis=0)
    res = minimize(distance_to_segments, x0, args=(cam_positions, rel_positions), method='BFGS', tol=1e-10, options={'maxiter': 10000, 'gtol': 1e-8})

    return res.x

if __name__ == "__main__":
    image_data_groups = read_multiple_image_data_from_txt('/code/multi_ray_localization/image_data_groups.txt')

    reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
    rec = reconstructions[0]

    all_cam_positions = []
    all_rel_positions = []
    all_image_names = []
    all_intersections = []

    for idx, image_data in enumerate(image_data_groups):
        scale = 10

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

        cam_positions = np.array(cam_positions)
        rel_positions = np.array(rel_positions)

        lines = np.column_stack((cam_positions, rel_positions)).reshape(-1, 2, 3)
        intersection = icp_objective(lines)
        x, y, z = intersection[0], intersection[1], intersection[2]
        print(f"Optimal Intersection {idx}: x={x:.8f}, y={y:.8f}, z={z:.8f}")

        all_cam_positions.extend(cam_positions)
        all_rel_positions.extend(rel_positions)
        all_image_names.extend([image_name for image_name, _ in image_data])
        all_intersections.append(intersection)

        save_all_intersections_to_obj(all_intersections, "/code/save_obj/all_optimal_intersections.obj")
        # save_to_obj([intersection], f"/code/save_obj/optimal_intersection_{idx}.obj")

    visualize_points_with_plotly(np.array(all_cam_positions), np.array(all_rel_positions), all_image_names, all_intersections)
