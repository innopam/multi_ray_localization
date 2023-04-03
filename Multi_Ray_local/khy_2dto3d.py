# import numpy as np
# from scipy.spatial import distance_matrix
# from scipy.optimize import minimize
# import matplotlib
# matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # from geometry import find_closest_points
# # from visualization import visualize_points_with_plotly
# from reconstruction import reconstruction_from_json
# from numpy import VisibleDeprecationWarning
# import warnings

# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

# # def icp_objective(x, lines):
# #     distances = np.sqrt(np.sum((lines - x)**2, axis=2))
# #     return np.sum(distances**2)

# def icp_objective(x, lines):
#     distances = np.abs(np.linalg.norm(np.cross(lines[:, 1] - lines[:, 0], lines[:, 0] - x), axis=1)
#                        / np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1))
#     return np.sum(distances ** 2)
                       

# def save_to_obj(vertices, filename):
#     with open(filename, 'w') as f:
#         for v in vertices:
#             vertex_str = ' '.join([f'{x:.6f}' for x in v])
#             f.write(f"v {vertex_str}\n")

# if __name__ == "__main__":
#     image_data = [
#         ('DJI_0010.JPG', (2100, 1332)),
#         ('DJI_0029.JPG', (1881, 647)),
#         ('DJI_0095.JPG', (2062, 538))
#     ]

#     # image_data = [
#     #     ('DJI_0010.JPG', (2100, 1332)),
#     #     ('DJI_0029.JPG', (1881, 647)),
#     #     ('DJI_0095.JPG', (2062, 538)),
#     #     ('DJI_0120.JPG', (2158, 1256)),
#     #     ('DJI_0879.JPG', (2150, 806)),
#     #     ('DJI_0886.JPG', (1240, 1448)),
#     #     ('DJI_0898.JPG', (1544, 1336)),
#     #     ('DJI_0942.JPG', (1728, 923)),
#     #     ('DJI_0946.JPG', (1712, 945)),
#     #     ('DJI_0991.JPG', (2492, 419)),
#     # ]

#     scale = 10

#     reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
#     rec = reconstructions[0]

#     cam_positions = []
#     rel_positions = []

#     for image_name, pixel_coordinates in image_data:
#         pose = rec.shots[image_name].pose
#         camera = rec.shots[image_name].camera

#         rel_pix = camera.pixel_to_normalized_coordinates(pixel_coordinates)
#         bearing = camera.pixel_bearing(rel_pix)
#         bearing /= bearing[2]

#         cam_pos = pose.transform_inverse((0, 0, 0))
#         rel_pos = pose.transform_inverse(bearing * scale)

#         cam_positions.append(cam_pos)
#         rel_positions.append(rel_pos)

#     lines = np.array(list(zip(cam_positions, rel_positions)))

#     x0 = np.mean(lines, axis=(0, 1))
#     print(x0)
#     result = minimize(icp_objective, x0, args=(lines,), method='Powell')
#     print(result)
#     intersection = result.x
#     print("Intersection:", intersection)
#     x, y, z = intersection[0], intersection[1], intersection[2]
#     print(f"Intersection: x={x:.8f}, y={y:.8f}, z={z:.8f}")

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for line in lines:
#         ax.plot(line[:, 0], line[:, 1], line[:, 2])
#     ax.scatter(intersection[0], intersection[1], intersection[2], c='r', s=100)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.savefig('intersection_plot.png')
#     # plt.show()

#     save_to_obj([intersection], "/code/save_obj/average_intersection2.obj")
############################################################################################################
# from reconstruction import reconstruction_from_json
# import numpy as np
# from shapely.geometry import LineString, Point
# from scipy.optimize import minimize
# import matplotlib
# matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from numpy import VisibleDeprecationWarning
# import warnings

# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

# def icp_objective(x, lines):
#     distances = []
#     for line in lines:
#         ls = LineString(line)
#         distances.append(ls.distance(Point(x)))
#     return np.sum(distances)

# def save_to_obj(vertices, filename):
#     with open(filename, 'w') as f:
#         for v in vertices:
#             vertex_str = ' '.join([f'{x:.6f}' for x in v])
#             f.write(f"v {vertex_str}\n")

# if __name__ == "__main__":
#     image_data = [
#         ('DJI_0010.JPG', (2100, 1332)),
#         ('DJI_0029.JPG', (1881, 647)),
#         ('DJI_0095.JPG', (2062, 538)),
#         ('DJI_0010.JPG', (1970, 1788)),
#         ('DJI_0027.JPG', (1384, 1071)),
#         ('DJI_0950.JPG', (2348, 1710))
#     ]

#     scale = 10

#     reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
#     rec = reconstructions[0]

#     cam_positions = []
#     rel_positions = []

#     for image_name, pixel_coordinates in image_data:
#         pose = rec.shots[image_name].pose
#         camera = rec.shots[image_name].camera

#         rel_pix = camera.pixel_to_normalized_coordinates(pixel_coordinates)
#         bearing = camera.pixel_bearing(rel_pix)
#         bearing /= bearing[2]

#         cam_pos = pose.transform_inverse((0, 0, 0))
#         rel_pos = pose.transform_inverse(bearing * scale)

#         cam_positions.append(cam_pos)
#         rel_positions.append(rel_pos)

#     lines = []
#     for cam_pos, rel_pos in zip(cam_positions, rel_positions):
#         line = [(cam_pos[0], cam_pos[1], cam_pos[2]), (rel_pos[0], rel_pos[1], rel_pos[2])]
#         lines.append(line)

#     x0 = np.mean(lines, axis=(0, 1))
#     print(x0)

#     result = minimize(icp_objective, x0, args=(lines,), method='Powell')
#     # print(result)
#     intersection = result.x
#     # print("Intersection:", intersection)
#     x, y, z = intersection[0], intersection[1], intersection[2]
#     # print(f"Intersection: x={x:.8f}, y={y:.8f}, z={z:.8f}")

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for line in lines:
#         ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])
    
#     # create a LineString object from the two lines
#     ls1 = LineString(lines[0])
#     ls2 = LineString(lines[1])
#     # calculate the intersection point
#     intersection = ls1.intersection(ls2)
#     print(intersection)
#     # plot the intersection point
#     ax.scatter(intersection.x, intersection.y, intersection.z, c='r', s=100)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.savefig('intersection_plot.png')
#     # plt.show()

#     save_to_obj([intersection.coords[0]], "/code/save_obj/average_intersection2.obj")

    ############################################################################################################

# from reconstruction import reconstruction_from_json
# import numpy as np
# from shapely.geometry import LineString, Point
# from scipy.optimize import minimize
# import matplotlib
# matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from numpy import VisibleDeprecationWarning
# import warnings
# from itertools import combinations

# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

# def icp_objective(x, lines):
#     distances = []
#     for line in lines:
#         ls = LineString(line)
#         distances.append(ls.distance(Point(x)))
#     return np.sum(distances)

# def save_to_obj(vertices, filename):
#     with open(filename, 'w') as f:
#         for v in vertices:
#             vertex_str = ' '.join([f'{x:.6f}' for x in v])
#             f.write(f"v {vertex_str}\n")

# if __name__ == "__main__":
#     image_data = [
#         ('DJI_0010.JPG', (2100, 1332)),
#         ('DJI_0029.JPG', (1881, 647)),
#         ('DJI_0095.JPG', (2062, 538)),
#         ('DJI_0031.JPG', (2419, 983)),
#         ('DJI_0037.JPG', (2893, 902)),
#         ('DJI_0047.JPG', (2342, 706))
#     ]

#     scale = 10

#     reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
#     rec = reconstructions[0]

#     cam_positions = []
#     rel_positions = []

#     for image_name, pixel_coordinates in image_data:
#         pose = rec.shots[image_name].pose
#         camera = rec.shots[image_name].camera

#         rel_pix = camera.pixel_to_normalized_coordinates(pixel_coordinates)
#         bearing = camera.pixel_bearing(rel_pix)
#         bearing /= bearing[2]

#         cam_pos = pose.transform_inverse((0, 0, 0))
#         rel_pos = pose.transform_inverse(bearing * scale)

#         cam_positions.append(cam_pos)
#         rel_positions.append(rel_pos)

#     lines = []
#     for cam_pos, rel_pos in zip(cam_positions, rel_positions):
#         line = [(cam_pos[0], cam_pos[1], cam_pos[2]), (rel_pos[0], rel_pos[1], rel_pos[2])]
#         lines.append(line)

#     x0 = np.mean(lines, axis=(0, 1))
#     # print(x0)

#     result = minimize(icp_objective, x0, args=(lines,), method='Powell')
#     # print(result)
#     intersection = result.x
#     # print("Intersection:", intersection)
#     x, y, z = intersection[0], intersection[1], intersection[2]
#     # print(f"Intersection: x={x:.8f}, y={y:.8f}, z={z:.8f}")

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for line in lines:
#         ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])

#     threshold = 0.08
#     # calculate all intersection points
#     intersections = []
#     for line1, line2 in combinations(lines, 2):
#         ls1 = LineString(line1)
#         ls2 = LineString(line2)
#         intersection = ls1.intersection(ls2)
#         if intersection.geom_type == 'Point':
#             # check if the point is a common point of the two lines
#             if intersection.equals(Point(line1[0])) or intersection.equals(Point(line1[1])):
#                 continue
#             if intersection.equals(Point(line2[0])) or intersection.equals(Point(line2[1])):
#                 continue
#             intersections.append(intersection)

#     # count the number of lines that intersect at each point
#     intersection_counts = {}
#     for intersection in intersections:
#         intersection_counts[tuple(intersection.coords)] = 0
#         for line in lines:
#             ls = LineString(line)
#             if ls.distance(intersection) < threshold:
#                 intersection_counts[tuple(intersection.coords)] += 1

#     # find the intersection points that intersect with 3 or more lines
#     common_intersections = []
#     for intersection in intersection_counts:
#         if intersection_counts[intersection] >= 3:
#             common_intersections.append(Point(intersection))

#     # plot the intersection points
#     for intersection in common_intersections:
#         ax.scatter(intersection.x, intersection.y, intersection.z, c='r', s=100)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.savefig('intersection_plot.png')
#     # plt.show()

#     save_to_obj([intersection.coords[0] for intersection in intersections], "/code/save_obj/average_intersection2.obj")





########

# from reconstruction import reconstruction_from_json
# import numpy as np
# from shapely.geometry import LineString, Point
# from scipy.optimize import minimize
# import matplotlib
# matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from numpy import VisibleDeprecationWarning
# import warnings
# from itertools import combinations

# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

# def icp_objective(x, lines):
#     distances = []
#     for line in lines:
#         ls = LineString(line)
#         distances.append(ls.distance(Point(x)))
#     return np.sum(distances)

# def save_to_obj(vertices, filename):
#     with open(filename, 'w') as f:
#         for v in vertices:
#             vertex_str = ' '.join([f'{x:.6f}' for x in v])
#             f.write(f"v {vertex_str}\n")

# if __name__ == "__main__":
#     image_data = [
#         ('DJI_0010.JPG', (2100, 1332)),
#         ('DJI_0029.JPG', (1881, 647)),
#         ('DJI_0095.JPG', (2062, 538)),
#         ('DJI_0031.JPG', (2419, 983)),
#         ('DJI_0037.JPG', (2893, 902)),
#         ('DJI_0047.JPG', (2342, 706))
#     ]

#     scale = 10

#     reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
#     rec = reconstructions[0]

#     cam_positions = []
#     rel_positions = []

#     for image_name, pixel_coordinates in image_data:
#         pose = rec.shots[image_name].pose
#         camera = rec.shots[image_name].camera

#         rel_pix = camera.pixel_to_normalized_coordinates(pixel_coordinates)
#         bearing = camera.pixel_bearing(rel_pix)
#         bearing /= bearing[2]

#         cam_pos = pose.transform_inverse((0, 0, 0))
#         rel_pos = pose.transform_inverse(bearing * scale)

#         cam_positions.append(cam_pos)
#         rel_positions.append(rel_pos)

#     lines = []
#     for cam_pos, rel_pos in zip(cam_positions, rel_positions):
#         line = [(cam_pos[0], cam_pos[1], cam_pos[2]), (rel_pos[0], rel_pos[1], rel_pos[2])]
#         lines.append(line)

#     x0 = np.mean(lines, axis=(0, 1))
#     # print(x0)

#     result = minimize(icp_objective, x0, args=(lines,), method='Powell')
#     # print(result)
#     intersection = result.x
#     # print("Intersection:", intersection)
#     x, y, z = intersection[0], intersection[1], intersection[2]
#     # print(f"Intersection: x={x:.8f}, y={y:.8f}, z={z:.8f}")

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for line in lines:
#         ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])

#     threshold = 0.08
#     # calculate all intersection points
#     intersections = []
#     for line1, line2 in combinations(lines, 2):
#         ls1 = LineString(line1)
#         ls2 = LineString(line2)
#         intersection = ls1.intersection(ls2)
#         if intersection.geom_type == 'Point':
#             # check if the point is a common point of the two lines
#             if intersection.equals(Point(line1[0])) or intersection.equals(Point(line1[1])):
#                 continue
#             if intersection.equals(Point(line2[0])) or intersection.equals(Point(line2[1])):
#                 continue
#             intersections.append(intersection)

#     # count the number of lines that intersect at each point
#     intersection_counts = {}
#     for intersection in intersections:
#         intersection_counts[tuple(intersection.coords)] = 0
#         for line in lines:
#             ls = LineString(line)
#             if ls.distance(intersection) < threshold:
#                 intersection_counts[tuple(intersection.coords)] += 1

#     # find the intersection points that intersect with 3 or more lines
#     common_intersections = []
#     for intersection in intersection_counts:
#         if intersection_counts[intersection] >= 3:
#             common_intersections.append(Point(intersection))

#     # print the intersection points
#     for intersection in common_intersections:
#         print(intersection.x, intersection.y, intersection.z)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.savefig('intersection_plot.png')
#     # plt.show()

#     # save the intersection points to obj file
#     save_to_obj([intersection.coords[0] for intersection in common_intersections], "/code/save_obj/common_intersections.obj")



from reconstruction import reconstruction_from_json
import numpy as np
from shapely.geometry import LineString, Point
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import VisibleDeprecationWarning
import warnings
from itertools import combinations

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


def icp_objective(x, lines):
    distances = []
    for line in lines:
        ls = LineString(line)
        distances.append(ls.distance(Point(x)))
    return np.sum(distances)


def save_to_obj(vertices, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            vertex_str = ' '.join([f'{x:.6f}' for x in v])
            f.write(f"v {vertex_str}\n")


if __name__ == "__main__":
    image_data = [
        ('DJI_0010.JPG', (2100, 1332)),
        ('DJI_0029.JPG', (1881, 647)),
        ('DJI_0095.JPG', (2062, 538)),
        ('DJI_0031.JPG', (2419, 983)),
        ('DJI_0037.JPG', (2893, 902)),
        ('DJI_0047.JPG', (2342, 706))
    ]

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

    lines = []
    for cam_pos, rel_pos in zip(cam_positions, rel_positions):
        line = [(cam_pos[0], cam_pos[1], cam_pos[2]), (rel_pos[0], rel_pos[1], rel_pos[2])]
        lines.append(line)

    x0 = np.mean(lines, axis=(0, 1))

    result = minimize(icp_objective, x0, args=(lines,), method='Powell')

    intersection = result.x

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for line in lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])

    threshold = 0.08
    # calculate all intersection points
    intersections = []
    for line1, line2 in combinations(lines, 2):
        ls1 = LineString(line1)
        ls2 = LineString(line2)
        intersection = ls1.intersection(ls2)
        if intersection.geom_type == 'Point':
            # check if the point is a common point of the two lines
            if intersection.equals(Point(line1[0])) or intersection.equals(Point(line1[1])):
                continue
            if intersection.equals(Point(line2[0])) or intersection.equals(Point(line2[1])):
                continue
            intersections.append(intersection)

    # count the number of lines that intersect at each point
    intersection_counts = {}
    for intersection in intersections:
        intersection_counts[tuple(intersection.coords)] = 0
        for line in lines:
            ls = LineString(line)
            if ls.distance(intersection) < threshold:
                intersection_counts[tuple(intersection.coords)] += 1

    # find the intersection points that intersect with 3 or more lines
    common_intersections = []
    for intersection, count in intersection_counts.items():
        if count >= 3:
            common_intersections.append(Point(intersection))

    # filter out non-maximal intersections in each group
    max_intersections = []
    while len(common_intersections) > 0:
        # sort by descending z-coordinate (since we assume camera is looking down)
        common_intersections.sort(key=lambda p: -p.z)
        # take the highest intersection
        max_intersection = common_intersections.pop(0)
        # filter out any intersections that are within a certain distance of the max intersection
        common_intersections = [p for p in common_intersections if p.distance(max_intersection) > 0.1]
        max_intersections.append(max_intersection)

    # print the intersection points
    for intersection in max_intersections:
        print(intersection.x, intersection.y, intersection.z)

    # save the intersection points to obj file
    save_to_obj([intersection.coords[0] for intersection in max_intersections], "/code/save_obj/max_intersections.obj")
from reconstruction import reconstruction_from_json
import numpy as np
from shapely.geometry import LineString, Point
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # 비 GUI 백엔드를 설정합니다.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import VisibleDeprecationWarning
import warnings
from itertools import combinations

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


def icp_objective(x, lines):
    distances = []
    for line in lines:
        ls = LineString(line)
        distances.append(ls.distance(Point(x)))
    return np.sum(distances)


def save_to_obj(vertices, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            vertex_str = ' '.join([f'{x:.6f}' for x in v])
            f.write(f"v {vertex_str}\n")


if __name__ == "__main__":
    image_data = [
        ('DJI_0010.JPG', (2100, 1332)),
        ('DJI_0029.JPG', (1881, 647)),
        ('DJI_0095.JPG', (2062, 538)),
        ('DJI_0031.JPG', (2419, 983)),
        ('DJI_0037.JPG', (2893, 902)),
        ('DJI_0047.JPG', (2342, 706))
    ]

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

    lines = []
    for cam_pos, rel_pos in zip(cam_positions, rel_positions):
        line = [(cam_pos[0], cam_pos[1], cam_pos[2]), (rel_pos[0], rel_pos[1], rel_pos[2])]
        lines.append(line)

    x0 = np.mean(lines, axis=(0, 1))

    result = minimize(icp_objective, x0, args=(lines,), method='Powell')

    intersection = result.x

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for line in lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])

    threshold = 0.08
    # calculate all intersection points
    intersections = []
    for line1, line2 in combinations(lines, 2):
        ls1 = LineString(line1)
        ls2 = LineString(line2)
        intersection = ls1.intersection(ls2)
        if intersection.geom_type == 'Point':
            # check if the point is a common point of the two lines
            if intersection.equals(Point(line1[0])) or intersection.equals(Point(line1[1])):
                continue
            if intersection.equals(Point(line2[0])) or intersection.equals(Point(line2[1])):
                continue
            intersections.append(intersection)

    # count the number of lines that intersect at each point
    intersection_counts = {}
    for intersection in intersections:
        intersection_counts[tuple(intersection.coords)] = 0
        for line in lines:
            ls = LineString(line)
            if ls.distance(intersection) < threshold:
                intersection_counts[tuple(intersection.coords)] += 1

    # find the intersection points that intersect with 3 or more lines
    common_intersections = []
    for intersection, count in intersection_counts.items():
        if count >= 3:
            common_intersections.append(Point(intersection))

    # filter out non-maximal intersections in each group
    max_intersections = []
    while len(common_intersections) > 0:
        # sort by descending z-coordinate (since we assume camera is looking down)
        common_intersections.sort(key=lambda p: -p.z)
        # take the highest intersection
        max_intersection = common_intersections.pop(0)
        # filter out any intersections that are within a certain distance of the max intersection
        common_intersections = [p for p in common_intersections if p.distance(max_intersection) > 0.1]
        max_intersections.append(max_intersection)

    # print the intersection points
    for intersection in max_intersections:
        print(intersection.x, intersection.y, intersection.z)

    # save the intersection points to obj file
    save_to_obj([intersection.coords[0] for intersection in max_intersections], "/code/save_obj/max_intersections.obj")

    # visualize the intersection points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for line in lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])

    for intersection in max_intersections:
        ax.scatter(intersection.x, intersection.y, intersection.z, c='r', marker='o')

    plt.savefig('/code/save_obj/intersections.png')
    # plt.show()

