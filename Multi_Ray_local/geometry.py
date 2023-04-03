import numpy as np
from itertools import combinations

def find_closest_points(cam_positions, rel_positions):

    """
    Find the closest pair of points between the relative positions given camera positions.

    Args:
        cam_positions (list): List of camera position coordinates.
        rel_positions (list): List of relative position coordinates.

    Returns:
        list: A list containing two tuples with the closest points and their indices.

    u: 첫 번째 상대 위치 p1에서 카메라 위치 cam1을 뺀 벡터 (이 벡터는 cam1에서 p1 방향)
    v: 두 번째 상대 위치 p2에서 카메라 위치 cam2을 뺀 벡터 (이 벡터는 cam2에서 p2 방향)
    w: 카메라 위치 cam1에서 카메라 위치 cam2을 뺀 벡터 (이 벡터는 cam1에서 cam2 방향)
    a: 벡터 u와 u 사이의 내적 (이 값은 벡터 u의 길이의 제곱과 동일)
    b: 벡터 u와 v 사이의 내적
    c: 벡터 v와 v 사이의 내적 (이 값은 벡터 v의 길이의 제곱과 동일)
    d: 벡터 u와 w 사이의 내적
    e: 벡터 v와 w 사이의 내적
    denom: 선분의 스칼라 곱의 값을 계산하는 분모로 사용되는 값으로, (a * c - b * b)로 계산됨.

    """

    min_distance = float('inf')  # Initialize minimum distance as infinity
    closest_points = []

    # Iterate through all combinations of relative position pairs
    for (i1, p1), (i2, p2) in combinations(enumerate(rel_positions), 2):
        cam1, cam2 = cam_positions[i1], cam_positions[i2]  # Get the corresponding camera positions
        u, v, w = p1 - cam1, p2 - cam2, cam1 - cam2  # Compute vector differences
        a, b, c, d, e = np.dot(u, u), np.dot(u, v), np.dot(v, v), np.dot(u, w), np.dot(v, w)
        denom = a * c - b * b

        # If the denominator is zero, skip this iteration
        if denom == 0:
            continue

        # Calculate s and t, the scalar multipliers for u and v
        s, t = (b * e - c * d) / denom, (a * e - b * d) / denom
        # Compute the points along the lines defined by u and v
        ps, pt = cam1 + s * u, cam2 + t * v
        # Calculate the distance between the two points
        dist = np.linalg.norm(ps - pt)

        # Update the closest points and minimum distance if a closer pair is found
        if dist < min_distance:
            min_distance = dist
            closest_points = [(ps, i1), (pt, i2)]

    # Return the closest pair of points and their indices
    return closest_points

# def find_closest_points_multi(cam_positions, rel_positions):

#     min_distance = float('inf')  # Initialize minimum distance as infinity
#     closest_points = []

#     # Iterate through all combinations of relative position pairs
#     for rel_pos_combination in combinations(enumerate(rel_positions), 2):
#         cam_combination = [(i, cam_positions[i]) for i, _ in rel_pos_combination]
        
#         closest_points_pair = find_closest_points([cam for _, cam in cam_combination], [rel for _, rel in rel_pos_combination])

#         dist = np.linalg.norm(closest_points_pair[0][0] - closest_points_pair[1][0])

#     # Update the closest points and minimum distance if a closer pair is found
#         if dist < min_distance:
#             min_distance = dist
#             closest_points = [(p, cam_combination[i][0]) for i, (p, _) in enumerate(closest_points_pair)]

#     # Return the closest pair of points and their indices
#     return closest_points
