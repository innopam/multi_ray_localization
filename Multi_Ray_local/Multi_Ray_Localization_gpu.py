import time
import json
from itertools import combinations
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
from reconstruction import reconstruction_from_json
from scipy.spatial.distance import euclidean
from shapely.geometry import LineString, Point
from pyproj import Proj, Transformer
from collections import defaultdict
from scipy.spatial import KDTree
from numba import cuda, float32, int32, jit


scale = 10

# 이미지 데이터(json 파일)를 불러오는 함수
def load_image_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    return [(item['image_name'], item['coordinates']) for item in data['image_data']]

# 최적화 함수에서 사용되는 목적 함수
# def objective_function(x: np.ndarray, lines: List[np.ndarray]) -> float:
#     distances = []
#     for line in lines:
#         ls = LineString(line)
#         distances.append(ls.distance(Point(x)))
#     return np.sum(distances)

# 최종적으로 찾은 점들을 obj 파일로 저장하는 함수
def save_to_obj(vertices: List[np.ndarray], filename: str) -> None:
    with open(filename, 'w') as f:
        for v in vertices:
            vertex_str = ' '.join([f'{x:.6f}' for x in v])
            f.write(f"v {vertex_str}\n")

# max_intersections의 좌표를 위도, 경도, 고도로 변환하는 함수
def convert_to_lla(intersections: List[Point], rec) -> List[Tuple[float, float, float]]:
    lla_intersections = []
    for intersection in intersections:
        lla = rec.get_reference().to_lla(intersection.x, intersection.y, intersection.z)
        lla_intersections.append(lla)
    return lla_intersections

def convert_to_utm52n(lla_coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652")
    utm52n_coords = [transformer.transform(lat, lon, alt) for lat, lon, alt in lla_coords]
    return utm52n_coords

# 두 직선 사이에서 가장 가까운 두 점과 그 점을 이용해 만들어진 교점을 계산하는 함수
"""
1. 두 선분의 끝점을 numpy 배열로 변환합니다.
2. 각 선분의 방향 벡터와 끝점의 차이 벡터, 끝점의 차이 벡터와 방향 벡터의 내적을 계산합니다.
3. 공식에 따라 s, t 값을 계산합니다.
4. 두 선분의 가장 가까운 두 점을 point1, point2에 저장합니다.
5. point1과 point2 사이의 거리가 threshold 이하인 경우, 교점을 계산합니다.
6. 교점이 선분의 끝점과 중복되는 경우, 교점을 무시합니다.
7. 교점이 중복되지 않는 경우, intersections 리스트에 교점을 추가합니다.
8. point1, point2를 반환합니다.
"""

@jit(nopython=True)
def closest_points_jit(line1, line2, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    A, B = np.array(line1)
    C, D = np.array(line2)

    v, w = B - A, D - C
    P = A - C

    a = np.dot(v, v)
    b = np.dot(v, w)
    c = np.dot(w, w)
    d = np.dot(v, P)
    e = np.dot(w, P)

    denom = a * c - b * b
    s, t = 0.0, 0.0

    if denom != 0:
        s = max(0, min(1, (b * e - c * d) / denom))
        t = max(0, min(1, (a * e - b * d) / denom))

    point1 = A + s * (B - A)
    point2 = C + t * (D - C)

    if euclidean(point1, point2) < threshold:
        intersection = (point1 + point2) / 2
        intersection = Point(intersection)

        if intersection.equals(Point(line1[0])) or intersection.equals(Point(line1[1])):
            return None, None
        if intersection.equals(Point(line2[0])) or intersection.equals(Point(line2[1])):
            return None, None

        intersections.append(intersection)
    else:
        return None, None

    return point1, point2

@cuda.jit
def calculate_intersections_kernel(lines, threshold, intersections, count):
    i = cuda.grid(1)
    if i < count:
        line1 = lines[i]
        for j in range(i+1, count):
            line2 = lines[j]
            point1, point2 = closest_points_jit(line1, line2, threshold)
            if point1 is not None and point2 is not None:
                intersection = (point1 + point2) / 2
                intersections.append(Point(intersection))

# 모든 직선들 사이의 교점을 계산하는 함수
def calculate_intersections(lines: List[List[Tuple[float, float, float]]], threshold: float) -> List[Point]:
    intersections = []
    n = len(lines)
    lines_np = np.array(lines, dtype=np.float32)
    intersections_np = np.zeros((n * (n - 1) // 2, 6), dtype=np.float32)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    calculate_intersections_kernel[blocks, threads_per_block](lines_np, threshold, intersections_np, n)

    for intersection in intersections_np:
        if intersection.sum() != 0:
            intersections.append(Point(intersection))
    return intersections

# 여러 개의 직선들과 구한 교점들을 이용하여, 공통으로 나타나는 교점들을 찾는 함수
"""
1. 교점마다 딕셔너리에 추가하고, 카운트를 0으로 초기화합니다.
2. 직선들과 교점들을 비교하면서, 직선과 교점 사이의 거리가 threshold 이하인 경우, 해당 교점의 카운트를 1 증가시킵니다.
3. 교점들 중에서, 카운트가 5 이상인 교점들을 common_intersections 리스트에 추가합니다.
4. common_intersections 리스트에서 가장 높은 z 좌표를 가진 교점을 선택하고, 해당 교점과 거리가 0.1 이하인 다른 교점들을 모두 제거합니다.
5. 선택된 교점을 max_intersections 리스트에 추가하고, common_intersections 리스트에서 해당 교점을 제거합니다.
6. common_intersections 리스트가 비어질 때까지 4-5 과정을 반복합니다.
7. max_intersections 리스트를 반환합니다.
"""

@jit(nopython=True)
def distance_to_line(point, line):
    ls = LineString(line)
    return ls.distance(Point(point))

@cuda.jit
def filter_common_intersections_kernel(lines, intersections, threshold, common_intersections, intersection_counts, n, m):
    idx = cuda.grid(1)
    if idx < n:
        intersection = intersections[idx]
        count = 0
        for line in lines:
            d = calculate_distance(line, intersection)
            if d <= threshold:
                count += 1
        common_intersections[idx] = intersection
        intersection_counts[idx] = count

def filter_common_intersections(lines: List[List[Tuple[float, float, float]]], intersections: List[Point], threshold: float, min_count: int = 5) -> List[Point]:
    n = len(intersections)
    m = len(lines)
    lines_np = np.array(lines, dtype=np.float32)
    intersections_np = np.array([np.array(intersection) for intersection in intersections], dtype=np.float32)
    common_intersections_np = np.zeros((n, 3), dtype=np.float32)
    intersection_counts_np = np.zeros(n, dtype=np.int32)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    filter_common_intersections_kernel[blocks, threads_per_block](lines_np, intersections_np, threshold, common_intersections_np, intersection_counts_np, n, m)

    common_intersections = [(Point(common_intersection), count) for common_intersection, count in zip(common_intersections_np, intersection_counts_np) if count >= min_count]
    
    max_intersections = []

    while common_intersections:
        print("Calculating intersections...")
        max_intersection, intersection_counts_np, common_intersections_np = select_intersection(intersection_counts_np, common_intersections_np)
        max_intersections.append(max_intersection)
        common_intersections = [(Point(common_intersection), count) for common_intersection, count in zip(common_intersections_np, intersection_counts_np) if count >= min_count]
        print(f"Max intersection: {max_intersection}, Remaining common intersections: {len(common_intersections)}")

    return max_intersections

def main():
    start_time = time.time()

    image_data = load_image_data('/code/Multi_Ray_local/test_data.json')

    reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
    rec = reconstructions[0]

    camera_positions = []
    ray_end_positions = []

    processed_images_count = 0
    total_images = len(image_data)

    for image_name, pixel_coordinates_list in image_data:
        try:
            if image_name not in rec.shots:
                print(f"Skipping image {image_name} as it does not exist in the reconstructions object")
                continue

            print(f"Starting processing of image {image_name} ({processed_images_count + 1}/{total_images})")
            for pixel_coordinates in pixel_coordinates_list:
                pose = rec.shots[image_name].pose
                camera = rec.shots[image_name].camera

                pixel_coordinates_np = np.array(pixel_coordinates).reshape(2, 1).astype(np.float64)
                normalized_coordinates = camera.pixel_to_normalized_coordinates(pixel_coordinates_np)
                bearing = camera.pixel_bearing(normalized_coordinates)
                bearing /= bearing[2]

                camera_pos = pose.transform_inverse((0, 0, 0))
                ray_end_pos = pose.transform_inverse(bearing * scale)

                camera_positions.append(camera_pos)
                ray_end_positions.append(ray_end_pos)

            processed_images_count += 1
            print(f"Finished processing image {image_name} ({processed_images_count}/{total_images})")
                
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
    print("Processing complete")



    lines = []
    for cam_pos, rel_pos in zip(camera_positions, ray_end_positions):
        line = [(cam_pos[0], cam_pos[1], cam_pos[2]), (rel_pos[0], rel_pos[1], rel_pos[2])]
        lines.append(line)
        print(len(lines))

        
    threshold = 0.015

    print("Calculating intersections")
    start_time1 = time.time()
    intersections = calculate_intersections(lines, threshold)
    end_time1 = time.time()
    step1_time = end_time1 - start_time1
    print(f"step1_time: {step1_time:.2f} seconds")
    print(f"Calculated {len(intersections)} intersections")

    start_time2 = time.time()
    max_intersections = filter_common_intersections(lines, intersections, threshold)
    end_time2 = time.time()
    step2_time = end_time2 - start_time2
    print(f"step2_time: {step2_time:.2f} seconds")
    print(f"Filtered {len(max_intersections)} common intersections")

    coordinates = [(p.x, p.y, p.z) for p in max_intersections]
    for coord in coordinates:
        print(f"Coordinate: x={coord[0]}, y={coord[1]}, z={coord[2]}")
    save_to_obj([(p.x, p.y, p.z) for p in max_intersections], "/code/save_obj/max_intersections.obj")

    lla_coordinates = convert_to_lla(max_intersections, rec)
    for lla in lla_coordinates:
        print(f"LLA Coordinate: lat={lla[0]}, lon={lla[1]}, alt={lla[2]}")
    save_to_obj(lla_coordinates, "/code/save_obj/lla_intersections.obj")

    utm52n_coordinates = convert_to_utm52n(lla_coordinates)
    for utm in utm52n_coordinates:
        print(f"UTM52N Coordinate: x={utm[0]}, y={utm[1]}, z={utm[2]}")
    save_to_obj(utm52n_coordinates, "/code/save_obj/UTM52N_intersections.obj")
    
    fig = go.Figure()

    for line in lines:
        fig.add_trace(go.Scatter3d(
            x=[line[0][0], line[1][0]],
            y=[line[0][1], line[1][1]],
            z=[line[0][2], line[1][2]],
            mode='lines',
            line=dict(width=2)
        ))
    for intersection in max_intersections:
        fig.add_trace(go.Scatter3d(
            x=[intersection.x],
            y=[intersection.y],
            z=[intersection.z],
            mode='markers',
            marker=dict(size=6, color='red')
        ))

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()