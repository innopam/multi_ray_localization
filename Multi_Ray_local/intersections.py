import itertools
import numpy as np
import json
from opensfm import io
from shapely.geometry import LineString
from reconstruction import reconstruction_from_json
from visualization import visualize_points_with_plotly


def main():
    # 재구성(reconstruction) 객체 로드
    reconstructions = reconstruction_from_json('/datasets/orange_shinhyo/opensfm/reconstruction.json')
    rec = reconstructions[0]
    # print(rec)

    # 선택된 이미지에서 선택된 2d 포인트들을 가져오기
    with open('/code/multi_ray_localization/image_data.txt', 'r') as f:
        lines = f.readlines()

    # 이미지 파일 경로와 선택된 2d 포인트의 좌표를 파싱하여 저장하기
    selected_points = {}
    for line in lines:
        image_file, x, y = line.strip().split()
        if image_file not in selected_points:
            selected_points[image_file] = []
        selected_points[image_file].append((float(x), float(y)))
    # print(selected_points)

    # 선택된 이미지에서 선택된 2d 포인트들에 대한 3d 포인트 구하기
    selected_points_3d = []
    for image_file, points in selected_points.items():
        # 선택된 이미지 id 구하기
        image_id = rec.shots[image_file].id
        # print(image_id)
        for point in points:
            # 선택된 이미지에서 선택된 2d 포인트의 위치 가져오기
            distances = []
            for key, value in rec.points.items():
                if key[0] != image_id:
                    continue
                proj = rec.shots[image_file].project(value)
                distances.append((np.linalg.norm(proj - point), key))
            if not distances:
                print(f"No matching 3D point found for image {image_file} and point {point}.")
                continue
            point_key = min(distances, key=lambda x: x[0])[1]
            point_3d = rec.points[point_key].coordinates
            # 3D 포인트 추가
            selected_points_3d.append(point_3d)

        print(list(rec.shots.keys()))
        print(list(rec.points.keys()))

    
    # 3D 선분 생성
    lines = []
    for pair in itertools.combinations(selected_points_3d, 2):
        lines.append(LineString([pair[0], pair[1]]))

    # 교차점 찾기
    intersections = []
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:], start=i+1):
            intersection = line1.intersection(line2)
            if not intersection.is_empty:
                intersections.append(intersection)

    # 교차점 시각화
    visualize_points_with_plotly(selected_points_3d, marker_color='blue', marker_size=2, marker_opacity=0.8)
    if intersections:
        visualize_points_with_plotly(intersections, marker_color='red', marker_size=4, marker_opacity=0.8)


if __name__ == "__main__":
    main()
    