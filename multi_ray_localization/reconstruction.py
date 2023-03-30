import json
from opensfm import io

def reconstruction_from_json(file_path):
    """
    Load reconstruction data from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the reconstruction data.

    Returns:
        list: A list of Reconstruction objects.
    """
    # Open the JSON file and read its content
    with open(file_path, 'r') as f:
        reconstructions_data = json.load(f)

    # Convert the JSON data into a list of Reconstruction objects
    reconstructions = io.reconstructions_from_json(reconstructions_data)

    # Return the list of Reconstruction objects
    return reconstructions

# import json
# from typing import Dict, List

# class Shot:
#     def __init__(self, id, reconstruction_id, camera_id, pose):
#         self.id = id
#         self.reconstruction_id = reconstruction_id
#         self.camera_id = camera_id
#         self.pose = pose

#     def to_dict(self):
#         return {
#             'id': self.id,
#             'reconstruction_id': self.reconstruction_id,
#             'camera_id': self.camera_id,
#             'pose': self.pose
#         }

#     @classmethod
#     def from_dict(cls, data):
#         if isinstance(data, str) or not data:
#             return cls(None, None, None, None)
#         return cls(data['id'], data['reconstruction_id'], data['camera_id'], data['pose'])



# class Camera:
#     def __init__(self, id, model, width, height, params):
#         self.id = id
#         self.model = model
#         self.width = width
#         self.height = height
#         self.params = params

#     def to_dict(self):
#         return {
#             'id': self.id,
#             'model': self.model,
#             'width': self.width,
#             'height': self.height,
#             'params': self.params
#         }

#     @classmethod
#     def from_dict(cls, data):
#         return cls(data['id'], data['model'], data['width'], data['height'], data['params'])

# class Point:
#     def __init__(self, id, color, coordinates, reprojection_error, reconstructed):
#         self.id = id
#         self.color = color
#         self.coordinates = coordinates
#         self.reprojection_error = reprojection_error
#         self.reconstructed = reconstructed

#     def to_dict(self):
#         return {
#             'id': self.id,
#             'color': self.color,
#             'coordinates': self.coordinates,
#             'reprojection_error': self.reprojection_error,
#             'reconstructed': self.reconstructed
#         }

#     @classmethod
#     def from_dict(cls, data):
#         if isinstance(data, str) or not data:
#             return cls(None, None, None, None, None)
#         return cls(data['id'], data['color'], data['coordinates'], data['reprojection_error'], data['reconstructed'])


# class Reconstruction:
#     def __init__(self, cameras, shots, points):
#         self.cameras = cameras
#         self.shots = shots
#         self.points = points

# def cameras_from_data(data: List[dict]) -> Dict[str, Camera]:
#     """
#     Convert a list of dictionary data into a dictionary of Camera objects.
#     """
#     cameras = {}
#     for camera_data in data:
#         if not camera_data:
#             continue
#         try:
#             camera_data = json.loads(camera_data)
#         except json.JSONDecodeError:
#             continue
#         camera = Camera.from_dict(camera_data)
#         cameras[camera.id] = camera
#     return cameras


# def shots_from_data(data: List[dict]) -> Dict[str, Shot]:
#     """
#     Convert a list of dictionary data into a dictionary of Shot objects.
#     """
#     shots = {}
#     for shot_data in data:
#         if not shot_data:
#             continue
#         try:
#             shot = Shot.from_dict(shot_data)
#         except KeyError:
#             continue
#         shots[shot.id] = shot
#     return shots

# def points_from_data(data: List[dict]) -> Dict[str, Point]:
#     """
#     Convert a list of dictionary data into a dictionary of Point objects.
#     """
#     points = {}
#     for point_data in data:
#         point = Point.from_dict(point_data)
#         points[point.id] = point
#     return points

# def reconstruction_from_json(file_path: str) -> List[Reconstruction]:
#     """
#     Load reconstruction data from a JSON file.

#     Args:
#         file_path (str): The path to the JSON file containing the reconstruction data.

#     Returns:
#         list: A list of Reconstruction objects.
#     """
#     # Open the JSON file and read its content
#     with open(file_path, 'r') as f:
#         reconstructions_data = json.load(f)

#     # Convert the JSON data into a list of Reconstruction objects
#     reconstructions = []
#     for reconstruction_data in reconstructions_data:
#         cameras = cameras_from_data(reconstruction_data['cameras'])
#         shots = shots_from_data(reconstruction_data['shots'])
#         points = points_from_data(reconstruction_data['points'])
#         reconstruction = Reconstruction(cameras, shots, points)
#         reconstructions.append(reconstruction)

#     # Return the list of Reconstruction objects
#     return reconstructions