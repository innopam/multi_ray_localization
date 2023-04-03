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