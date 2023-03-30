import plotly.graph_objs as go

def create_trace(cam_pos, rel_pos, image_name):
    """
    Create Scatter3D traces for camera and relative positions and a line trace to connect them.

    Args:
        cam_pos (tuple): Camera position coordinates (x, y, z).
        rel_pos (tuple): Relative position coordinates (x, y, z).
        image_name (str): Image name to be used as a label.

    Returns:
        tuple: A tuple containing three Scatter3D traces (cam_trace, rel_trace, line_trace).
    """
    # Create a scatter trace for camera position
    cam_trace = go.Scatter3d(
        x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
        mode='markers',
        marker=dict(size=6, color='red', symbol='circle'),
        name=f'Camera Position {image_name}'
    )
    # Create a scatter trace for relative position
    rel_trace = go.Scatter3d(
        x=[rel_pos[0]], y=[rel_pos[1]], z=[rel_pos[2]],
        mode='markers',
        marker=dict(size=6, color='blue', symbol='circle'),
        name=f'Relative Position {image_name}'
    )
    # Create a line trace to connect camera and relative positions
    line_trace = go.Scatter3d(
        x=[cam_pos[0], rel_pos[0]], y=[cam_pos[1], rel_pos[1]], z=[cam_pos[2], rel_pos[2]],
        mode='lines',
        line=dict(color='green', width=4),
        name=f'Connection {image_name}'
    )

    # Return the traces
    return cam_trace, rel_trace, line_trace

def visualize_points_with_plotly(cam_positions, rel_positions, image_names, intersections=None):
    """
    Visualize camera positions, relative positions, and intersections using Plotly.

    Args:
        cam_positions (list): List of camera position coordinates.
        rel_positions (list): List of relative position coordinates.
        image_names (list): List of image names used as labels.
        intersections (list, optional): List of intersection points. Defaults to None.
    """
    traces = []

    # Iterate through camera and relative positions and create traces for each pair
    for i, (cam_pos, rel_pos) in enumerate(zip(cam_positions, rel_positions)):
        cam_trace, rel_trace, line_trace = create_trace(cam_pos, rel_pos, image_names[i])
        traces.extend([cam_trace, rel_trace, line_trace])

    # Add intersection traces if provided
    if intersections is not None:
        for idx, intersection in enumerate(intersections):
            intersection_trace = go.Scatter3d(
                x=[intersection[0]],
                y=[intersection[1]],
                z=[intersection[2]],
                mode='markers',
                marker=dict(size=8, color='purple', symbol='diamond'),
                name=f'Intersection {idx + 1}'
            )
            traces.append(intersection_trace)

    # Create a layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Create a figure with the traces and layout
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    fig.show()