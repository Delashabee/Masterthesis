import numpy as np
import matplotlib.pyplot as plt


def read_coordinates(file_path):
    """
    Read coordinates from a file, where every three numbers constitute one coordinate point.
    """
    coordinates = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Remove empty lines and newline characters
        lines = [line.strip() for line in lines if line.strip()]
        # Convert strings to floating-point numbers
        numbers = [float(line) for line in lines]
        # Every three numbers form a coordinate
        coordinates = [numbers[i:i + 3] for i in range(0, len(numbers), 3)]
    return coordinates

def read_indices(file_path):
    """
    Read point indices from a file and store them as a list of integers.
    """
    indices = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Remove empty lines and newline characters
        lines = [line.strip() for line in lines if line.strip()]
        # Convert strings to integers
        indices = [int(line) for line in lines]
    return indices

def get_coordinates_by_indices(coordinates, indices):
    """
    Retrieve the corresponding coordinate points based on the given list of indices.
    """
    selected_coords = []
    for idx in indices:
        # Assuming indices start from 1
        if 1 <= idx <= len(coordinates):
            selected_coords.append(coordinates[idx - 1])
        else:
            print(f"Index {idx} is out of range of the coordinate list.")
    return selected_coords

def write_coordinates(coordinates, output_file):
    """
    Write coordinate points to an output file, with each value on a separate line.
    """
    with open(output_file, 'w') as f:
        for coord in coordinates:
            for value in coord:
                f.write(f"{value}\n")

def compute_quaternions(selected_coords):
    quaternions = []
    for coord in selected_coords:
        direction = -np.array(coord)
        norm = np.linalg.norm(direction)
        if norm == 0:
            print("Warning: Point is at the origin, cannot compute orientation.")
            quaternion = [1, 0, 0, 0]
        else:
            direction_normalized = direction / norm
            # Use the Z-axis as the reference vector
            reference_vector = np.array([0, 0, 1])
            axis = np.cross(reference_vector, direction_normalized)
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                if np.dot(reference_vector, direction_normalized) > 0:
                    quaternion = [1, 0, 0, 0]
                else:
                    quaternion = [0, 1, 0, 0]  # Rotate 180 degrees around the X-axis
            else:
                axis = axis / axis_norm
                angle = np.arccos(np.dot(reference_vector, direction_normalized))
                half_angle = angle / 2
                w = np.cos(half_angle)
                x, y, z = axis * np.sin(half_angle)
                quaternion = [w, x, y, z]
        quaternions.append(quaternion)
    return quaternions


def write_quaternions(quaternions, output_file):
    """
    Write the list of quaternions to an output file, one quaternion per line in the format w x y z.
    """
    with open(output_file, 'w') as f:
        for quat in quaternions:
            f.write(' '.join(map(str, quat)) + '\n')

def visualize_coordinates(coordinates, selected_coords):
    """
    Visualize all coordinate points and highlight the selected points, each with an arrow pointing to the origin.
    """
    # Separate the coordinate list into X, Y, Z components
    X_all = [coord[0] for coord in coordinates]
    Y_all = [coord[1] for coord in coordinates]
    Z_all = [coord[2] for coord in coordinates]

    # X, Y, Z components of selected points
    X_sel = [coord[0] for coord in selected_coords]
    Y_sel = [coord[1] for coord in selected_coords]
    Z_sel = [coord[2] for coord in selected_coords]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the origin
    ax.scatter(0, 0, 0, c='blue', label='Origin')
    # Plot the selected points
    ax.scatter(X_sel, Y_sel, Z_sel, c='red', label='Chosen Viewpoints')

    # Draw an arrow from each selected point to the origin
    for x, y, z in selected_coords:
        ax.quiver(x, y, z, -x, -y, -z, length=1.0, normalize=True, color='green', arrow_length_ratio=0.1)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Specify the paths to your coordinate and index files
    coordinates_file = 'D:/Programfiles/Myscvp/points_on_sphere/pack.3.40_r30.txt'
    indices_file = 'D:/Programfiles/Myscvp/SCVPNet/log/input_voxel_NBVNet.txt'

    # Read coordinates and indices from files
    coordinates = read_coordinates(coordinates_file)
    indices = read_indices(indices_file)

    # Get the selected coordinate points
    selected_coords = get_coordinates_by_indices(coordinates, indices)

    # Write the selected coordinate points to a file (optional)
    output_coordinates_file = 'D:/Programfiles/Myscvp/SCVPNet/log/selected_coordinates.txt'
    write_coordinates(selected_coords, output_coordinates_file)

    # Compute quaternions for the selected points
    quaternions = compute_quaternions(selected_coords)

    # Write the quaternions to a file
    quaternions_file = 'D:/Programfiles/Myscvp/SCVPNet/log/selected_quaternions.txt'
    write_quaternions(quaternions, quaternions_file)

    # Visualize the coordinate points and draw arrows to the origin
    visualize_coordinates(coordinates, selected_coords)
