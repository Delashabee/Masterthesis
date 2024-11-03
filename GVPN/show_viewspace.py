# Name: Jiaming Yu
# Time:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the file and parse the coordinates
def read_points_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Convert each line to a float and group every three numbers into a coordinate
    coordinates = np.array([float(line.strip()) for line in lines])
    points = coordinates.reshape(-1, 3)
    return points

# Visualization function
def visualize_points_with_arrows(points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Viewspace')

    # Set up color cycle
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

    # Plot each point and draw arrows pointing towards the origin
    for i, (x, y, z) in enumerate(points):
        ax.scatter(x, y, z, color=colors[i], label=f'Point {i+1}')
        ax.quiver(x, y, z, -x, -y, -z, length=0.1, normalize=True, color=colors[i])
        # Add labels to the points
        ax.text(x, y, z, f'{i+1}', color='black', fontsize=8, weight='bold')
    ax.scatter(0, 0, 0, color='red', s=100, label='Origin')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# File path
file_path = 'D:/Programfiles/Myscvp/points_on_sphere/pack.3.40.txt'
# Read point data
points = read_points_from_file(file_path)
# Visualize
visualize_points_with_arrows(points)
