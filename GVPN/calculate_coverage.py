# Name: Jiaming Yu
# Time:
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def delete_files(base_dir, prefix, extension):
    # Traverse the specified directory and its subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Find all files that match the conditions
        for file in files:
            if file.startswith(prefix) and file.endswith(extension):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

'''# Set base directory, file prefix, and extension
base_directory = 'D:/Programfiles/Myscvp/coverage'
file_prefix = '030'
file_extension = '.txt'

# Delete files
delete_files(base_directory, file_prefix, file_extension)'''

def calculate_average_coverage_rate_from_folder(base_dir):
    coverage_rates = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(base_dir, file_name)
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith("Coverage Rate:"):
                            rate_str = line.split(':')[1].strip()
                            coverage_rate = float(rate_str)
                            print(coverage_rate)
                            coverage_rates.append(coverage_rate)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    if coverage_rates:
        average_rate = sum(coverage_rates) / len(coverage_rates)
        output_file_path = os.path.join(base_dir, 'average_coverage_rate.txt')
        try:
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"Average Coverage Rate: {average_rate:.6f}\n")
            print(f"Average coverage rate for folder {base_dir}: {average_rate:.6f}")
        except Exception as e:
            print(f"Error writing to file {output_file_path}: {e}")
    else:
        print(f"No coverage rate data found in folder {base_dir}.")

def calculate_average_coverage_rate(base_dir):
    for i in range(4, 65):
        folder_name = str(i)
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_path}, not a directory.")
            continue

        coverage_rates = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r') as file:
                        first_line = file.readline().strip()
                        if first_line.startswith("Coverage Rate:"):
                            rate_str = first_line.split(':')[1].strip()
                            coverage_rate = float(rate_str)
                            coverage_rates.append(coverage_rate)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

        if coverage_rates:
            average_rate = sum(coverage_rates) / len(coverage_rates)
            output_file_path = os.path.join(folder_path, 'average_coverage_rate.txt')
            try:
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"Average Coverage Rate: {average_rate:.6f}\n")
                print(f"Average coverage rate for folder {folder_name}: {average_rate:.6f}")
            except Exception as e:
                print(f"Error writing to file {output_file_path}: {e}")
        else:
            print(f"No coverage rate data found in folder {folder_name}.")

'''# Set base directory
base_directory = 'D:/Programfiles/Myscvp/coverage'

# Calculate the average and save to a txt file
calculate_average_coverage_rate(base_directory)'''

def read_average_coverage_rates(base_dir):
    x_values = []
    y_values = []

    for i in range(4, 65):
        folder_name = str(i)
        folder_path = os.path.join(base_dir, folder_name)
        avg_file_path = os.path.join(folder_path, 'average_coverage_rate.txt')

        if not os.path.isfile(avg_file_path):
            print(f"Skipping {avg_file_path}, file not found.")
            continue

        try:
            with open(avg_file_path, 'r') as file:
                first_line = file.readline().strip()
                if first_line.startswith("Average Coverage Rate:"):
                    avg_rate_str = first_line.split(':')[1].strip()
                    average_rate = float(avg_rate_str)
                    x_values.append(i)
                    y_values.append(average_rate)
        except Exception as e:
            print(f"Error reading file {avg_file_path}: {e}")

    return x_values, y_values


def plot_coverage_rates(x_values, y_values):
    # Convert lists to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Create a spline of order 3 (cubic)
    spline = make_interp_spline(x_values, y_values, k=3)

    # Generate new x values for smooth curve
    x_new = np.linspace(x_values.min(), x_values.max(), 300)
    y_smooth = spline(x_new)

    plt.figure(figsize=(10, 6))
    plt.plot(x_new, y_smooth, color='b', label='Smooth Curve')
    plt.scatter(x_values, y_values, color='r', marker='o', label='Original Points')
    plt.xlabel('Number of Views')
    plt.ylabel('Average Coverage Rate')
    plt.title('Average Coverage Rate for Different Number of Views')
    plt.xticks(range(4, 65, 2))  # Set x-axis ticks
    plt.grid(True)
    plt.legend()
    plt.savefig('average_coverage_rate_plot.png')
    plt.show()

def calculate_derivative_and_plot(x_values, y_values, output_file_path, average_groundtruth_num):
    # Calculate the derivative (difference between successive points)
    derivatives = np.diff(y_values)
    voxel_increase= derivatives * average_groundtruth_num
    derivatives *= 100
    with open(output_file_path, 'w') as file:
        file.write("Number of Views\tDerivative of Average Coverage Rate (%)\tvoxel increase\n")
        for x, derivative, voxel in zip(x_values[:-1], derivatives, voxel_increase):
            file.write(f"{x}\t{derivative:.6f}\t{voxel:.6f}\n")

    # Since the derivative is between points, we plot it against the midpoints of the x_values
    plt.figure(figsize=(10, 6))
    plt.plot(x_values[:-1], derivatives, color='g', marker='o', label='Derivative of Coverage Rate')
    plt.xlabel('Number of Views (midpoints)')
    plt.ylabel('Derivative of Average Coverage Rate')
    plt.title('Derivative of Average Coverage Rate vs. Number of Views')
    plt.xticks(range(4, 64, 2))  # Set x-axis ticks
    plt.yticks(np.arange(-0.6, 3.4, 0.1))
    plt.axhline(0, color='red', linestyle='-', linewidth=1.5, label='y=0')
    plt.grid(True)
    plt.legend()
    plt.savefig('derivative_coverage_rate_plot.png')
    plt.show()


def calculate_average_ground_truth_num(directory):
    ground_truth_nums = []

    # Traverse all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            # Read file content
            with open(file_path, 'r') as file:
                for line in file:
                    if 'groud truth num:' in line:
                        # Extract the number after 'groud truth num:'
                        num_str = line.split('groud truth num:')[1].strip()
                        try:
                            ground_truth_num = int(num_str)
                            ground_truth_nums.append(ground_truth_num)
                        except ValueError:
                            print(f"Invalid number format in file {filename}: {num_str}")
                        break  # Stop further searching once found

    if ground_truth_nums:
        # Calculate the average
        average = sum(ground_truth_nums) / len(ground_truth_nums)
        print(f"The average of 'groud truth num:' is: {average:.2f}")
    else:
        print("No 'groud truth num:' found in the files.")
    return average

# Set the base directory
directory = 'D:/Programfiles/Myscvp/SCVPNet/log/ResNeXt_coverage'  # Replace this with the actual directory path
calculate_average_coverage_rate_from_folder(directory)

'''# Calculate and output the average
average_groundtruth_num = calculate_average_ground_truth_num(directory)

# Set base directory
base_directory = 'D:/Programfiles/Myscvp/coverage'
#calculate_average_coverage_rate(base_directory)
#base_directory2 = 'D:/Programfiles/SCVP-Simulation-main/SCVP-Simulation-main/Set_covering_data_generator/coverage'
# Read averages and plot the curve
x_vals, y_vals = read_average_coverage_rates(base_directory)
output_file_path = os.path.join(base_directory, 'derivative_output.txt')
#plot_coverage_rates(x_vals, y_vals)
calculate_derivative_and_plot(x_vals, y_vals, output_file_path, average_groundtruth_num)
#calculate_average_coverage_rate(base_directory2)'''
