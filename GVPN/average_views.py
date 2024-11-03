# Name: Jiaming Yu
# Time:
import os

def count_views_in_file(file_path):
    """Count the number of lines in a file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

def find_ids_files_and_average_views(main_folder):
    """Find files with 'ids' in the name and calculate the average number of lines."""
    total_files = 0
    total_views = 0

    # Walk through all subdirectories and files
    for root, _, files in os.walk(main_folder):
        for file in files:
            if 'ids' in file:
                file_path = os.path.join(root, file)
                view_count = count_views_in_file(file_path)
                total_views += view_count
                total_files += 1
                print(f"Processed file: {file_path}, Lines: {view_count}")

    if total_files == 0:
        print("No 'ids' files found.")
        return 0

    average_lines = total_views / total_files
    return average_lines

if __name__ == "__main__":
    main_folder = "D:/Programfiles/Myscvp/industrial_label_data/48_views"
    average_views = find_ids_files_and_average_views(main_folder)
    print(f"Average number of views in 'ids' files: {average_views}")
    average_views_file_path = os.path.join(main_folder, 'average_views.txt')
    with open(average_views_file_path, 'w') as avg_file:
        avg_file.write(f"Average number of views in 'ids' files: {average_views}\n")

    print(f"Average number of lines saved to {average_views_file_path}")
