# Name: Jiaming Yu
# Time:
import os

def list_subfolders(directory, output_file):

    subfolders = [f.name for f in os.scandir(directory) if f.is_dir()]


    with open(output_file, 'w') as file:
        for subfolder in subfolders:
            file.write(subfolder + '\n')

if __name__ == "__main__":

    target_directory = 'D:/Programfiles/Myscvp/industrial_label_data/64_views/novaltest'
    output_file_path = 'D:/Programfiles/Myscvp/industrial_label_data/64_views/Name_of_Novaltest_Objects.txt'


    list_subfolders(target_directory, output_file_path)
