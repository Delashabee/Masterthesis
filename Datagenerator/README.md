# Set_covering_data_generator
These codes are for the generation of your own dataset.
DefaultConfiguration.yaml is the input config file of our code.
points_on_sphere contians some condidate view space (thanks Spherical Codes neilsloane.com/packings).
## Installion
These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 9.1.1.
Note that Gurobi is only free for academic use.
Our codes can be compiled by Visual Studio 2019 with c++ 14 and run on Windows 10. For other system, please check the file read/write or multithreading functions in the codes.
## Note
Change "const static size_t maxSize = 100000;" to "const static size_t maxSize = 1000" in file OcTreeKey.h, so that the code will run faster.
## Usage
1. Sample 3d object model from *.obj or *.ply to *.pcd and you can find the code in tools.
2. Change the model_path and name_of_pcd in file DefaultConfiguration.yaml.
3. Run compiled program of main.cpp
4. The output data are labeled datasets.
