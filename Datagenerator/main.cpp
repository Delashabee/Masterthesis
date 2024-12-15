// #include <windows.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <pcl/point_cloud.h>
#include <fstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

typedef unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include <gurobi_c++.h>


//Virtual_Perception_3D.hpp
void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model,Share_Data* share_data);

class Perception_3D {
public:
    Share_Data* share_data;
    octomap::ColorOcTree* ground_truth_model;
    int full_voxels;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

    // Constructor: Initializes the Perception_3D object with shared data
    Perception_3D(Share_Data* _share_data) {
        share_data = _share_data;
        ground_truth_model = share_data->ground_truth_model;
        full_voxels = share_data->full_voxels;
    }

    // Destructor
    ~Perception_3D() {
        // No dynamic memory to free
    }

    // Simulates perception from the current best view
    bool precept(View* now_best_view) {
        double now_time = clock();

        // Initialize a point cloud to store perception results in parallel
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_parallel(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud_parallel->is_dense = false;
        cloud_parallel->points.resize(full_voxels);

        // Get the camera pose in the world coordinate system
        Eigen::Matrix4d view_pose_world;
        now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
        view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();

        // Get the octomap key for the camera origin
        octomap::OcTreeKey key_origin;
        bool key_origin_have = ground_truth_model->coordToKeyChecked(
            now_best_view->init_pos(0),
            now_best_view->init_pos(1),
            now_best_view->init_pos(2),
            key_origin
        );

        if (key_origin_have) {
            // Convert the key to coordinates
            octomap::point3d origin = ground_truth_model->keyToCoord(key_origin);

            // Prepare the endpoints (coordinates of each voxel in the ground truth model)
            octomap::point3d* end = new octomap::point3d[full_voxels];
            octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs();
            for (int i = 0; i < full_voxels; i++) {
                end[i] = it.getCoordinate();
                it++;
            }

            // Create threads for parallel processing of rays from the camera to each voxel
            thread** precept_process = new thread * [full_voxels];
            for (int i = 0; i < full_voxels; i++) {
                precept_process[i] = new thread(
                    precept_thread_process,
                    i,
                    cloud_parallel,
                    &origin,
                    &end[i],
                    &view_pose_world,
                    ground_truth_model,
                    share_data
                );
            }

            // Wait for all threads to finish
            for (int i = 0; i < full_voxels; i++)
                (*precept_process[i]).join();

            // Clean up allocated memory
            delete[] end;
            for (int i = 0; i < full_voxels; i++)
                precept_process[i]->~thread();
            delete[] precept_process;
        } else {
            cout << "View out of map. Check." << endl;
        }

        // Collect valid points from the parallel point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud = temp;
        cloud->is_dense = false;
        cloud->points.resize(full_voxels);

        auto ptr = cloud->points.begin();
        int valid_point = 0;
        auto p = cloud_parallel->points.begin();

        for (int i = 0; i < cloud_parallel->points.size(); i++, p++) {
            if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
            (*ptr).x = (*p).x;
            (*ptr).y = (*p).y;
            (*ptr).z = (*p).z;
            (*ptr).b = (*p).b;
            (*ptr).g = (*p).g;
            (*ptr).r = (*p).r;
            valid_point++;
            ptr++;
        }

        // Resize the cloud to contain only valid points
        cloud->width = valid_point;
        cloud->height = 1;
        cloud->points.resize(valid_point);

        // Record the current captured cloud
        share_data->vaild_clouds++;
        share_data->clouds.push_back(cloud);

        cout << "Virtual cloud obtained with execution time " << clock() - now_time << " ms." << endl;

        if (share_data->show) { // Visualization
            pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
            viewer1->setBackgroundColor(255, 255, 255);
            viewer1->addCoordinateSystem(0.1);
            viewer1->initCameraParameters();
            viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");

            // Draw the camera's coordinate axes
            Eigen::Vector4d X(0.05, 0, 0, 1);
            Eigen::Vector4d Y(0, 0.05, 0, 1);
            Eigen::Vector4d Z(0, 0, 0.05, 1);
            Eigen::Vector4d O(0, 0, 0, 1);
            X = view_pose_world * X;
            Y = view_pose_world * Y;
            Z = view_pose_world * Z;
            O = view_pose_world * O;

            viewer1->addLine<pcl::PointXYZ>(
                pcl::PointXYZ(O(0), O(1), O(2)),
                pcl::PointXYZ(X(0), X(1), X(2)),
                255, 0, 0, "X" + to_string(-1)
            );
            viewer1->addLine<pcl::PointXYZ>(
                pcl::PointXYZ(O(0), O(1), O(2)),
                pcl::PointXYZ(Y(0), Y(1), Y(2)),
                0, 255, 0, "Y" + to_string(-1)
            );
            viewer1->addLine<pcl::PointXYZ>(
                pcl::PointXYZ(O(0), O(1), O(2)),
                pcl::PointXYZ(Z(0), Z(1), Z(2)),
                0, 0, 255, "Z" + to_string(-1)
            );

            viewer1->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                5, "X" + to_string(-1)
            );
            viewer1->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                5, "Y" + to_string(-1)
            );
            viewer1->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                5, "Z" + to_string(-1)
            );

            // Spin the viewer until manually closed
            while (!viewer1->wasStopped()) {
                viewer1->spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
        }

        // Clean up the parallel point cloud
        cloud_parallel->~PointCloud();
        return true;
    }
};


// Projects a pixel in image space to a 3D point in world space along a ray at a given maximum range
inline octomap::point3d project_pixel_to_ray_end(
	int x, int y,
	rs2_intrinsics& color_intrinsics,
	Eigen::Matrix4d& now_camera_pose_world,
	float max_range
) {
	// Create a pixel array with the given x and y coordinates
	float pixel[2] = { x, y };
	float point[3];

	// De-project the pixel to a 3D point in the camera coordinate system using the camera intrinsics and max range
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);

	// Convert the point to homogeneous coordinates (add a 1 for the w component)
	Eigen::Vector4d point_world(point[0], point[1], point[2], 1);

	// Transform the point from the camera coordinate system to the world coordinate system
	point_world = now_camera_pose_world * point_world;

	// Return the point as an octomap::point3d object (discarding the homogeneous coordinate)
	return octomap::point3d(point_world(0), point_world(1), point_world(2));
}

// Thread function that processes a single ray from the camera to a voxel in the octomap
void precept_thread_process(
	int i,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	octomap::point3d* _origin,
	octomap::point3d* _end,
	Eigen::Matrix4d* _view_pose_world,
	octomap::ColorOcTree* _ground_truth_model,
	Share_Data* share_data
) {
	// Dereference the pointers to get the actual data
	octomap::point3d origin = *_origin;
	Eigen::Matrix4d view_pose_world = *_view_pose_world;
	octomap::ColorOcTree* ground_truth_model = _ground_truth_model;

	// Initialize a point with default coordinates (0,0,0) and color
	pcl::PointXYZRGB point;
	point.x = 0; point.y = 0; point.z = 0;

	// Transform the endpoint from world coordinates to the camera coordinate system
	Eigen::Vector4d end_3d(_end->x(), _end->y(), _end->z(), 1);
	Eigen::Vector4d vertex = view_pose_world.inverse() * end_3d;

	// Convert the 3D point to an array for projection
	float point_3d[3] = { vertex(0), vertex(1), vertex(2) };
	float pixel[2];

	// Project the 3D point onto the image plane using the camera intrinsics
	rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);

	// Check if the projected pixel is within the image bounds
	if (pixel[0] < 0 || pixel[0] > share_data->color_intrinsics.width ||
		pixel[1] < 0 || pixel[1] > share_data->color_intrinsics.height) {
		// If not, assign the default point and exit
		cloud->points[i] = point;
		return;
	}

	// Use the pixel to compute the corresponding 3D point at a maximum range
	octomap::point3d end = project_pixel_to_ray_end(
		static_cast<int>(pixel[0]), static_cast<int>(pixel[1]),
		share_data->color_intrinsics, view_pose_world, 1.0
	);

	// Compute the direction vector from the origin to the end point
	octomap::point3d direction = end - origin;
	octomap::point3d end_point;

	// Cast a ray from the origin in the direction to find the first occupied voxel in the octomap
	bool found_end_point = ground_truth_model->castRay(
		origin, direction, end_point, true, 6.0 * share_data->predicted_size
	);

	// If no occupied voxel is found along the ray, assign the default point and exit
	if (!found_end_point) {
		cloud->points[i] = point;
		return;
	}

	// If the end point is the same as the origin, the view is inside the object (unexpected)
	if (end_point == origin) {
		std::cout << "View inside the object. Check!" << std::endl;
		cloud->points[i] = point;
		return;
	}

	// Check if the end point is within the boundaries of the octomap
	octomap::OcTreeKey key_end;
	bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);

	if (key_end_have) {
		// Search for the node corresponding to the end point
		octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
		if (node != nullptr) {
			// Retrieve the color of the voxel
			octomap::ColorOcTreeNode::Color color = node->getColor();

			// Assign the coordinates and color to the point
			point.x = end_point.x();
			point.y = end_point.y();
			point.z = end_point.z();
			point.b = color.b;
			point.g = color.g;
			point.r = color.r;
		}
	}

	// Store the point in the point cloud at index i
	cloud->points[i] = point;
}


// views_voxels_LM.hpp
class views_voxels_LM {
public:
	Share_Data* share_data;
	View_Space* view_space;
	vector<vector<bool>> graph; // Graph representing which views cover which voxels
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map; // Mapping from voxels to indices
	int num_of_voxel; // Total number of voxels
	set<int>* chosen_views; // Set of views that have been already chosen
	GRBEnv* env; // Gurobi environment for optimization
	GRBModel* model; // Gurobi model for optimization
	vector<GRBVar> x; // Variables representing whether a view is selected (1) or not (0)
	GRBLinExpr obj; // Objective function for the optimization

	// Function to solve the optimization problem
	void solve() {
		// Optimize the model
		model->optimize();
		// Show nonzero variables (commented out)
		/*
		for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0)
				cout << x[i].get(GRB_StringAttr_VarName) << " " << x[i].get(GRB_DoubleAttr_X) << endl;
		// Show number of views selected
		cout << "Obj: " << model->get(GRB_DoubleAttr_ObjVal) << endl;
		*/
	}

	// Function to get the set of selected view IDs after optimization
	vector<int> get_view_id_set() {
		vector<int> ans;
		for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0)
				ans.push_back(i);
		return ans;
	}

	// Constructor: Initializes the optimization problem
	views_voxels_LM(Share_Data* _share_data, View_Space* _view_space, set<int>* _chosen_views) {
		double now_time = clock();
		share_data = _share_data;
		view_space = _view_space;
		chosen_views = _chosen_views;

		// Create a mapping from voxels to unique IDs
		num_of_voxel = 0;
		voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < share_data->voxels.size(); i++) {
			for (auto& it : *share_data->voxels[i]) {
				if (voxel_id_map->find(it.first) == voxel_id_map->end()) {
					(*voxel_id_map)[it.first] = num_of_voxel++;
				}
			}
		}
		//cout << num_of_voxel << " real | gt " << share_data->full_voxels << endl;

		// Initialize the graph representing which views cover which voxels
		graph.resize(share_data->num_of_views);
		for (int i = 0; i < share_data->num_of_views; i++) {
			graph[i].resize(num_of_voxel);
			for (int j = 0; j < num_of_voxel; j++) {
				graph[i][j] = 0;
			}
		}

		// Set to store voxels that do not need to be covered again
		set<int> voxels_not_need;
		for (int i = 0; i < share_data->voxels.size(); i++) {
			for (auto& it : *share_data->voxels[i]) {
				graph[i][(*voxel_id_map)[it.first]] = 1; // Mark that view i covers voxel j
				if (chosen_views->find(i) != chosen_views->end()) {
					voxels_not_need.insert((*voxel_id_map)[it.first]); // Voxel is already covered
				}
			}
		}

		//cout << (*voxel_id_map).size() << endl;
		//cout << voxels_not_need.size() << endl;

		// The following code is commented out; it was possibly used for debugging or visualization
		/*
		octomap::ColorOcTree* octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
		for (auto& it : *voxel_id_map) {
			if (voxels_not_need.find(it.second) == voxels_not_need.end()) continue;
			octo_model->setNodeValue(it.first, (float)0, true);
			octo_model->setNodeColor(it.first, 255, 0, 0);
		}
		octo_model->updateInnerOccupancy();
		octo_model->write(share_data->save_path + "/Utest.ot");
		*/

		// Construct the integer linear programming model
		now_time = clock();
		env = new GRBEnv();
		model = new GRBModel(*env);
		x.resize(share_data->num_of_views);

		// Create variables: x[i] is 1 if view i is selected, 0 otherwise
		for (int i = 0; i < share_data->num_of_views; i++)
			x[i] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + to_string(i));

		// Set the objective function: minimize the total number of selected views
		for (int i = 0; i < share_data->num_of_views; i++)
			obj += x[i];
		model->setObjective(obj, GRB_MINIMIZE);

		// Add constraints: each voxel must be covered by at least one selected view
		for (int j = 0; j < num_of_voxel; j++) {
			if (voxels_not_need.find(j) != voxels_not_need.end()) continue; // Skip voxels already covered
			GRBLinExpr subject_of_voxel;
			for (int i = 0; i < share_data->num_of_views; i++)
				if (graph[i][j] == 1)
					subject_of_voxel += x[i]; // If view i covers voxel j, add x[i] to the constraint
			model->addConstr(subject_of_voxel >= 1, "c" + to_string(j)); // Ensure voxel j is covered
		}

		// Set a time limit for the optimization
		model->set("TimeLimit", "10");
		//cout << "Integer linear program formulated with executed time " << clock() - now_time << " ms." << endl;
	}

	// Destructor: Clean up allocated resources
	~views_voxels_LM() {
		delete voxel_id_map;
		delete env;
		delete model;
	}
};


// SC_NBV_Labeler: Class for Next Best View (NBV) labeling
class SC_NBV_Labeler {
public:
    Share_Data* share_data;  // Pointer to shared data structure
    View_Space* view_space;  // Pointer to view space, which contains different possible views
    Perception_3D* percept;  // Pointer to Perception_3D object that simulates sensor data
    pcl::visualization::PCLVisualizer::Ptr viewer;  // PCL visualizer for 3D visualization
    int toward_state;  // State that determines the current object orientation
    int rotate_state;  // State that determines the current object rotation

    // Function to check the size of the predicted bounding box (BBX)
    // It returns the percentage of valid points that fall inside the predicted size
    double check_size(double predicted_size, Eigen::Vector3d object_center_world, vector<Eigen::Vector3d>& points) {
        int valid_points = 0;  // Count of points that are valid (inside the BBX)
        
        // Iterate over all points to check if they fall within the predicted BBX
        for (auto& ptr : points) {
            if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
            if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
            if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
            valid_points++;
        }
        
        // Return the ratio of valid points to the total number of points
        return (double)valid_points / (double)points.size();
    }

    // Constructor: Initializes the NBV Labeler with shared data, and sets the object orientation and rotation
    SC_NBV_Labeler(Share_Data* _share_data, int _toward_state = 0, int _rotate_state = 0) {
        share_data = _share_data;  // Initialize shared data pointer
        toward_state = _toward_state;  // Set the current object orientation state
        rotate_state = _rotate_state;  // Set the current object rotation state

        // Transform the point cloud according to the specified orientation (toward_state)
        pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, share_data->get_toward_pose(toward_state));

        // Rotate the point cloud by 45 degrees (rotate_state)
        Eigen::Matrix3d rotation;  // Define a 3D rotation matrix
        rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(45 * rotate_state * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());  // Rotate around the Z-axis
                   
        // Create a transformation matrix and apply the rotation
        Eigen::Matrix4d T_pose(Eigen::Matrix4d::Identity(4, 4));
        T_pose(0, 0) = rotation(0, 0); T_pose(0, 1) = rotation(0, 1); T_pose(0, 2) = rotation(0, 2);
        T_pose(1, 0) = rotation(1, 0); T_pose(1, 1) = rotation(1, 1); T_pose(1, 2) = rotation(1, 2);
        T_pose(2, 0) = rotation(2, 0); T_pose(2, 1) = rotation(2, 1); T_pose(2, 2) = rotation(2, 2);
        pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, T_pose);  // Apply the transformation

        // Initialize the ground truth (GT) point cloud
        share_data->cloud_ground_truth->is_dense = false;
        share_data->cloud_ground_truth->points.resize(share_data->cloud_pcd->points.size());
        share_data->cloud_ground_truth->width = share_data->cloud_pcd->points.size();
        share_data->cloud_ground_truth->height = 1;

        // Iterator for ground truth point cloud
        auto ptr = share_data->cloud_ground_truth->points.begin();
        auto p = share_data->cloud_pcd->points.begin();
        
        // Check the point cloud dimensions and adjust the scale unit if necessary (mm -> cm -> dm -> m)
        float unit = 1.0;  // Default unit (mm)
        for (auto& ptr : share_data->cloud_pcd->points) {
            if (fabs(ptr.x) >= 10 || fabs(ptr.y) >= 10 || fabs(ptr.z) >= 10) {
                unit = 0.1;
                cout << "change unit from <mm> to <cm>." << endl;
                for (auto& ptr_inner : share_data->cloud_pcd->points) {
                    if (fabs(ptr_inner.x * unit) >= 10 || fabs(ptr_inner.y * unit) >= 10 || fabs(ptr_inner.z * unit) >= 10) {
                        unit = 0.01;
                        cout << "change unit from <cm> to <dm>." << endl;
                        for (auto& ptr_inner2 : share_data->cloud_pcd->points) {
                            if (fabs(ptr_inner2.x * unit) >= 10 || fabs(ptr_inner2.y * unit) >= 10 || fabs(ptr_inner2.z * unit) >= 10) {
                                unit = 0.001;
                                cout << "change unit from <dm> to <m>." << endl;
                                break;
                            }
                        }
                        break;
                    }
                }
                break;
            }
        }

        // Convert all points to the appropriate scale and store them in an Eigen vector
        vector<Eigen::Vector3d> points;
        for (auto& ptr : share_data->cloud_pcd->points) {
            Eigen::Vector3d pt(ptr.x * unit, ptr.y * unit, ptr.z * unit);  // Scale points
            points.push_back(pt);
        }

        // Compute the object's center in the world frame (average position of all points)
        Eigen::Vector3d object_center_world(0, 0, 0);
        for (auto& ptr : points) {
            object_center_world(0) += ptr(0);
            object_center_world(1) += ptr(1);
            object_center_world(2) += ptr(2);
        }
        object_center_world /= points.size();  // Calculate the average to find the center

        // Perform binary search to determine the appropriate bounding box (BBX) size
        double l = 0, r = 0, mid;
        for (auto& ptr : points) {
            r = max(r, (object_center_world - ptr).norm());  // Find the maximum distance from the center
        }
        mid = (l + r) / 2;
        double percent = check_size(mid, object_center_world, points);  // Check the coverage percentage
        double prev_percent = percent;

        // Perform binary search until the BBX covers 90-95% of the object points
        while (percent > 0.95 || percent < 1.0) {
            if (percent > 0.95) {
                r = mid;
            } else if (percent < 1.0) {
                l = mid;
            }
            mid = (l + r) / 2;
            percent = check_size(mid, object_center_world, points);  // Recalculate the percentage
            if (fabs(prev_percent - percent) < 0.001) break;
            prev_percent = percent;
        }

        // Estimate the predicted size of the object and apply scaling if necessary
        double predicted_size = 1.2 * mid;
        float scale = 1.0;
        if (predicted_size > 0.1) {
            scale = 0.1 / predicted_size;  // Scale the object down to a manageable size
            cout << "object large. change scale to about 0.1 m." << endl;
        }

        // Transform the object and assign color values to the point cloud
        double min_z = object_center_world(2);  // Track the minimum Z value (for ground truth)
        for (int i = 0; i < share_data->cloud_pcd->points.size(); i++, p++) {
            (*ptr).x = (*p).x * scale * unit;
            (*ptr).y = (*p).y * scale * unit;
            (*ptr).z = (*p).z * scale * unit;
            (*ptr).b = 0;  // Set color values
            (*ptr).g = 0;
            (*ptr).r = 255;

            // Update the ground truth octomap model based on the point coordinates
            octomap::OcTreeKey key;
            bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
            if (key_have) {
                octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
                if (voxel == NULL) {
                    share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
                    share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
                }
            }
            min_z = min(min_z, (double)(*ptr).z);  // Update minimum Z value

            // Update GT sample voxel map
            octomap::OcTreeKey key_sp;
            bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
            if (key_have_sp) {
                octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
                if (voxel_sp == NULL) {
                    share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
                    share_data->GT_sample->integrateNodeColor(key_sp, (*ptr).r, (*ptr).g, (*ptr).b);
                }
            }
            ptr++;
        }

        // Store the minimum Z value to track the base level of the object
        share_data->min_z_table = min_z - share_data->ground_truth_resolution;

        // Update the inner occupancy of the octomap models
        share_data->ground_truth_model->updateInnerOccupancy();
        share_data->GT_sample->updateInnerOccupancy();

        // Count the number of voxels in the GT sample and the full voxel set
        share_data->init_voxels = 0;
        for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it) {
            share_data->init_voxels++;
        }
        cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;

        share_data->full_voxels = 0;
        for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
            share_data->full_voxels++;
        }

        // Initialize the view space and perception model
        view_space = new View_Space(share_data);
        percept = new Perception_3D(share_data);

        srand(time(0));  // Initialize random seed
    }


	double compute_coverage() {
		double total_voxels = share_data->full_voxels;
		double now_time = clock();
		// Project each view and count the voxels
		int full_num = 0;
		unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < view_space->views.size(); i++) {
			percept->precept(&view_space->views[i]);
			// Get voxel map
			int num = 0;
			unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
			for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
				octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
				if (voxel->find(key) == voxel->end()) {
					(*voxel)[key] = num++;
				}
				if (all_voxel->find(key) == all_voxel->end()) {
					(*all_voxel)[key] = full_num++;
				}
			}
			share_data->voxels.push_back(voxel);
		}
		delete all_voxel;
		cout << "All voxels (cloud) num is " << full_num << endl;
		cout << "Ground truth voxels num is " << total_voxels << endl;
		cout << "All virtual cloud processing time: " << clock() - now_time << " ms." << endl;

		// Calculate the coverage rate
		double coverage_rate = full_num / total_voxels;
		share_data->access_directory(share_data->save_path);
		string file_path = share_data->save_path + "/" + share_data->name_of_pcd + "_" + to_string(share_data->num_of_views) + ".txt";
		ofstream fout(file_path);
		if (fout.is_open()) {
			fout << "Coverage Rate: " << coverage_rate << endl;
			fout << "Full num: " << full_num << endl;
			fout << "Ground truth num: " << total_voxels << endl;
			fout.close();
		}
		else {
			cerr << "Error opening file for writing: " << file_path << endl;
		}
		return 0;
	}


	/*double compute_coverage() {
    // Extract surface voxels from the ground truth
    unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> surface_voxels;
    octomap::ColorOcTree* ground_truth_model = share_data->ground_truth_model;
    double resolution = share_data->ground_truth_resolution;

    // Define the six directional offsets
    double neighbor_offsets[6][3] = {
        {1, 0, 0}, {-1, 0, 0},
        {0, 1, 0}, {0, -1, 0},
        {0, 0, 1}, {0, 0, -1}
    };

    // Iterate through all voxels
    for (octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs(), end = ground_truth_model->end_leafs(); it != end; ++it) {
        octomap::OcTreeKey key = it.getKey();
        bool is_surface = false;
        // Check the six directions for neighbors
        for (int i = 0; i < 6; ++i) {
            octomap::OcTreeKey neighbor_key = key;
            neighbor_key[0] += neighbor_offsets[i][0];
            neighbor_key[1] += neighbor_offsets[i][1];
            neighbor_key[2] += neighbor_offsets[i][2];

            // If the neighbor does not exist or is unoccupied, the current voxel is a surface voxel
            if (!ground_truth_model->search(neighbor_key)) {
                is_surface = true;
                break;  // Stop checking once we determine it is a surface voxel
            }
        }

        // If it's a surface voxel, add it to the set
        if (is_surface) {
            surface_voxels.insert(key);
        }
    }

    double total_voxels = share_data->full_voxels;
    double total_surface_voxels = surface_voxels.size();
    double now_time = clock();

    int full_num = 0;
    unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
    for (int i = 0; i < view_space->views.size(); i++) {
        percept->precept(&view_space->views[i]);
        // Get voxel map
        int num = 0;
        unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
        for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
            octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
            if (voxel->find(key) == voxel->end()) {
                (*voxel)[key] = num++;
            }
            if (all_voxel->find(key) == all_voxel->end()) {
                (*all_voxel)[key] = full_num++;
            }
        }
        share_data->voxels.push_back(voxel);
    }
    delete all_voxel;
    cout << "All voxels (cloud) num is " << full_num << endl;
    cout << "Ground truth voxels num is " << total_voxels << endl;
    cout << "Ground truth's surface voxels num is " << total_surface_voxels << endl;
    cout << "All virtual cloud processing time: " << clock() - now_time << " ms." << endl;

    // Calculate the coverage rate
    double coverage_rate = full_num / total_surface_voxels;
    share_data->access_directory(share_data->save_path);
    string file_path = share_data->save_path + "/" + share_data->name_of_pcd + "_" + to_string(share_data->num_of_views) + ".txt";

    ofstream fout(file_path);
    if (fout.is_open()) {
        fout << "Coverage Rate: " << coverage_rate << endl;
        fout << "Detected surface voxels num: " << full_num << endl;
        fout << "Ground truth surface voxels num: " << total_surface_voxels << endl;
        fout.close();
    } else {
        cerr << "Error opening file for writing: " << file_path << endl;
    }

    return coverage_rate;
	}
*/


	int transformer_label() {
		double now_time = clock();
		for (int i = 0; i < view_space->views.size(); i++) {
			percept->precept(&view_space->views[i]);
			//get voxel map
			int num = 0;
			unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
			for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
				octomap::OcTreeKey key = share_data->octo_model->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
				if (voxel->find(key) == voxel->end()) {
					(*voxel)[key] = num++;
				}
			}
			share_data->voxels.push_back(voxel);
		}
		cout << "all virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		now_time = clock();
		for (int i = 0; i < view_space->views.size(); i++) //if(i==0)
		{
			set<int> chosen_views;
			chosen_views.insert(i);
			views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data, view_space, &chosen_views);
			SCOP_solver->solve();
			vector<int> need_views = SCOP_solver->get_view_id_set();
			delete SCOP_solver;
			share_data->access_directory(share_data->save_path);
			string filename = share_data->save_path + "/pcd_toward" + to_string(toward_state) + "_rotate" + to_string(rotate_state) + "_view" + to_string(i) + ".pcd";
			if (pcl::io::savePCDFileBinary(filename, *share_data->clouds[i]) == -1) {
				cerr << "Error saving point cloud to " << filename << endl;
			}
			else {
				cout << "Saved " << share_data->clouds[i]->points.size() << " points to " << filename << endl;
			}
			ofstream fout_view_ids(share_data->save_path + "/ids_toward" + to_string(toward_state) + "_rotate" + to_string(rotate_state) + "_view" + to_string(i) + ".txt");
			
			for (int i = 0; i < need_views.size(); i++) {
				fout_view_ids << need_views[i] << '\n';
			}
			cout << "labed " << i << " getted with executed time " << clock() - now_time << " ms." << endl;
		}
		return 0;
	}

	int simulation() {
		// Simulate the perception for each view in the view space (assuming percept and view_space are already defined and initialized)
		for (size_t i = 0; i < view_space->views.size(); i++) {
			percept->precept(&view_space->views[i]);
		}

		// Read the required view IDs from a file
		vector<int> need_views;
		need_views.push_back(0); // Initial view
		string filename = "D:/Programfiles/Myscvp/SCVPNet/log/" + share_data->name_of_pcd + "_NBVNet.txt";

		ifstream infile(filename);
		if (!infile.is_open()) {
			std::cerr << "Unable to open file: " << filename << endl;
			return 1;
		}

		int view_id;
		while (infile >> view_id) {
			need_views.push_back(view_id);
		}
		infile.close();

		// Verify the content read from the file
		for (size_t i = 0; i < need_views.size(); ++i) {
			std::cout << "View ID: " << need_views[i] << endl;
		}

		double total_voxels = share_data->full_voxels; // Total number of voxels
		double now_time = clock();

		// Count the required voxels and compute the coverage rate
		int full_num = 0;
		std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> all_voxel;

		// Merge the point clouds
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		for (size_t j = 0; j < need_views.size(); j++) {
			int view_index = need_views[j];
			*merged_cloud += *(share_data->clouds[view_index]); // Merge the point clouds from different views

			// Iterate through each point, generate voxel keys, and check if they exist
			for (size_t i = 0; i < share_data->clouds[view_index]->points.size(); i++) {
				octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(
					share_data->clouds[view_index]->points[i].x,
					share_data->clouds[view_index]->points[i].y,
					share_data->clouds[view_index]->points[i].z
				);

				// If the key does not exist in all_voxel, add and count it
				if (all_voxel.find(key) == all_voxel.end()) {
					all_voxel[key] = full_num++;
				}
			}
		}

		// Output the statistics
		cout << "All voxels (cloud) num is " << full_num << endl;
		cout << "Ground truth voxels num is " << total_voxels << endl;

		// Compute the coverage rate
		double coverage_rate = static_cast<double>(full_num) / total_voxels;

		// Voxel filter: used to reduce point cloud resolution and remove duplicate points
		pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
		voxel_filter.setInputCloud(merged_cloud);
		voxel_filter.setLeafSize(0.002f, 0.002f, 0.002f);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
		voxel_filter.filter(*cloud_filtered);

		// Save the merged point cloud to a file
		pcl::io::savePCDFileASCII("D:/Programfiles/Myscvp/SCVPNet/log/merged_cloud_filtered" + share_data->name_of_pcd + "NBVNet.pcd", *cloud_filtered);

		// Save the point count and coverage rate to a txt file
		std::ofstream outfile("D:/Programfiles/Myscvp/SCVPNet/log/point_count_merged_cloud_filtered" + share_data->name_of_pcd + "NBVNet.txt");
		if (outfile.is_open()) {
			outfile << "Filtered cloud point count: " << cloud_filtered->points.size() << endl;
			outfile << "Coverage Rate: " << coverage_rate << endl;
			outfile << "Full num: " << full_num << endl;
			outfile << "Ground truth num: " << total_voxels << endl;
			outfile.close();
		}
		else {
			cerr << "Unable to open output file to write point cloud data" << endl;
			return 1;
		}

		return 0;
	}


	int reconstruction_gpt() {
		// Load the already stored point cloud from a file (without RGB data)
		pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud(new pcl::PointCloud<pcl::PointXYZ>());
		std::string cloud_filename = "D:/Programfiles/Myscvp/SCVPNet/log/merged_cloud_filtered" + share_data->name_of_pcd + "NBVNet.pcd";

		if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_filename, *stored_cloud) == -1) {
			std::cerr << "Couldn't read file: " << cloud_filename << std::endl;
			return 1;
		}

		// Validate the loaded point cloud
		std::cout << "Loaded " << stored_cloud->points.size() << " points from " << cloud_filename << std::endl;

		// Perform normal estimation on the loaded point cloud
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz(new pcl::search::KdTree<pcl::PointXYZ>());
		tree_xyz->setInputCloud(stored_cloud);
		ne.setInputCloud(stored_cloud);
		ne.setSearchMethod(tree_xyz);
		ne.setKSearch(20); // Set the number of nearest neighbors for normal estimation
		ne.compute(*normals);

		// Combine the point cloud and its normals
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
		pcl::concatenateFields(*stored_cloud, *normals, *cloud_with_normals);

		// Ensure the sizes match
		if (cloud_with_normals->points.size() != stored_cloud->points.size()) {
			std::cerr << "Size mismatch between cloud and normals." << std::endl;
			return 1;
		}

		// Use KdTree for the point cloud with normals
		pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
		tree->setInputCloud(cloud_with_normals);

		// Greedy Projection Triangulation
		pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
		gp3.setSearchRadius(0.05); // Set the maximum distance between connected points (larger values fill more gaps)
		gp3.setMu(3); // Maximum allowable deviation from the surface
		gp3.setMaximumNearestNeighbors(200); // Maximum number of nearest neighbors to consider
		gp3.setMaximumSurfaceAngle(M_PI / 4); // Max surface angle (in radians) between connected points
		gp3.setMinimumAngle(M_PI / 18); // Minimum angle (in radians) for triangles
		gp3.setMaximumAngle(2 * M_PI / 3); // Maximum angle (in radians) for triangles
		gp3.setNormalConsistency(true); // Set to true if normals are consistent
		//gp3.setSamplesPerNode(8);

		pcl::PolygonMesh triangles;
		gp3.setInputCloud(cloud_with_normals);
		gp3.setSearchMethod(tree);
		gp3.reconstruct(triangles);

		// Save the reconstructed model to a file
		pcl::io::savePLYFile("D:/Programfiles/Myscvp/SCVPNet/log/reconstructed_model_greedy_filtered_" + share_data->name_of_pcd + "NBVNet.ply", triangles);

		// Visualize the reconstructed model
		/*pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Reconstructed Model Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPolygonMesh(triangles, "mesh");
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();

		viewer->spin();*/

		return 0;
	}
	// Normal estimation
	/*pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	tree->setInputCloud(merged_cloud);
	ne.setInputCloud(merged_cloud);
	ne.setSearchMethod(tree);
	ne.setKSearch(20); // Set the number of nearest neighbors for searching
	ne.compute(*normals);

	// Merge point cloud and normals into PointXYZRGBNormal
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::concatenateFields(*merged_cloud, *normals, *cloud_with_normals);

	// Check the merged point cloud
	if (cloud_with_normals->points.size() != merged_cloud->points.size()) {
		std::cerr << "Point cloud and normals size mismatch." << std::endl;
		return 1;
	}

	// Create Poisson reconstruction object and set parameters
	pcl::Poisson<pcl::PointXYZRGBNormal> pn;
	pn.setDegree(4); // Set degree parameter for Poisson reconstruction
	pn.setDepth(12);  // Set maximum depth of the octree
	pn.setIsoDivide(10);
	pn.setSamplesPerNode(15);  // Set minimum number of sample points per octree node
	pn.setScale(1.25f);        // Set scale of the bounding cube (note the use of float)
	pn.setSolverDivide(10);

	pn.setConfidence(true);
	pn.setManifold(false);
	pn.setOutputPolygons(false);

	pcl::PolygonMesh triangles;
	pn.setInputCloud(cloud_with_normals);
	pn.performReconstruction(triangles);

	// Save the reconstructed 3D model
	pcl::io::savePLYFile("D:/Programfiles/Myscvp/SCVPNet/log/reconstructed_model_085.ply", triangles);

	// Visualize the reconstructed 3D model
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Merged Cloud Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(triangles, "mesh");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	// Set display mode (optional)
	// viewer->setRepresentationToPointsForAllActors(); // Point mode
	// viewer->setRepresentationToWireframeForAllActors(); // Wireframe mode
	// viewer->setRepresentationToSurfaceForAllActors(); // Surface mode

	viewer->spin();*/



	int label() {
		double now_time = clock();
		for (int i = 0; i < view_space->views.size(); i++) {
			percept->precept(&view_space->views[i]);
			//get voxel map
			int num = 0;
			unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
			for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
				octomap::OcTreeKey key = share_data->octo_model->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
				if (voxel->find(key) == voxel->end()) {
					(*voxel)[key] = num++;
				}
			}
			share_data->voxels.push_back(voxel);
		}
		cout << "all virtual cloud get with executed time " << clock() - now_time << " ms." << endl;

		//show covering
		set<int> chosen_views;
		views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data, view_space, &chosen_views);
		SCOP_solver->solve();
		vector<int> need_views = SCOP_solver->get_view_id_set();
		delete SCOP_solver;
		pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Label"));
		viewer1->setBackgroundColor(255, 255, 255);
		//viewer1->addCoordinateSystem(0.1);
		viewer1->initCameraParameters();
		//�� �� �� �� ��� �� �� ��
		int r[8] = {255,0,0,255,255,0,128};
		int g[8] = {0,255,0,255,0,255,0};
		int b[8] = {0,0,255,0,255,255,255};
		for (int j = 0; j < need_views.size(); j++) { // 2 3 9 13 23 27 30 31
			cout << "view id is " << need_views[j] << "." << endl;
			//int r = rand() % 256;
			//int g = rand() % 256;
			//int b = rand() % 256;
			cout << r[j] << " " << g[j] << " " << b[j] << endl;
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(share_data->clouds[need_views[j]], r[j], g[j], b[j]);
			//pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZRGB> RandomColor(share_data->clouds[need_views[j]]);
			viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->clouds[need_views[j]], single_color, "cloud" + to_string(need_views[j]));
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			view_space->views[need_views[j]].get_next_camera_pos(view_space->now_camera_pose_world, view_space->object_center_world);
			Eigen::Matrix4d view_pose_world = (view_space->now_camera_pose_world * view_space->views[need_views[j]].pose.inverse()).eval();
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), r[j], g[j], b[j], "X" + to_string(j));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), r[j], g[j], b[j], "Y" + to_string(j));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), r[j], g[j], b[j], "Z" + to_string(j));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(j));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(j));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(j));
			// Add text annotation near the origin O for each view
			//viewer1->addText3D("View ID: " + std::to_string(need_views[j]), pcl::PointXYZ(O(0), O(1), O(2)), 0.02, r[j] / 255.0, g[j] / 255.0, b[j] / 255.0, "Text" + std::to_string(j));
		}
		//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_black(share_data->clouds[i], 0, 0, 0);
		//viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->clouds[i], color_black, "cloud" + to_string(i));
		while (!viewer1->wasStopped())
		{
			viewer1->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}

		now_time = clock();
		for (int i = 0; i < view_space->views.size(); i++) //if(i==0)
		{
			set<int> chosen_views;
			chosen_views.insert(i);
			views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data, view_space, &chosen_views);
			SCOP_solver->solve();
			vector<int> need_views =  SCOP_solver->get_view_id_set();
			delete SCOP_solver;

			octomap::ColorOcTree* octo_model_test = new octomap::ColorOcTree(share_data->octomap_resolution);
			for (auto p : share_data->clouds[i]->points) {
				octo_model_test->setNodeValue(p.x, p.y, p.z, octo_model_test->getProbHitLog(), true);
				octo_model_test->integrateNodeColor(p.x, p.y, p.z, 255, 0, 0);
			}
			for (double x = share_data->object_center_world(0) - 0.2; x <= share_data->object_center_world(0) + 0.2; x += share_data->octomap_resolution)
				for (double y = share_data->object_center_world(2) - 0.2; y <= share_data->object_center_world(2) + 0.2; y += share_data->octomap_resolution) {
					double z = share_data->min_z_table;
					octo_model_test->setNodeValue(x, y, z, octo_model_test->getProbHitLog(), true);
					octo_model_test->integrateNodeColor(x, y, z, 0, 0, 255);
				}
			octo_model_test->updateInnerOccupancy();
			//octo_model_test->write(share_data->save_path + "/test.ot");
			int num_of_test = 0;
			for (octomap::ColorOcTree::leaf_iterator it = octo_model_test->begin_leafs(), end = octo_model_test->end_leafs(); it != end; ++it) {
				num_of_test++;
			}
			//cout << num_of_test << " " << share_data->clouds[i]->points.size() << endl;
			Perception_3D test(share_data);
			test.ground_truth_model = octo_model_test;
			test.full_voxels = num_of_test;
			test.precept(&view_space->views[i]);
			delete octo_model_test;
			/*
			Eigen::Matrix4d view_pose_world;
			view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
			view_pose_world = (share_data->now_camera_pose_world * view_space->views[i].pose.inverse()).eval();
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(0, 0, 0);
			viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->clouds[view_space->views.size()], "cloud");
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
			*/

			octomap::ColorOcTree* octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
			for (double x = share_data->object_center_world(0) - share_data->predicted_size; x <= share_data->object_center_world(0) + share_data->predicted_size; x += share_data->octomap_resolution)
				for (double y = share_data->object_center_world(1) - share_data->predicted_size; y <= share_data->object_center_world(1) + share_data->predicted_size; y += share_data->octomap_resolution)
					for (double z = share_data->object_center_world(2) - share_data->predicted_size; z <= share_data->object_center_world(2) + share_data->predicted_size; z += share_data->octomap_resolution)
						octo_model->setNodeValue(x, y, z, (float)0, true); //��ʼ������0.5����logoddsΪ0
			octo_model->updateInnerOccupancy();
			octomap::Pointcloud cloud_octo;
			for (auto p : share_data->clouds[view_space->views.size()]->points) {
				cloud_octo.push_back(p.x, p.y, p.z);
			}
			octo_model->insertPointCloud(cloud_octo, octomap::point3d(view_space->views[i].init_pos(0), view_space->views[i].init_pos(1), view_space->views[i].init_pos(2)), -1, true, false);
			for (auto p : share_data->clouds[i]->points) {
				if (p.z >= share_data->min_z_table + share_data->octomap_resolution) octo_model->integrateNodeColor(p.x, p.y, p.z, 255, 0, 0);
				else octo_model->integrateNodeColor(p.x, p.y, p.z, 0, 0, 255);
			}
			for (octomap::ColorOcTree::leaf_iterator it = octo_model->begin_leafs(), end = octo_model->end_leafs(); it != end; ++it) {
				if (it->getOccupancy() > 0.65) {
					if (it.getZ() >= share_data->min_z_table + share_data->octomap_resolution) octo_model->integrateNodeColor(it.getKey(), 255, 0, 0);
					else octo_model->integrateNodeColor(it.getKey(), 0, 0, 255);
				}
			}
			octo_model->updateInnerOccupancy();

			share_data->clouds[view_space->views.size()]->~PointCloud();
			share_data->clouds.pop_back();

			share_data->access_directory(share_data->save_path);
			ofstream fout_grid(share_data->save_path + "/grid_toward" + to_string(toward_state) + "_rotate" + to_string(rotate_state) + "_view" + to_string(i) + ".txt");
			ofstream fout_view_ids(share_data->save_path + "/ids_toward" + to_string(toward_state) + "_rotate" + to_string(rotate_state) + "_view" + to_string(i) + ".txt");
			//octo_model->write(share_data->save_path + "/grid_toward" + to_string(toward_state) + "_rotate" + to_string(rotate_state) + "_view" + to_string(i) + ".ot");
			//octo_model->write(share_data->save_path + "/grid.ot");
			int num_of_squared_voxels = 0;
			//octomap::ColorOcTree* octo_model_square = new octomap::ColorOcTree(share_data->octomap_resolution);
			for (double x = share_data->object_center_world(0) - share_data->predicted_size; x <= share_data->object_center_world(0) + share_data->predicted_size; x += share_data->octomap_resolution)
				for (double y = share_data->object_center_world(1) - share_data->predicted_size; y <= share_data->object_center_world(1) + share_data->predicted_size; y += share_data->octomap_resolution)
					for (double z = share_data->object_center_world(2) - share_data->predicted_size; z <= share_data->object_center_world(2) + share_data->predicted_size; z += share_data->octomap_resolution)
					{
						auto node = octo_model->search(x, y, z);
						if (node == NULL) cout << "what?" << endl;
						//fout_grid << x - share_data->object_center_world(0) << ' ' << y - share_data->object_center_world(1) << ' ' << z - share_data->object_center_world(2) << ' ' << node->getOccupancy() << '\n';
						fout_grid << node->getOccupancy() << '\n';
						num_of_squared_voxels++;
						//octo_model_square->setNodeValue(x, y, z, node->getLogOdds(), true);
						//if (node->getOccupancy() > 0.65) {
						//	if(z >= share_data->min_z_table + share_data->octomap_resolution) octo_model_square->integrateNodeColor(x, y, z, 255, 0, 0);
						//	else octo_model_square->integrateNodeColor(x, y, z, 0, 0, 255);
						//}
					}
			if (num_of_squared_voxels != 32*32*32) cout << "voxels size wrong." << endl;
			for (int i = 0; i < need_views.size(); i++) {
				fout_view_ids << need_views[i] << '\n';
			}
			//octo_model_square->updateInnerOccupancy();
			//octo_model_square->write(share_data->save_path + "/square_grid.ot");
			//delete octo_model_square;
			delete octo_model;

			if (share_data->show) { //��ʾBBX�����λ�á�GT
			//if (true) { //��ʾBBX�����λ�á�GT
				pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration"));
				viewer->setBackgroundColor(255, 255, 255);
				//viewer->addCoordinateSystem(0.05);
				viewer->initCameraParameters();
				//view_space->add_bbx_to_cloud(viewer);
				//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> gray(share_data->cloud_ground_truth, 128, 128, 128);
				//viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, gray, "cloud_ground_truth");
				//��ѡȡλ��
				view_space->views[i].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), share_data->object_center_world);
				Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * view_space->views[i].pose.inverse()).eval();
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red(share_data->clouds[i], 255, 0, 0);
				viewer->addPointCloud<pcl::PointXYZRGB>(share_data->clouds[i], red, "cloud_i");
				//���λ��
				Eigen::Vector4d X(0.05, 0, 0, 1);
				Eigen::Vector4d Y(0, 0.05, 0, 1);
				Eigen::Vector4d Z(0, 0, 0.05, 1);
				Eigen::Vector4d O(0, 0, 0, 1);
				X = view_pose_world * X;
				Y = view_pose_world * Y;
				Z = view_pose_world * Z;
				O = view_pose_world * O;
				viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X-1");
				viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 255, 0, 0, "Y-1");
				viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 255, 0, 0, "Z-1");
				viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X-1");
				viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y-1");
				viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z-1");
				for (int j = 0; j < need_views.size(); j++) {
					view_space->views[need_views[j]].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * view_space->views[need_views[j]].pose.inverse()).eval();
					//���λ��
					Eigen::Vector4d X(0.05, 0, 0, 1);
					Eigen::Vector4d Y(0, 0.05, 0, 1);
					Eigen::Vector4d Z(0, 0, 0.05, 1);
					Eigen::Vector4d O(0, 0, 0, 1);
					X = view_pose_world * X;
					Y = view_pose_world * Y;
					Z = view_pose_world * Z;
					O = view_pose_world * O;
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(j));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(j));
					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(j));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(j));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(j));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(j));
				}
				while (!viewer->wasStopped())
				{
					viewer->spinOnce(100);
					boost::this_thread::sleep(boost::posix_time::microseconds(100000));
				}
			}
			cout << "labed "<< i <<" getted with executed time " << clock() - now_time << " ms." << endl;
		}
		return 0;
	}

	~SC_NBV_Labeler() {
		delete view_space;
		delete percept;
	}
};

int reconstruction_from_gt() {
	// Load the already stored point cloud from a file (without RGB data)
	pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	std::string cloud_filename = "D:/Programfiles/Myscvp/SCVPNet/log/merged_cloud_filtered085_1.pcd";

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_filename, *stored_cloud) == -1) {
		std::cerr << "Couldn't read file: " << cloud_filename << std::endl;
		return 1;
	}

	// Validate the loaded point cloud
	std::cout << "Loaded " << stored_cloud->points.size() << " points from " << cloud_filename << std::endl;

	// Perform normal estimation on the loaded point cloud
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	tree->setInputCloud(stored_cloud);
	ne.setInputCloud(stored_cloud);
	ne.setSearchMethod(tree);
	ne.setKSearch(20); // Set the number of nearest neighbors for normal estimation
	ne.compute(*normals);

	// Combine the point cloud and its normals
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::concatenateFields(*stored_cloud, *normals, *cloud_with_normals);

	// Ensure the sizes match
	if (cloud_with_normals->points.size() != stored_cloud->points.size()) {
		std::cerr << "Size mismatch between cloud and normals." << std::endl;
		return 1;
	}

	// Poisson reconstruction
	pcl::Poisson<pcl::PointXYZRGBNormal> pn;
	pn.setDegree(2);  // Change degree to avoid B-spline up-sampling error
	pn.setDepth(8);  // Set the maximum depth of the octree
	pn.setIsoDivide(5);
	pn.setSamplesPerNode(5);  // Set the minimum number of points per octree node
	pn.setScale(1.0f);        // Set the scale of the cube (float type)
	pn.setSolverDivide(5);
	pn.setConfidence(false);
	pn.setManifold(false);
	pn.setOutputPolygons(false);

	pcl::PolygonMesh triangles;
	pn.setInputCloud(cloud_with_normals);
	pn.performReconstruction(triangles);

	// Save the reconstructed model to a file
	pcl::io::savePLYFile("D:/Programfiles/Myscvp/SCVPNet/log/reconstructed_model_filtered_085_1.ply", triangles);

	// Visualize the reconstructed model
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Reconstructed Model Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(triangles, "mesh");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer->spin();

	return 0;
}

int reconstruction_gpt() {
	// Load the already stored point cloud from a file (without RGB data)
	pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	std::string cloud_filename = "D:/Programfiles/Myscvp/SCVPNet/log/merged_cloud_filtered085_1.pcd";

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_filename, *stored_cloud) == -1) {
		std::cerr << "Couldn't read file: " << cloud_filename << std::endl;
		return 1;
	}

	// Validate the loaded point cloud
	std::cout << "Loaded " << stored_cloud->points.size() << " points from " << cloud_filename << std::endl;

	// Perform normal estimation on the loaded point cloud
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_xyz->setInputCloud(stored_cloud);
	ne.setInputCloud(stored_cloud);
	ne.setSearchMethod(tree_xyz);
	ne.setKSearch(20); // Set the number of nearest neighbors for normal estimation
	ne.compute(*normals);

	// Combine the point cloud and its normals
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::concatenateFields(*stored_cloud, *normals, *cloud_with_normals);

	// Ensure the sizes match
	if (cloud_with_normals->points.size() != stored_cloud->points.size()) {
		std::cerr << "Size mismatch between cloud and normals." << std::endl;
		return 1;
	}

	// Use KdTree for the point cloud with normals
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
	tree->setInputCloud(cloud_with_normals);

	// Greedy Projection Triangulation
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	gp3.setSearchRadius(0.05); // Set the maximum distance between connected points (larger values fill more gaps)
	gp3.setMu(3); // Maximum allowable deviation from the surface
	gp3.setMaximumNearestNeighbors(200); // Maximum number of nearest neighbors to consider
	gp3.setMaximumSurfaceAngle(M_PI / 4); // Max surface angle (in radians) between connected points
	gp3.setMinimumAngle(M_PI / 18); // Minimum angle (in radians) for triangles
	gp3.setMaximumAngle(2 * M_PI / 3); // Maximum angle (in radians) for triangles
	gp3.setNormalConsistency(true); // Set to true if normals are consistent
	//gp3.setSamplesPerNode(8);

	pcl::PolygonMesh triangles;
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree);
	gp3.reconstruct(triangles);

	// Save the reconstructed model to a file
	pcl::io::savePLYFile("D:/Programfiles/Myscvp/SCVPNet/log/reconstructed_model_greedy_filtered_085_1.ply", triangles);

	// Visualize the reconstructed model
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Reconstructed Model Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(triangles, "mesh");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer->spin();

	return 0;
}

int reconstruction_gpt_with_upsampling() {
	// Load the already stored point cloud from a file (without RGB data)
	pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	std::string cloud_filename = "D:/Programfiles/Myscvp/SCVPNet/log/merged_cloud_filtered085_1.pcd";

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_filename, *stored_cloud) == -1) {
		std::cerr << "Couldn't read file: " << cloud_filename << std::endl;
		return 1;
	}

	// Validate the loaded point cloud
	std::cout << "Loaded " << stored_cloud->points.size() << " points from " << cloud_filename << std::endl;

	// Moving Least Squares (MLS) for upsampling and smoothing
	pcl::PointCloud<pcl::PointXYZ>::Ptr mls_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
	mls.setInputCloud(stored_cloud);
	mls.setSearchRadius(0.01);  // Set the radius for the MLS smoothing
	mls.setPolynomialFit(true);  // Enable polynomial fit
	mls.setComputeNormals(false); // We will compute normals later
	mls.process(*mls_cloud);

	std::cout << "After upsampling, the point cloud has: " << mls_cloud->points.size() << " points" << std::endl;

	// Perform normal estimation on the upsampled point cloud
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_xyz->setInputCloud(mls_cloud);
	ne.setInputCloud(mls_cloud);
	ne.setSearchMethod(tree_xyz);
	ne.setKSearch(20); // Set the number of nearest neighbors for normal estimation
	ne.compute(*normals);

	// Combine the upsampled point cloud and its normals
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::concatenateFields(*mls_cloud, *normals, *cloud_with_normals);

	// Ensure the sizes match
	if (cloud_with_normals->points.size() != mls_cloud->points.size()) {
		std::cerr << "Size mismatch between cloud and normals." << std::endl;
		return 1;
	}

	// Use KdTree for the point cloud with normals
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
	tree->setInputCloud(cloud_with_normals);

	// Greedy Projection Triangulation
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	gp3.setSearchRadius(0.02); // Set the maximum distance between connected points (larger values fill more gaps)
	gp3.setMu(2.5); // Maximum allowable deviation from the surface
	gp3.setMaximumNearestNeighbors(100); // Maximum number of nearest neighbors to consider
	gp3.setMaximumSurfaceAngle(M_PI / 4); // Max surface angle (in radians) between connected points
	gp3.setMinimumAngle(M_PI / 18); // Minimum angle (in radians) for triangles
	gp3.setMaximumAngle(2 * M_PI / 3); // Maximum angle (in radians) for triangles
	gp3.setNormalConsistency(false); // Set to true if normals are consistent

	pcl::PolygonMesh triangles;
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree);
	gp3.reconstruct(triangles);

	// Save the reconstructed model to a file
	pcl::io::savePLYFile("D:/Programfiles/Myscvp/SCVPNet/log/reconstructed_model_greedy_upsampled.ply", triangles);

	// Visualize the reconstructed model
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Reconstructed Model Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(triangles, "mesh");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer->spin();

	return 0;
}



atomic<bool> stop = false;		//���Ƴ������
Share_Data* share_data;			//����������ָ��
SC_NBV_Labeler* labeler;

#define DebugOne 0
#define TestAll 1
#define TrainAll 2
#define Coverage_rate 3
#define Transformer_Data 4
#define Simulation 5
#define Re_from_gt 6
#define Re_gpt 7
#define Re_gpt_up 8

int main(int argc, char* argv[]) {
	int mode = DebugOne;
	if (argc > 1) {
		mode = stoi(argv[1]);
	}

	ios::sync_with_stdio(false);


	if (mode == DebugOne)
	{
		//int part_num = 6;
		//cout << part_num << " thread, input index form 0:";
		//int index;
		//cin >> index;
		//NBV�滮�ڳ�ʼ��
		for (int i = 0; i < 6; i++)
		{
			//int i = index;
			for (int j = 0; j < 8; j++)
			{
				//��������ʼ��
				//j = index;
				share_data = new Share_Data("../DefaultConfiguration.yaml", "", "");
				labeler = new SC_NBV_Labeler(share_data, i, j);
				if (share_data->init_voxels < 300) continue;
				labeler->label();
				delete labeler;
				delete share_data;
			}
		}
	}

	else if (mode == Re_from_gt) {
		reconstruction_from_gt();
	}
	else if (mode == Re_gpt) {
		reconstruction_gpt();
	}
	else if (mode == Re_gpt_up) {
		reconstruction_gpt_with_upsampling();
	}
	else if (mode == Simulation) {
		share_data = new Share_Data("../DefaultConfiguration.yaml", "", "");
		labeler = new SC_NBV_Labeler(share_data);
		labeler->simulation();
		labeler->reconstruction_gpt();
		delete labeler;
		delete share_data;
	}
	else if (mode == Coverage_rate) {
		share_data = new Share_Data("../DefaultConfiguration.yaml", "", "");
		labeler = new SC_NBV_Labeler(share_data);
		labeler->compute_coverage();
		delete labeler;
		delete share_data;
	}
	else if (mode == Transformer_Data) {
		for (int i = 0; i < 6; i++)
		{
			//int i = index;
			for (int j = 0; j < 8; j++)
			{
				//��������ʼ��
				//j = index;
				share_data = new Share_Data("../DefaultConfiguration.yaml", "", "");
				labeler = new SC_NBV_Labeler(share_data, i, j);
				if (share_data->init_voxels < 300) continue;
				labeler->transformer_label();
				delete labeler;
				delete share_data;
			}
		}
	}
	else if (mode == TestAll){
		//���Լ�
		vector<string> names;
		names.push_back("Armadillo");
		names.push_back("Dragon");
		names.push_back("Stanford_Bunny");
		names.push_back("Happy_Buddha");
		names.push_back("Thai_Statue");
		names.push_back("Lucy");
		names.push_back("LM1");
		names.push_back("LM2");
		names.push_back("LM3");
		names.push_back("LM4");
		names.push_back("LM5");
		names.push_back("LM6");
		names.push_back("LM7");
		names.push_back("LM8");
		names.push_back("LM9");
		names.push_back("LM10");
		names.push_back("LM11");
		names.push_back("LM12");
		names.push_back("obj_000001");
		names.push_back("obj_000002");
		names.push_back("obj_000003");
		names.push_back("obj_000004");
		names.push_back("obj_000005");
		names.push_back("obj_000006");
		names.push_back("obj_000007");
		names.push_back("obj_000008");
		names.push_back("obj_000009");
		names.push_back("obj_000010");
		names.push_back("obj_000011");
		names.push_back("obj_000012");
		names.push_back("obj_000013");
		names.push_back("obj_000014");
		names.push_back("obj_000015");
		names.push_back("obj_000016");
		names.push_back("obj_000017");
		names.push_back("obj_000018");
		names.push_back("obj_000019");
		names.push_back("obj_000020");
		names.push_back("obj_000021");
		names.push_back("obj_000022");
		names.push_back("obj_000023");
		names.push_back("obj_000024");
		names.push_back("obj_000025");
		names.push_back("obj_000026");
		names.push_back("obj_000027");
		names.push_back("obj_000028");
		names.push_back("obj_000029");
		names.push_back("obj_000030");
		names.push_back("obj_000031");
		//����
		//int part_num = 4;
		//cout << part_num << " thread, input index form 0:";
		//int index;
		//cin >> index;
		for (int i = 0; i < names.size(); i++) {
		//for (int i = names.size() / part_num * index; i < names.size() / part_num * (index + 1); i++) {
			for (int j = 0; j < 6; j++){
				for (int k = 0; k < 8; k++) {
					//��������ʼ��
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], "");
					//NBV�滮�ڳ�ʼ��
					labeler = new SC_NBV_Labeler(share_data, j, k);
					if (share_data->init_voxels < 300) continue;
					labeler->label();
					delete labeler;
					delete share_data;
				}
			}
		}
	}
	else if (mode == TrainAll) {
		//���Լ�
		double now_time = clock();
		/*
		//remove_wrong
		vector<string> allPath;
		ifstream f_repath_in("../reneed_path.txt");
		string retemp_path;
		while (f_repath_in >> retemp_path) allPath.push_back(retemp_path);
		for (int i = 0; i < allPath.size(); i++)
		{
			string path = allPath[i];
			for (int i = 0; i < path.size(); i++) {
				if (path[i] == '\\') path[i] = '/';
			}
			string save_path = "../SC_label_data/" + path;
			cout << save_path << endl;
			remove((save_path + "/grid.txt").c_str());
			remove((save_path + "/view_ids.txt").c_str());
			cout << _rmdir(save_path.c_str()) << endl;
		}
		*/
		
		/*
		//check ans
		string shape_net_path = "D:\\Software\\PC-NBV\\models\\ShapeNetCore.v1";
		vector<string> allPath = getFilesList(shape_net_path);
		vector<string> reneedPath;
		for (int i = 0; i < allPath.size(); i++)
		{
			string path = allPath[i];
			for (int i = 0; i < path.size(); i++) {
				if (path[i] == '\\') path[i] = '/';
			}
			string save_path = "../SC_label_data/" + path;
			//cout << save_path << endl;
			ifstream f_grid(save_path + "/grid.txt");
			ifstream f_view(save_path + "/view_ids.txt");
			if (!f_grid.is_open() || !f_view.is_open()) {
				continue;
			}
			string line;
			int line_num = 0;
			while (getline(f_grid, line)) line_num++;
			int temp_index;
			int cnt_view = 0;
			while (f_view >> temp_index) cnt_view++;
			//cout << allPath[i] << endl;
			//cout << line_num << " , " << cnt_view << endl;
			if (line_num != 64000 || cnt_view <= 0) reneedPath.push_back(allPath[i]);
			if (i % 100 == 99) cout << i + 1 << " models checked. " << reneedPath.size() << " needed found." << endl;
		}
		ofstream f_repath_out("../reneed_path.txt");
		for (int i = 0; i < reneedPath.size(); i++) {
			f_repath_out << reneedPath[i] << endl;
		}
		*/
		/*
		//multi_reprocess����
		vector<string> reneedPath;
		ifstream f_repath_in("../reneed_path.txt");
		string retemp_path;
		while (f_repath_in >> retemp_path) reneedPath.push_back(retemp_path);
		int repart_num = 8;
		cout << repart_num << " thread, input index form 0:";
		int reindex;
		cin >> reindex;
		for (int i = reneedPath.size() / repart_num * reindex; i < reneedPath.size() / repart_num * (reindex + 1); i++){
			share_data = new Share_Data("../DefaultConfiguration.yaml", "", reneedPath[i]);
			thread runner(get_run);
			runner.join();
			delete share_data;
		}
		return 1;
		*/

		//find path
		string shape_net_path = "D:\\Software\\PC-NBV\\models\\ShapeNetCore.v1";
		vector<string> allPath = getFilesList(shape_net_path);
		cout << allPath.size() << " file path readed with " << clock() - now_time << "ms." << endl;
		/*
		//get statics
		vector<int> num_of_case,sum_num_of_case;
		num_of_case.resize(64);
		sum_num_of_case.resize(64);
		for (int i = 0; i < 64; i++) {
			num_of_case[i] = sum_num_of_case[i] = 0;
		}
		for (int i = 0; i < allPath.size(); i++) {
			string path = allPath[i];
			for (int i = 0; i < path.size(); i++) {
				if (path[i] == '\\') path[i] = '/';
			}
			string save_path = "../SC_label_data/" + path;
			ifstream f_view(save_path + "/view_ids.txt");
			if (f_view.is_open()) {
				int cnt_view = 0;
				int temp_index;
				while (f_view >> temp_index) cnt_view++;
				num_of_case[cnt_view]++;
				//cout << save_path << "  : " << cnt_view << endl;
			}
		}
		sum_num_of_case[0] = num_of_case[0];
		cout<< 0 << '\t' << num_of_case[0] << '\t' << sum_num_of_case[0] << endl;
		for (int i = 1; i < 64; i++) {
			sum_num_of_case[i] = sum_num_of_case[i-1] + num_of_case[i];
			//cout << "view num " << i << ": " << num_of_case[i] << " ,sum: " << sum_num_of_case[i] << endl;
			cout << i << '\t' << num_of_case[i] << '\t' << sum_num_of_case[i] << endl;
		}
		return 1;
		*/
		//check path
		now_time = clock();
		vector<string> needPath;
		for (int i = 0; i < allPath.size(); i++) 
		{
			string path = allPath[i];
			for (int i = 0; i < path.size(); i++) {
				if (path[i] == '\\') path[i] = '/';
			}
			string save_path = "../Pan_SC_label_data/" + path;
			//cout << save_path << endl;
			ifstream f_grid(save_path +"/grid.txt");
			ifstream f_view(save_path +"/view_ids.txt");
			if (!f_grid.is_open() || !f_view.is_open()) {
				share_data = new Share_Data("../DefaultConfiguration.yaml", "", allPath[i]);
				labeler = new SC_NBV_Labeler(share_data);
				if (share_data->init_voxels >= 300)	needPath.push_back(allPath[i]);
				delete labeler;
				delete share_data;
			}
		}
		//out path
		ofstream f_path_out("../need_path.txt");
		for (int i = 0; i < needPath.size(); i++) {
			f_path_out << needPath[i] << endl;
		}
		
		/*
		//read path
		vector<string> needPath;
		ifstream f_path_in("../need_path.txt");
		string temp_path;
		while (f_path_in >> temp_path) needPath.push_back(temp_path);
		*/

		cout << needPath.size() << " needed file path getted with " << clock() - now_time << "ms." << endl;

		//multi_process����
		int part_num = 8;
		cout << part_num << " thread, input index form 0:";
		int index;
		cin >> index;
		
		//for (int i = 0; i < needPath.size(); i++) {
		for (int i = needPath.size() / part_num * index; i < needPath.size() / part_num * (index + 1); i++){
			//NBV�滮�ڳ�ʼ��
			for (int i = 0; i < 6; i++) {
				for (int j = 0; j < 8; j++) {
					//��������ʼ��
					share_data = new Share_Data("../DefaultConfiguration.yaml", "", needPath[i]);
					labeler = new SC_NBV_Labeler(share_data, i, j);
					if (share_data->init_voxels < 300) continue;
					labeler->label();
					delete labeler;
					delete share_data;
				}
			}
		}
	}
	cout << "System over." << endl;
	return 0;
}
