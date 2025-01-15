#pragma once
#include <unordered_set>
#include <algorithm>
#include <iostream>
// #include <direct.h>
#include <sys/types.h>
#include <filesystem>
#include <fstream>  
#include <stdio.h>  
#include <string>  
#include <vector> 
#include <thread>
#include <chrono>
#include <atomic>
#include <ctime> 
#include <cmath>
#include <mutex>
#include <map>
#include <set>
// #include <io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

using namespace std;

/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
typedef enum rs2_distortion
{
	RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
	RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
	RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
	RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
	RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
	RS2_DISTORTION_KANNALA_BRANDT4, /**< Four parameter Kannala Brandt distortion model */
	RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_distortion;

/** \brief Video stream intrinsics. */
typedef struct rs2_intrinsics
{
	int           width;     /**< Width of the image in pixels */
	int           height;    /**< Height of the image in pixels */
	float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
	float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
	float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
	float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
	rs2_distortion model;    /**< Distortion model of the image */
	float         coeffs[5]; /**< Distortion coefficients */
} rs2_intrinsics;

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics* intrin, const float point[3])
{
	float x = point[0] / point[2], y = point[1] / point[2];

	if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
		(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
	{

		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		x *= f;
		y *= f;
		float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = dx;
		y = dy;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
		x *= rd / r;
		y *= rd / r;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float theta = atan(r);
		float theta2 = theta * theta;
		float series = 1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])));
		float rd = theta * series;
		x *= rd / r;
		y *= rd / r;
	}

	pixel[0] = x * intrin->fx + intrin->ppx;
	pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}

		float theta = rd;
		float theta2 = rd * rd;
		for (int i = 0; i < 4; i++)
		{
			float f = theta * (1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])))) - rd;
			if (abs(f) < FLT_EPSILON)
			{
				break;
			}
			float df = 1 + theta2 * (3 * intrin->coeffs[0] + theta2 * (5 * intrin->coeffs[1] + theta2 * (7 * intrin->coeffs[2] + 9 * theta2 * intrin->coeffs[3])));
			theta -= f / df;
			theta2 = theta * theta;
		}
		float r = tan(theta);
		x *= r / rd;
		y *= r / rd;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}
		float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
		x *= r / rd;
		y *= r / rd;
	}

	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}

#define NBVSampleMethod 0
#define LongTailSampleMethod 1

class Share_Data
{
public:
	//�ɱ��������
	string pcd_file_path;
	string yaml_file_path;
	string name_of_pcd;
	string nbv_net_path;

	int num_of_views;					//һ�β����ӵ����
	double cost_weight;
	rs2_intrinsics color_intrinsics;
	double depth_scale;

	//���в���
	int process_cnt;					//���̱��
	atomic<double> pre_clock;			//ϵͳʱ��
	atomic<bool> over;					//�����Ƿ����
	bool show;
	int num_of_max_iteration;

	//��������
	atomic<int> vaild_clouds;
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;							//������
	vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* > voxels;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_truth;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final;
	bool move_wait;
	map<string, double> mp_scale;

	//�˲��ͼ
	octomap::ColorOcTree* octo_model;
	//octomap::ColorOcTree* cloud_model;
	octomap::ColorOcTree* ground_truth_model;
	octomap::ColorOcTree* GT_sample;
	double octomap_resolution;
	double ground_truth_resolution;
	double map_size;
	double p_unknown_upper_bound; //! Upper bound for voxels to still be considered uncertain. Default: 0.97.
	double p_unknown_lower_bound; //! Lower bound for voxels to still be considered uncertain. Default: 0.12.
	
	//�����ռ����ӵ�ռ�
	atomic<bool> now_view_space_processed;
	atomic<bool> now_views_infromation_processed;
	atomic<bool> move_on;

	Eigen::Matrix4d now_camera_pose_world;
	Eigen::Vector3d object_center_world;		//��������
	double predicted_size;						//����BBX�뾶

	int method_of_IG;
	pcl::visualization::PCLVisualizer::Ptr viewer;

	double stop_thresh_map;
	double stop_thresh_view;

	double skip_coefficient;

	double sum_local_information;
	double sum_global_information;

	double sum_robot_cost;
	double camera_to_object_dis;
	bool robot_cost_negtive;

	int num_of_max_flow_node;
	double interesting_threshold;

	double see_threshold;
	double need_threshold;

	int init_voxels;     //����voxel����
	int full_voxels;     //����voxel����
	int voxels_in_BBX;   //��ͼvoxel����
	double init_entropy; //��ͼ��Ϣ��

	string pre_path;
	string gt_path;
	string save_path;
	string save_path_nbvnet;
	string save_path_pcnbv;

	vector<vector<double>> pt_sphere;
	double pt_norm;
	double min_z_table;

	int need_case_1;
	int sampling_method;
	vector<long long> view_cases;

	Share_Data(string _config_file_path, string test_name = "", int _sampling_method = -1,int _need_case_1 = -1)
	{
		process_cnt = -1;
		yaml_file_path = _config_file_path;
		//��ȡyaml�ļ�
		cv::FileStorage fs;
		fs.open(yaml_file_path, cv::FileStorage::READ);
		fs["pre_path"] >> pre_path;
		fs["model_path"] >> pcd_file_path;
		fs["name_of_pcd"] >> name_of_pcd;
		fs["method_of_IG"] >> method_of_IG;
		fs["octomap_resolution"] >> octomap_resolution;
		fs["ground_truth_resolution"] >> ground_truth_resolution;
		fs["sampling_method"] >> sampling_method;
		fs["need_case_1"] >> need_case_1;
		fs["num_of_max_iteration"] >> num_of_max_iteration;
		fs["show"] >> show;
		fs["move_wait"] >> move_wait;
		fs["nbv_net_path"] >> nbv_net_path;
		fs["p_unknown_upper_bound"] >> p_unknown_upper_bound;
		fs["p_unknown_lower_bound"] >> p_unknown_lower_bound;
		fs["num_of_views"] >> num_of_views;
		fs["cost_weight"] >> cost_weight;
		fs["robot_cost_negtive"] >> robot_cost_negtive;
		fs["skip_coefficient"] >> skip_coefficient;
		fs["num_of_max_flow_node"] >> num_of_max_flow_node;
		fs["interesting_threshold"] >> interesting_threshold;
		fs["see_threshold"] >> see_threshold;
		fs["need_threshold"] >> need_threshold;
		fs["color_width"] >> color_intrinsics.width;
		fs["color_height"] >> color_intrinsics.height;
		fs["color_fx"] >> color_intrinsics.fx;
		fs["color_fy"] >> color_intrinsics.fy;
		fs["color_ppx"] >> color_intrinsics.ppx;
		fs["color_ppy"] >> color_intrinsics.ppy;
		fs["color_model"] >> color_intrinsics.model;
		fs["color_k1"] >> color_intrinsics.coeffs[0];
		fs["color_k2"] >> color_intrinsics.coeffs[1];
		fs["color_k3"] >> color_intrinsics.coeffs[2];
		fs["color_p1"] >> color_intrinsics.coeffs[3];
		fs["color_p2"] >> color_intrinsics.coeffs[4];
		fs["depth_scale"] >> depth_scale;
		fs.release();
		if (test_name != "") name_of_pcd = test_name;
		if (_sampling_method != -1) sampling_method = _sampling_method;
		if (_need_case_1 != -1) need_case_1 = _need_case_1;
		//��ȡת����ģ�͵�pcd�ļ�
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp_pcd(new pcl::PointCloud<pcl::PointXYZ>);
		cloud_pcd = temp_pcd;
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path + name_of_pcd + ".pcd", *cloud_pcd) == -1) {
			cout << "Can not read 3d model file. Check." << endl;
		}
		//��תZ������
		set<string> names_rotate;
		names_rotate.insert("Armadillo");
		names_rotate.insert("Asian_Dragon");
		names_rotate.insert("Dragon");
		names_rotate.insert("Stanford_Bunny");
		names_rotate.insert("Happy_Buddha");
		names_rotate.insert("Thai_Statue");
		//names_rotate.insert("Lucy");
		if (names_rotate.count(name_of_pcd)) {
			pcl::transformPointCloud(*cloud_pcd, *cloud_pcd, get_toward_pose(4));
		}
		//������С����ˮ�ܱ���
		mp_scale["Armadillo"] = 0.02;
		mp_scale["Asian_Dragon"] = 0.05;
		mp_scale["Dragon"] = 0.05;
		mp_scale["Stanford_Bunny"] = 0.04;
		mp_scale["Happy_Buddha"] = 0.07;
		mp_scale["Thai_Statue"] = 0.25;
		mp_scale["Lucy"] = 1.39;
		mp_scale["LM11"] = 0.03;
		mp_scale["LM12"] = 0.04;
		mp_scale["obj_000001"] = 0.02;
		mp_scale["obj_000002"] = 0.06;
		mp_scale["obj_000004"] = 0.02;
		mp_scale["obj_000005"] = 0.02;
		mp_scale["obj_000007"] = 0.05;
		mp_scale["obj_000008"] = 0.03;
		mp_scale["obj_000009"] = 0.03;
		mp_scale["obj_000010"] = 0.03;
		mp_scale["obj_000011"] = 0.06;
		mp_scale["obj_000012"] = 0.02;
		mp_scale["obj_000018"] = 0.02;
		mp_scale["obj_000020"] = 0.08;
		mp_scale["obj_000021"] = 0.03;
		mp_scale["obj_000022"] = 0.02;
		mp_scale["obj_000023"] = 0.04;
		mp_scale["obj_000024"] = 0.05;
		mp_scale["obj_000025"] = 0.05;
		mp_scale["obj_000026"] = 0.01;
		mp_scale["obj_000027"] = 0.09;
		mp_scale["obj_000028"] = 0.13;
		mp_scale["obj_000029"] = 0.02;
		mp_scale["obj_000030"] = 0.18;

		//octo_model = new octomap::ColorOcTree(octomap_resolution);
		//octo_model->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//octo_model->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//octo_model->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//octo_model->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		//octo_model->setOccupancyThres(0.5);	//���ýڵ�ռ����ֵ����ʼ0.5
		ground_truth_model = new octomap::ColorOcTree(ground_truth_resolution);
		//ground_truth_model->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//ground_truth_model->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//ground_truth_model->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//ground_truth_model->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		//GT_sample = new octomap::ColorOcTree(octomap_resolution);
		//GT_sample->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//GT_sample->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//GT_sample->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//GT_sample->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		/*cloud_model = new octomap::ColorOcTree(ground_truth_resolution);
		//cloud_model->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//cloud_model->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//cloud_model->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//cloud_model->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		//cloud_model->setOccupancyThres(0.5);	//���ýڵ�ռ����ֵ����ʼ0.5*/
		if (num_of_max_flow_node == -1) num_of_max_flow_node = num_of_views;
		now_camera_pose_world = Eigen::Matrix4d::Identity(4, 4);
		over = false;
		pre_clock = clock();
		vaild_clouds = 0;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_final = temp;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_gt(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_ground_truth = temp_gt;
		//path
		//pre_path = "D:/Data/LongTail_MA-SCVP/";
		gt_path = pre_path + "NBV_GT_label/";
		pre_path += to_string(need_case_1) + "/";
		if (sampling_method == NBVSampleMethod) {
			save_path = pre_path + "MASCVP_NBVSample/" ;
			save_path_nbvnet = pre_path + "NBVNET_NBVSample/";
			save_path_pcnbv = pre_path + "PCNBV_NBVSample/";
		}
		else if (sampling_method == LongTailSampleMethod) {
			save_path = pre_path + "MASCVP_LongTailSample/";
			save_path_nbvnet = pre_path + "NBVNET_LongTailSample/";
			save_path_pcnbv = pre_path + "PCNBV_LongTailSample/";
		}
		cout << "save_path_mascvp is: " << save_path << endl;
		cout << "save_path_nbvnet is: " << save_path_nbvnet << endl;
		cout << "save_path_pcnbv is: " << save_path_pcnbv << endl;
		srand(clock());
		//read viewspace
		ifstream fin_sphere("../view_space.txt");
		pt_sphere.resize(num_of_views);
		for (int i = 0; i < num_of_views; i++) {
			pt_sphere[i].resize(3);
			for (int j = 0; j < 3; j++) {
				fin_sphere >> pt_sphere[i][j];
				//cout << pt_sphere[i][j] << " ??? " << endl;
			}
		}
		Eigen::Vector3d pt0(pt_sphere[0][0], pt_sphere[0][1], pt_sphere[0][2]);
		pt_norm = pt0.norm();
		//read view cases
		ifstream fin_view_cases;
		fin_view_cases.open("../test_view_cases.txt");
		long long cas;
		while (fin_view_cases >> cas) {
			view_cases.push_back(cas);
		}
		//cout << "test view case num is: " << view_cases.size() << endl;
	}

	~Share_Data() {
		delete octo_model;
		delete ground_truth_model;
		delete GT_sample;
		cloud_pcd->~PointCloud();
		cloud_final->~PointCloud();
		cloud_ground_truth->~PointCloud();
		for (int i = 0; i < clouds.size(); i++)
			clouds[i]->~PointCloud();
		for (int i = 0; i < voxels.size(); i++)
			delete voxels[i];
	}

	Eigen::Matrix4d get_toward_pose(int toward_state)
	{
		Eigen::Matrix4d pose(Eigen::Matrix4d::Identity(4, 4));
		switch (toward_state) {
			case 0://z<->z
				return pose;
			case 1://z<->-z
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = 0; pose(2, 2) = -1; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 2://z<->x
				pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = 1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 3://z<->-x
				pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = -1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 4://z<->y
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = 1; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 5://z<->-y
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = -1; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
		}
		return pose;
	}

	double out_clock()
	{   //������ʱ��������ʱ��
		double now_clock = clock();
		double time_len = now_clock - pre_clock;
		pre_clock = now_clock;
		return time_len;
	}

	void access_directory(string cd)
	{   //���༶Ŀ¼���ļ����Ƿ���ڣ������ھʹ���
		string temp;
		for (int i = 0; i < cd.length(); i++)
			if (cd[i] == '/') {
				if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str(),0755);
				temp += cd[i];
			}
			else temp += cd[i];
		if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str(),0755);
	}

	void save_posetrans_to_disk(Eigen::Matrix4d& T, string cd, string name, int frames_cnt)
	{   //�����ת��������������
		std::stringstream pose_stream, path_stream;
		std::string pose_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		pose_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".txt";
		pose_stream >> pose_file;
		ofstream fout(pose_file);
		fout << T;
	}

	void save_octomap_log_to_disk(int voxels, double entropy, string cd, string name,int iterations)
	{
		std::stringstream log_stream, path_stream;
		std::string log_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		log_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << iterations << ".txt";
		log_stream >> log_file;
		ofstream fout(log_file);
		fout << voxels << " " << entropy << endl;
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name)
	{   //��ŵ������������̣��ٶȺ���������
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << save_path << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << save_path << cd << "/" << name << ".pcd";
		cloud_stream >> cloud_file;
		pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name, int frames_cnt)
	{   //��ŵ������������̣��ٶȺ���������
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".pcd";
		cloud_stream >> cloud_file;
		pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_octomap_to_disk(octomap::ColorOcTree* octo_model, string cd, string name)
	{   //��ŵ������������̣��ٶȺ���������
		std::stringstream octomap_stream, path_stream;
		std::string octomap_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		octomap_stream << "../data" << "_" << process_cnt << cd << "/" << name << ".ot";
		octomap_stream >> octomap_file;
		octo_model->write(octomap_file);
	}

};

inline double pow2(double x) {
	return x * x;
}

namespace fs = std::filesystem;
vector<string> getFilesList(const string& dir)
{
    vector<string> allPath;

    // 如果目标路径不存在或不是目录，则直接返回空向量
    if (!fs::exists(dir) || !fs::is_directory(dir))
    {
        cout << "Path not found or is not a directory: " << dir << endl;
        return allPath;
    }

    // 迭代该目录下的所有元素（文件 / 子目录）
    for (const auto& entry : fs::directory_iterator(dir))
    {
        const auto& path = entry.path();
        // 获取文件或子目录的名称（不包含路径）
        string filename = path.filename().string();

        // 判断是否是目录
        if (fs::is_directory(path))
        {
            // 跳过 "." 和 ".."
            if (filename == "." || filename == "..")
                continue;
            
            // 递归调用：如果是子目录，则继续遍历
            vector<string> tempPath = getFilesList(path.string());
            // 将子目录中的结果合并到 allPath
            allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());
        }
        else
        {
            // 如果是普通文件，检查文件名是否为 "model.pcd"
            if (filename == "model.pcd")
            {
                // 根据你的逻辑决定如何存储最终结果
                // 原代码中做了 dir.substr(1) 的处理，这里也可以相应移植
                if (dir.size() > 1)
                    allPath.push_back(dir.substr(1));
                else
                    allPath.push_back(dir);
            }
        }
    }

    return allPath;
}