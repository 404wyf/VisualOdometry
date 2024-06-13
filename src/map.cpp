#include "slam/map.h"

namespace myslam
{
    vector<Eigen::Isometry3d> C_W_poses; // 存储位姿
    vector<Point3f> current_W_frame_points; // 当前帧的特征点的3D坐标
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 相机内参
} // namespace myslam
