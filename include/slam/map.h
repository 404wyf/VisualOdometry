#ifndef MAP_H
#define MAP_H

#include "common.h"

namespace myslam{
   
        extern vector<Eigen::Isometry3d> C_W_poses; // 存储位姿
        extern vector<Point3f> current_W_frame_points; // 当前帧的特征点的世界3D坐标
        extern Mat K ; // 相机内参
  
}


#endif