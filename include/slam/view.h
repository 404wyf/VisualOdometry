#ifndef View_H
#define View_H

#include "common.h"
   class View{
    public:
    void draw_camera(const Eigen::Isometry3d& pose);
    void show_trajectory(vector<Eigen::Isometry3d>& poses, vector<Point3f>& current_frame_points);
    void draw_features(const Mat& img, const vector<KeyPoint>& keypoints, const vector<KeyPoint>& new_keypoints) ;
    vector<KeyPoint> get_new_keypoints(const vector<KeyPoint>& current_keypoints, const vector<KeyPoint>& previous_keypoints);
   
   };
  
#endif