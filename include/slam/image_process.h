#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include "common.h"
#include "view.h"
   class ImageProcess {
    public:
    Point2d pixel2cam(const Point2d& p, const Mat& K);
    void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2, std::vector<DMatch>& matches) ;
    Eigen::Isometry3d estimateTransform(const vector<Point3f>& pts1, const vector<Point3f>& pts2) ;
    int image_process(View& view);
   };
  
#endif