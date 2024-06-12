#include "slam/view.h"
#include "slam/image_process.h" 
#include "slam/map.h"
 using namespace myslam;

int main(int argc, char** argv) {
    View view=View();
    ImageProcess img_pro=ImageProcess();
    // 使用 lambda 表达式来调用 img_pro 的成员函数
    thread image_thread(&ImageProcess::image_process, &img_pro, ref(view));
    
    while (true) {
        view.show_trajectory(myslam::poses, myslam:: current_frame_points);
    }

    image_thread.join();
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

