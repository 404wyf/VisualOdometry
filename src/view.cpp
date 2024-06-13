#include"slam/common.h"
#include"slam/view.h"


 void View:: draw_camera(const Eigen::Isometry3d& pose) {
    const float w = 0.5;
    const float h = 0.25;
    const float z = 0.6;

    glPushMatrix();
    Eigen::Matrix4d m = pose.matrix();
    glMultMatrixd(m.data());

    glLineWidth(2.0);
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}


// 使用Pangolin进行可视化
void View:: show_trajectory(vector<Eigen::Isometry3d>& poses, vector<Point3f>& current_frame_points) 
{
    if (poses.empty()) return;

    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置背景颜色为白色

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 200, 200, 512, 389, 0.2, 1000), // 调整焦距参数，放大视角
        pangolin::ModelViewLookAt(0, -5, -5, 0, 0, 0, pangolin::AxisZ) // 调整相机位置，使其离场景更近
    );

    const int UI_WIDTH = 180;

    // 右侧用于显示视口
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    // 左侧用于创建控制面板
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    // 创建控制面板的控件对象，pangolin中
    pangolin::Var<bool> A_Checkbox("ui.Follow Camera", true, true); // 相机
    pangolin::Var<bool> B_Checkbox("ui.Show Points", true, true); // 特征点
    pangolin::Var<bool> C_Checkbox("ui.Show KeyFrames", true, true); // 位姿

    bool should_quit = false;
    while (!should_quit) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glLineWidth(2);

        if (C_Checkbox) {
            // 绘制轨迹
            for (size_t i = 0; i < poses.size() - 1; i++) {
                glColor3f(1.0 - i / (float)poses.size(), 0.0f, i / (float)poses.size());
                glBegin(GL_LINES);
                auto p1 = poses[i];
                auto p2 = poses[i + 1];
                glVertex3f(p1(0, 3), p1(1, 3), p1(2, 3));
                glVertex3f(p2(0, 3), p2(1, 3), p2(2, 3));
                glEnd();
            }
        }

        if (B_Checkbox) {
            // 绘制当前帧的特征点
            glPointSize(5);
            glBegin(GL_POINTS);
            glColor3f(1.0, 0.0, 0.0); // 红色
            for (const auto& point : current_frame_points) {
                glVertex3f(point.x, point.y, point.z);
            }
            glEnd();
        }

        if (A_Checkbox) {
            // 绘制当前相机位置
            draw_camera(poses.back());

            // 更新视角位置以跟随相机但保持第三人称视角
            Eigen::Isometry3d last_pose = poses.back();
            Eigen::Vector3d camera_position = last_pose.translation();
            Eigen::Vector3d forward = last_pose.rotation() * Eigen::Vector3d(0, 0, 1);
            Eigen::Vector3d up = last_pose.rotation() * Eigen::Vector3d(0, -1, 0);

            // 设置一个相对相机后方的位置作为观察点，使视角更大
            Eigen::Vector3d third_person_position = camera_position - forward * 2.0 + up * 4.0; // 调整此参数以改变相机视角
            Eigen::Vector3d look_at_point = camera_position; // 看向相机位置

            //更新相机位置，follow camera
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                third_person_position.x(), third_person_position.y(), third_person_position.z(),
                look_at_point.x(), look_at_point.y(), look_at_point.z(),
                up.x(), up.y(), up.z()
            ));
        }

        pangolin::FinishFrame();
        // 检查是否应退出循环
        should_quit = pangolin::ShouldQuit();
    }
}


// 绘制特征点和新增特征点
void View:: draw_features(const Mat& img, const vector<KeyPoint>& keypoints, const vector<KeyPoint>& new_keypoints) {
    Mat img_with_features;
    img.copyTo(img_with_features); // 直接复制原图像

    for (const auto& kp : keypoints) {
        circle(img_with_features, kp.pt, 2, Scalar(0, 255, 0), -1); // 绿色
    }

    for (const auto& kp : new_keypoints) {
        circle(img_with_features, kp.pt, 2, Scalar(0, 0, 255), -1); // 红色
    }

    imshow("Features", img_with_features);
    waitKey(30); // 等待用户按键以便查看图像
}

// 获取新增的特征点
vector<KeyPoint> View:: get_new_keypoints(const vector<KeyPoint>& current_keypoints, const vector<KeyPoint>& previous_keypoints) {
    vector<KeyPoint> new_keypoints;
    for (const auto& kp : current_keypoints) {
        bool is_new = true;
        for (const auto& pkp : previous_keypoints) {
            if (cv::norm(kp.pt - pkp.pt) < 2.0) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            new_keypoints.push_back(kp);
        }
    }
    return new_keypoints;
}
 