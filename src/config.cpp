#include "slam/config.h"


ConfigReader::ConfigReader(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << std::endl;
        throw std::runtime_error("Could not open config file");
    }
    fs["dataset_dir"] >> dataset_dir;
    fs.release();
}

std::string ConfigReader::getDatasetDir() const {
    return dataset_dir;
}