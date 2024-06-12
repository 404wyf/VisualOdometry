#ifndef CONFIG_H
#define CONFIG_H

#include "common.h"

#include <string>

class ConfigReader {
public:
    // Constructor that takes the YAML file path
    ConfigReader(const std::string& filename);

    // Method to get the dataset directory
    std::string getDatasetDir() const;

private:
    std::string dataset_dir;
};

  
#endif