//
//  ros2_pytorch.hpp
//
//  Created by Samer Labban on 07/13/2022 
//  Copyright © 2022 Samer Labban. All rights reserved.
//

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <iostream>

#include <torch/script.h> // One-stop header.
#include <ATen/ATen.h>


#include "rclcpp/rclcpp.hpp"
#include "rcpputils/asserts.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>



class PyTorchNode : public rclcpp::Node
{
public:
    PyTorchNode();

    
private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    torch::jit::script::Module module_;
    c10::DeviceType device = at::kCPU;
    

};