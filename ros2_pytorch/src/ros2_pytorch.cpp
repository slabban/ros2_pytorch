
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

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "cv_bridge/cv_bridge.h"

#include "ros2_pytorch.hpp"
using std::placeholders::_1;

class PyTorchNode : public rclcpp::Node
{
public:
    PyTorchNode()
    : Node("pytorch_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 10, std::bind(&PyTorchNode::topic_callback, this, _1));
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic_out", 1);

        this->declare_parameter("GPU", 0);
        int cuda = this->get_parameter("GPU").as_int();

        if(cuda>0){
            device = at::kCUDA;
        }

    }
    
private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        
        RCLCPP_INFO(this->get_logger(), "We got an image");
    
        
        std::shared_ptr<cv_bridge::CvImage> image_ = cv_bridge::toCvCopy(msg, "bgr8");

        at::TensorOptions options(at::ScalarType::Byte);
        std::vector<int64_t> sizes = {1, 3, msg->height, msg->width};
        at::Tensor tensor_image = torch::from_blob(image_->image.data, at::IntList(sizes), options);
        
        std::vector<torch::jit::IValue> inputs;

        module_.to(device);
        
        tensor_image = tensor_image.to(device).toType(at::kFloat);
        inputs.emplace_back(tensor_image);


        // Execute the model and turn its output into a tensor.
        at::Tensor output = module_.forward(inputs).toTensor();
        
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
        
        auto message = std_msgs::msg::String();
        message.data = "Prediction done";
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);

        
    }
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    torch::jit::script::Module module_ = torch::jit::load("/home/ros2_ws/src/ros2_pytorch/model.pt");
    c10::DeviceType device = at::kCPU;
    
    cv_bridge::CvImage img_bridge;

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    std::cout << "Starting pytorch node" << std::endl;
    rclcpp::spin(std::make_shared<PyTorchNode>());
    rclcpp::shutdown();
    return 0;
}
