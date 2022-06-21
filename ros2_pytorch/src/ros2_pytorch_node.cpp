#include "ros2_pytorch.hpp"





int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::cout << "Starting pytorch node" << std::endl;
    rclcpp::spin(std::make_shared<PyTorchNode>());
    rclcpp::shutdown();
    return 0;
}