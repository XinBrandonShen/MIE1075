/**
 * @file mtc_graspnet_node.cpp
 * @brief MoveIt Task Constructor node with GraspNet integration (Multi-Grasp Version)
 *
 * This node implements pick and place using MTC with:
 * - Multiple GraspNet poses (top 5) tried using Alternatives container
 * - 6DOF poses only (ignoring gripper width)
 * - Center-of-object as fallback
 * - Different planners for different motion types
 * - Cost-based solution ranking across all grasp alternatives
 *
 * @author Brandon & Claude
 * @date December 2024
 */

#include <rclcpp/rclcpp.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/task_constructor/container.h>

#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <Eigen/Geometry>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

// Handling the GetPlanningScene service
#include "mycobot_mtc_pick_place_demo/get_planning_scene_client.h"

namespace mtc = moveit::task_constructor;

// ============================================================================
// Utility Functions
// ============================================================================

namespace {

Eigen::Isometry3d vectorToEigen(const std::vector<double>& values) {
  return Eigen::Translation3d(values[0], values[1], values[2]) *
         Eigen::AngleAxisd(values[3], Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(values[4], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(values[5], Eigen::Vector3d::UnitZ());
}

geometry_msgs::msg::Pose vectorToPose(const std::vector<double>& values) {
  return tf2::toMsg(vectorToEigen(values));
}

double extractJsonDouble(const std::string& json, const std::string& key) {
  size_t pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) return 0.0;
  pos = json.find(":", pos);
  if (pos == std::string::npos) return 0.0;
  size_t start = json.find_first_of("-0123456789.", pos);
  size_t end = json.find_first_not_of("-0123456789.eE+", start);
  if (start == std::string::npos) return 0.0;
  try {
    return std::stod(json.substr(start, end - start));
  } catch (...) {
    return 0.0;
  }
}

std::string extractJsonString(const std::string& json, const std::string& key) {
  size_t pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) return "";
  pos = json.find(":", pos);
  if (pos == std::string::npos) return "";
  size_t start = json.find("\"", pos) + 1;
  size_t end = json.find("\"", start);
  return json.substr(start, end - start);
}

int extractJsonInt(const std::string& json, const std::string& key) {
  size_t pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) return 0;
  pos = json.find(":", pos);
  if (pos == std::string::npos) return 0;
  size_t start = json.find_first_of("-0123456789", pos);
  size_t end = json.find_first_not_of("-0123456789", start);
  if (start == std::string::npos) return 0;
  try {
    return std::stoi(json.substr(start, end - start));
  } catch (...) {
    return 0;
  }
}

}  // namespace

// ============================================================================
// Grasp Types
// ============================================================================

enum class GraspSource { GRASPNET, CENTER, NONE };

struct GraspPoseInfo {
  geometry_msgs::msg::PoseStamped pose;
  double score = 0.0;
  int index = 0;
  GraspSource source = GraspSource::NONE;
  bool valid = false;
};

// ============================================================================
// MTCGraspNetNode Class
// ============================================================================

class MTCGraspNetNode : public rclcpp::Node
{
public:
  MTCGraspNetNode(const rclcpp::NodeOptions& options);

  void doTask();
  void setupPlanningScene();

private:
  mtc::Task task_;
  mtc::Task createTask(const std::vector<GraspPoseInfo>& grasp_poses);

  // Multiple grasp pose methods
  std::vector<GraspPoseInfo> getGraspNetPoses();
  GraspPoseInfo getCenterBasedPose();
  std::vector<GraspPoseInfo> getBestGraspPoses();
  bool triggerGraspNetPipeline();

  // Async service call helpers
  struct ServiceCallResult {
    bool success = false;
    std::string message;
    bool completed = false;
  };
  
  void callSaveDataService(std::shared_ptr<ServiceCallResult> result);
  void callGetMultipleGraspsService(std::shared_ptr<ServiceCallResult> result);

  // Scene data
  moveit_msgs::msg::PlanningSceneWorld scene_world_;
  sensor_msgs::msg::PointCloud2 full_cloud_;
  sensor_msgs::msg::Image rgb_image_;
  std::string target_object_id_;
  std::string support_surface_id_;
  bool service_success_ = false;

  // Service clients
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr graspnet_multi_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr save_data_client_;

  void updateObjectParameters(const moveit_msgs::msg::CollisionObject& collision_object);
  
  // Parse multiple grasps from JSON
  std::vector<GraspPoseInfo> parseMultipleGrasps(const std::string& json);
};

// ============================================================================
// Constructor
// ============================================================================

MTCGraspNetNode::MTCGraspNetNode(const rclcpp::NodeOptions& options)
  : Node("mtc_graspnet_node", options)
{
  auto declare_parameter = [this](const std::string& name, const auto& default_value, 
                                   const std::string& description = "") {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = description;
    if (!this->has_parameter(name)) {
      this->declare_parameter(name, default_value, descriptor);
    }
  };

  // General
  declare_parameter("execute", false, "Execute the planned task");
  declare_parameter("max_solutions", 10, "Max solutions to compute");
  declare_parameter("use_graspnet", true, "Use GraspNet for grasp poses");
  declare_parameter("fallback_to_center", true, "Fallback to center if GraspNet fails");
  declare_parameter("num_grasp_candidates", 5, "Number of grasp candidates to try");

  // Controllers
  declare_parameter("controller_names", 
    std::vector<std::string>{"arm_controller", "grip_action_controller"}, "Controllers");

  // Robot config
  declare_parameter("arm_group_name", "arm", "Arm group name");
  declare_parameter("gripper_group_name", "gripper", "Gripper group name");
  declare_parameter("gripper_frame", "link6_flange", "Gripper frame");
  declare_parameter("gripper_open_pose", "open", "Open pose name");
  declare_parameter("gripper_close_pose", "half_closed", "Close pose name");
  declare_parameter("arm_home_pose", "home", "Home pose name");
  declare_parameter("world_frame", "base_link", "World frame");

  // Object
  declare_parameter("object_name", "object", "Object name");
  declare_parameter("object_type", "cylinder", "Object type");
  declare_parameter("object_reference_frame", "base_link", "Object reference frame");
  declare_parameter("object_dimensions", std::vector<double>{0.10, 0.025}, "Object dimensions");
  declare_parameter("object_pose", std::vector<double>{0.35, 0.08, 0.05, 0.0, 0.0, 0.0}, "Object pose");

  // Grasp
  declare_parameter("grasp_frame_transform", std::vector<double>{0.0, 0.0, 0.096, 1.5708, 0.0, 0.0}, "Grasp transform");
  declare_parameter("place_pose", std::vector<double>{0.0, -0.20, 0.05, 0.0, 0.0, 0.0}, "Place pose");

  // Motion
  declare_parameter("approach_object_min_dist", 0.005, "Min approach");
  declare_parameter("approach_object_max_dist", 0.15, "Max approach");
  declare_parameter("lift_object_min_dist", 0.01, "Min lift");
  declare_parameter("lift_object_max_dist", 0.15, "Max lift");
  declare_parameter("lower_object_min_dist", 0.01, "Min lower");
  declare_parameter("lower_object_max_dist", 0.20, "Max lower");
  declare_parameter("move_to_pick_timeout", 10.0, "Pick timeout");
  declare_parameter("move_to_place_timeout", 10.0, "Place timeout");

  // Grasp generation
  declare_parameter("grasp_pose_angle_delta", 0.2618, "Angle delta");
  declare_parameter("grasp_pose_max_ik_solutions", 8, "Max IK solutions");
  declare_parameter("grasp_pose_min_solution_distance", 0.5, "Min solution distance");
  declare_parameter("place_pose_max_ik_solutions", 8, "Place IK solutions");

  // Cartesian
  declare_parameter("cartesian_max_velocity_scaling", 0.3, "Velocity scaling");
  declare_parameter("cartesian_max_acceleration_scaling", 0.3, "Acceleration scaling");
  declare_parameter("cartesian_step_size", 0.005, "Step size");

  // Directions
  declare_parameter("approach_object_direction_z", 1.0, "Approach Z");
  declare_parameter("lift_object_direction_z", 1.0, "Lift Z");
  declare_parameter("lower_object_direction_z", -1.0, "Lower Z");
  declare_parameter("retreat_direction_z", -1.0, "Retreat Z");

  // Other
  declare_parameter("place_pose_z_offset_factor", 0.5, "Place Z offset");
  declare_parameter("retreat_min_distance", 0.01, "Min retreat");
  declare_parameter("retreat_max_distance", 0.15, "Max retreat");
  declare_parameter("graspnet_timeout", 30.0, "GraspNet timeout");

  // Create service clients
  graspnet_multi_client_ = this->create_client<std_srvs::srv::Trigger>("/get_multiple_grasp_poses");
  save_data_client_ = this->create_client<std_srvs::srv::Trigger>("/save_graspnet_data");

  RCLCPP_INFO(this->get_logger(), "MTCGraspNetNode (Multi-Grasp) initialized");
  RCLCPP_INFO(this->get_logger(), "  use_graspnet: %s", 
    this->get_parameter("use_graspnet").as_bool() ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  fallback_to_center: %s",
    this->get_parameter("fallback_to_center").as_bool() ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  num_grasp_candidates: %ld",
    this->get_parameter("num_grasp_candidates").as_int());
}

// ============================================================================
// Async Service Calls
// ============================================================================

void MTCGraspNetNode::callSaveDataService(std::shared_ptr<ServiceCallResult> result) {
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  
  save_data_client_->async_send_request(request,
    [result](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      try {
        auto response = future.get();
        result->success = response->success;
        result->message = response->message;
      } catch (const std::exception& e) {
        result->success = false;
        result->message = e.what();
      }
      result->completed = true;
    });
}

void MTCGraspNetNode::callGetMultipleGraspsService(std::shared_ptr<ServiceCallResult> result) {
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  
  graspnet_multi_client_->async_send_request(request,
    [result](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      try {
        auto response = future.get();
        result->success = response->success;
        result->message = response->message;
      } catch (const std::exception& e) {
        result->success = false;
        result->message = e.what();
      }
      result->completed = true;
    });
}

// ============================================================================
// GraspNet Integration
// ============================================================================

bool MTCGraspNetNode::triggerGraspNetPipeline() {
  RCLCPP_INFO(this->get_logger(), "Triggering GraspNet pipeline...");

  // Check if save_data service is available
  if (!save_data_client_->service_is_ready()) {
    RCLCPP_WARN(this->get_logger(), "save_graspnet_data service not ready, waiting...");
    int wait_count = 0;
    while (!save_data_client_->service_is_ready() && wait_count < 10) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      wait_count++;
    }
    if (!save_data_client_->service_is_ready()) {
      RCLCPP_ERROR(this->get_logger(), "save_graspnet_data service not available");
      return false;
    }
  }

  // Step 1: Call save data service
  auto save_result = std::make_shared<ServiceCallResult>();
  callSaveDataService(save_result);

  // Wait for completion
  auto start = std::chrono::steady_clock::now();
  while (!save_result->completed) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 10.0) {
      RCLCPP_ERROR(this->get_logger(), "save_graspnet_data timeout");
      return false;
    }
  }

  if (!save_result->success) {
    RCLCPP_ERROR(this->get_logger(), "save_graspnet_data failed: %s", save_result->message.c_str());
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Data saved. Waiting for GraspNet processing...");

  // Step 2: Poll for grasp poses
  double timeout = this->get_parameter("graspnet_timeout").as_double();
  start = std::chrono::steady_clock::now();

  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < timeout) {
    if (!graspnet_multi_client_->service_is_ready()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    auto grasp_result = std::make_shared<ServiceCallResult>();
    callGetMultipleGraspsService(grasp_result);

    // Wait for this call
    auto call_start = std::chrono::steady_clock::now();
    while (!grasp_result->completed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - call_start).count() > 5.0) {
        break;
      }
    }

    if (grasp_result->completed && grasp_result->success) {
      RCLCPP_INFO(this->get_logger(), "GraspNet processing complete!");
      return true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  RCLCPP_WARN(this->get_logger(), "GraspNet timeout after %.1f seconds", timeout);
  return false;
}

std::vector<GraspPoseInfo> MTCGraspNetNode::parseMultipleGrasps(const std::string& json) {
  std::vector<GraspPoseInfo> grasps;
  
  // Parse number of grasps
  int num_grasps = extractJsonInt(json, "num_grasps");
  
  if (num_grasps == 0) {
    RCLCPP_WARN(this->get_logger(), "No grasps found in response");
    return grasps;
  }
  
  std::string frame_id = extractJsonString(json, "frame_id");
  
  // Find grasps array
  size_t grasps_pos = json.find("\"grasps\"");
  if (grasps_pos == std::string::npos) {
    RCLCPP_ERROR(this->get_logger(), "No grasps array in response");
    return grasps;
  }
  
  // Parse each grasp
  size_t current_pos = grasps_pos;
  for (int i = 0; i < num_grasps; i++) {
    // Find this grasp's data
    std::string index_key = "\"index\": " + std::to_string(i);
    size_t grasp_start = json.find(index_key, current_pos);
    if (grasp_start == std::string::npos) {
      // Try alternative format
      index_key = "\"index\":" + std::to_string(i);
      grasp_start = json.find(index_key, current_pos);
    }
    
    if (grasp_start == std::string::npos) continue;
    
    // Find the position block for this grasp
    size_t pos_start = json.find("\"position\"", grasp_start);
    if (pos_start == std::string::npos || pos_start > grasp_start + 500) continue;
    
    // Extract position - look within a limited range
    std::string pos_section = json.substr(pos_start, 150);
    double x = extractJsonDouble(pos_section, "x");
    double y = extractJsonDouble(pos_section, "y");
    double z = extractJsonDouble(pos_section, "z");
    
    // Find orientation block
    size_t ori_start = json.find("\"orientation\"", pos_start);
    if (ori_start == std::string::npos || ori_start > pos_start + 300) continue;
    
    std::string ori_section = json.substr(ori_start, 150);
    double qx = extractJsonDouble(ori_section, "x");
    double qy = extractJsonDouble(ori_section, "y");
    double qz = extractJsonDouble(ori_section, "z");
    double qw = extractJsonDouble(ori_section, "w");
    
    // Find score - look between index and position
    std::string score_section = json.substr(grasp_start, pos_start - grasp_start);
    double score = extractJsonDouble(score_section, "score");
    
    // Create grasp info
    GraspPoseInfo info;
    info.index = i;
    info.score = score;
    info.source = GraspSource::GRASPNET;
    info.valid = true;
    
    info.pose.header.frame_id = frame_id;
    info.pose.header.stamp = this->now();
    info.pose.pose.position.x = x;
    info.pose.pose.position.y = y;
    info.pose.pose.position.z = z;
    info.pose.pose.orientation.x = qx;
    info.pose.pose.orientation.y = qy;
    info.pose.pose.orientation.z = qz;
    info.pose.pose.orientation.w = qw;
    
    grasps.push_back(info);
    
    RCLCPP_INFO(this->get_logger(), "Parsed grasp %d: [%.3f, %.3f, %.3f], score=%.3f",
      i, x, y, z, score);
    
    current_pos = ori_start;
  }
  
  return grasps;
}

std::vector<GraspPoseInfo> MTCGraspNetNode::getGraspNetPoses() {
  std::vector<GraspPoseInfo> grasps;

  if (!graspnet_multi_client_->service_is_ready()) {
    RCLCPP_WARN(this->get_logger(), "get_multiple_grasp_poses service not ready");
    return grasps;
  }

  auto result = std::make_shared<ServiceCallResult>();
  callGetMultipleGraspsService(result);

  // Wait with timeout
  auto start = std::chrono::steady_clock::now();
  while (!result->completed) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 10.0) {
      RCLCPP_ERROR(this->get_logger(), "get_multiple_grasp_poses timeout");
      return grasps;
    }
  }

  if (!result->success) {
    RCLCPP_WARN(this->get_logger(), "GraspNet returned no grasps");
    return grasps;
  }

  // Parse multiple grasps from JSON
  grasps = parseMultipleGrasps(result->message);
  
  RCLCPP_INFO(this->get_logger(), "Received %zu grasp poses from GraspNet", grasps.size());
  
  return grasps;
}

GraspPoseInfo MTCGraspNetNode::getCenterBasedPose() {
  GraspPoseInfo info;
  info.source = GraspSource::CENTER;

  auto object_pose = this->get_parameter("object_pose").as_double_array();
  auto object_dimensions = this->get_parameter("object_dimensions").as_double_array();
  auto world_frame = this->get_parameter("world_frame").as_string();

  info.pose.header.frame_id = world_frame;
  info.pose.header.stamp = this->now();
  info.pose.pose.position.x = object_pose[0];
  info.pose.pose.position.y = object_pose[1];
  info.pose.pose.position.z = object_pose[2] + object_dimensions[0] * 0.5;
  info.pose.pose.orientation.x = 0.0;
  info.pose.pose.orientation.y = 0.7071;
  info.pose.pose.orientation.z = 0.0;
  info.pose.pose.orientation.w = 0.7071;
  info.score = 0.5;
  info.index = 0;
  info.valid = true;

  RCLCPP_INFO(this->get_logger(), "Center-based pose: [%.4f, %.4f, %.4f]",
    info.pose.pose.position.x, info.pose.pose.position.y, info.pose.pose.position.z);

  return info;
}

std::vector<GraspPoseInfo> MTCGraspNetNode::getBestGraspPoses() {
  bool use_graspnet = this->get_parameter("use_graspnet").as_bool();
  bool fallback = this->get_parameter("fallback_to_center").as_bool();

  std::vector<GraspPoseInfo> grasps;

  if (use_graspnet) {
    RCLCPP_INFO(this->get_logger(), "=== Attempting GraspNet grasp ===");
    
    if (triggerGraspNetPipeline()) {
      grasps = getGraspNetPoses();
    }

    if (!grasps.empty()) {
      RCLCPP_INFO(this->get_logger(), "Got %zu GraspNet poses", grasps.size());
      return grasps;
    }

    if (!fallback) {
      RCLCPP_ERROR(this->get_logger(), "GraspNet failed, fallback disabled");
      return grasps;
    }

    RCLCPP_WARN(this->get_logger(), "GraspNet failed, using center-based fallback");
  }

  RCLCPP_INFO(this->get_logger(), "=== Using center-based grasp ===");
  grasps.push_back(getCenterBasedPose());
  return grasps;
}

// ============================================================================
// Planning Scene
// ============================================================================

void MTCGraspNetNode::updateObjectParameters(const moveit_msgs::msg::CollisionObject& obj) {
  if (obj.primitives.empty() || obj.primitive_poses.empty()) return;

  const auto& prim = obj.primitives[0];
  const auto& pose = obj.primitive_poses[0];

  std::vector<double> pose_vec = {pose.position.x, pose.position.y, pose.position.z, 0.0, 0.0, 0.0};
  this->set_parameter(rclcpp::Parameter("object_pose", pose_vec));

  std::vector<double> dims;
  if (prim.type == shape_msgs::msg::SolidPrimitive::CYLINDER) {
    dims = {prim.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT],
            prim.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS]};
    this->set_parameter(rclcpp::Parameter("object_type", "cylinder"));
  } else if (prim.type == shape_msgs::msg::SolidPrimitive::BOX) {
    dims = {prim.dimensions[0], prim.dimensions[1], prim.dimensions[2]};
    this->set_parameter(rclcpp::Parameter("object_type", "box"));
  }

  if (!dims.empty()) {
    this->set_parameter(rclcpp::Parameter("object_dimensions", dims));
  }

  RCLCPP_INFO(this->get_logger(), "Updated from collision object: %s", obj.id.c_str());
}

void MTCGraspNetNode::setupPlanningScene() {
  RCLCPP_INFO(this->get_logger(), "Setting up planning scene...");

  auto ps_client = std::make_shared<GetPlanningSceneClient>();

  auto object_type = this->get_parameter("object_type").as_string();
  auto object_dimensions = this->get_parameter("object_dimensions").as_double_array();

  auto response = ps_client->call_service(object_type, object_dimensions);

  if (!response.success) {
    RCLCPP_WARN(this->get_logger(), "Planning scene service returned failure");
    RCLCPP_WARN(this->get_logger(), "Will use default parameters for pick-and-place");
    service_success_ = false;
    support_surface_id_ = "support_surface";
    target_object_id_ = this->get_parameter("object_name").as_string();
    return;
  }

  scene_world_ = response.scene_world;
  full_cloud_ = response.full_cloud;
  rgb_image_ = response.rgb_image;
  target_object_id_ = response.target_object_id;
  support_surface_id_ = response.support_surface_id;
  service_success_ = true;

  for (const auto& obj : scene_world_.collision_objects) {
    if (obj.id == target_object_id_) {
      updateObjectParameters(obj);
      this->set_parameter(rclcpp::Parameter("object_name", obj.id));
      break;
    }
  }

  RCLCPP_INFO(this->get_logger(), "Planning scene ready: target=%s, surface=%s",
    target_object_id_.c_str(), support_surface_id_.c_str());
}

// ============================================================================
// Task Creation with Multiple Grasp Alternatives
// ============================================================================

mtc::Task MTCGraspNetNode::createTask(const std::vector<GraspPoseInfo>& grasp_poses) {
  mtc::Task task;
  task.stages()->setName("pick_place_task");
  task.loadRobotModel(shared_from_this());

  // Get all parameters
  auto arm = this->get_parameter("arm_group_name").as_string();
  auto gripper = this->get_parameter("gripper_group_name").as_string();
  auto gripper_frame = this->get_parameter("gripper_frame").as_string();
  auto open_pose = this->get_parameter("gripper_open_pose").as_string();
  auto close_pose = this->get_parameter("gripper_close_pose").as_string();
  auto home_pose = this->get_parameter("arm_home_pose").as_string();
  auto world = this->get_parameter("world_frame").as_string();
  auto controllers = this->get_parameter("controller_names").as_string_array();
  auto object = this->get_parameter("object_name").as_string();
  auto obj_dims = this->get_parameter("object_dimensions").as_double_array();
  auto grasp_tf = this->get_parameter("grasp_frame_transform").as_double_array();
  auto place = this->get_parameter("place_pose").as_double_array();

  auto approach_min = this->get_parameter("approach_object_min_dist").as_double();
  auto approach_max = this->get_parameter("approach_object_max_dist").as_double();
  auto lift_min = this->get_parameter("lift_object_min_dist").as_double();
  auto lift_max = this->get_parameter("lift_object_max_dist").as_double();
  auto lower_min = this->get_parameter("lower_object_min_dist").as_double();
  auto lower_max = this->get_parameter("lower_object_max_dist").as_double();
  auto pick_timeout = this->get_parameter("move_to_pick_timeout").as_double();
  auto place_timeout = this->get_parameter("move_to_place_timeout").as_double();
  auto max_ik = static_cast<uint32_t>(this->get_parameter("grasp_pose_max_ik_solutions").as_int());
  auto min_dist = this->get_parameter("grasp_pose_min_solution_distance").as_double();
  auto place_ik = static_cast<uint32_t>(this->get_parameter("place_pose_max_ik_solutions").as_int());
  auto vel_scale = this->get_parameter("cartesian_max_velocity_scaling").as_double();
  auto acc_scale = this->get_parameter("cartesian_max_acceleration_scaling").as_double();
  auto step = this->get_parameter("cartesian_step_size").as_double();
  auto approach_z = this->get_parameter("approach_object_direction_z").as_double();
  auto lift_z = this->get_parameter("lift_object_direction_z").as_double();
  auto lower_z = this->get_parameter("lower_object_direction_z").as_double();
  auto retreat_z = this->get_parameter("retreat_direction_z").as_double();
  auto place_z_off = this->get_parameter("place_pose_z_offset_factor").as_double();
  auto retreat_min = this->get_parameter("retreat_min_distance").as_double();
  auto retreat_max = this->get_parameter("retreat_max_distance").as_double();

  // Planners
  std::unordered_map<std::string, std::string> ompl_map = {{"ompl", arm + "[RRTConnectkConfigDefault]"}};
  auto ompl = std::make_shared<mtc::solvers::PipelinePlanner>(this->shared_from_this(), ompl_map);
  auto interp = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
  auto cart = std::make_shared<mtc::solvers::CartesianPath>();
  cart->setMaxVelocityScalingFactor(vel_scale);
  cart->setMaxAccelerationScalingFactor(acc_scale);
  cart->setStepSize(step);

  // Task properties
  task.setProperty("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
  task.setProperty("group", arm);
  task.setProperty("eef", gripper);
  task.setProperty("ik_frame", gripper_frame);

  // Current State
  mtc::Stage* current_state = nullptr;
  {
    auto s = std::make_unique<mtc::stages::CurrentState>("current state");
    current_state = s.get();
    task.add(std::move(s));
  }

  // Open Gripper
  {
    auto s = std::make_unique<mtc::stages::MoveTo>("open gripper", interp);
    s->setGroup(gripper);
    s->setGoal(open_pose);
    s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
    task.add(std::move(s));
  }

  // Move to Pick
  {
    auto s = std::make_unique<mtc::stages::Connect>("move to pick",
      mtc::stages::Connect::GroupPlannerVector{{arm, ompl}, {gripper, interp}});
    s->setTimeout(pick_timeout);
    s->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(s));
  }

  mtc::Stage* attach_stage = nullptr;

  // Pick Object with ALTERNATIVES for multiple grasp poses
  {
    auto pick = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(pick->properties(), {"eef", "group", "ik_frame"});
    pick->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group", "ik_frame"});

    // Approach
    {
      auto s = std::make_unique<mtc::stages::MoveRelative>("approach object", cart);
      s->properties().set("marker_ns", "approach");
      s->properties().set("link", gripper_frame);
      s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
      s->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
      s->setMinMaxDistance(approach_min, approach_max);
      geometry_msgs::msg::Vector3Stamped v;
      v.header.frame_id = gripper_frame;
      v.vector.z = approach_z;
      s->setDirection(v);
      pick->insert(std::move(s));
    }

    // === ALTERNATIVES: Try multiple grasp poses ===
    {
      auto alternatives = std::make_unique<mtc::Alternatives>("grasp alternatives");

      for (const auto& grasp_info : grasp_poses) {
        // Create a serial container for each grasp alternative
        auto grasp_option = std::make_unique<mtc::SerialContainer>(
          "grasp_" + std::to_string(grasp_info.index));
        
        // Allow collision BEFORE computing IK
        {
          auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (gripper,object)");
          s->allowCollisions(object, 
            task.getRobotModel()->getJointModelGroup(gripper)->getLinkModelNamesWithCollisionGeometry(), 
            true);
          grasp_option->insert(std::move(s));
        }
        
        // Generate pose for this grasp
        {
          RCLCPP_INFO(this->get_logger(), "Adding grasp alternative %d: [%.3f, %.3f, %.3f], score=%.3f",
            grasp_info.index,
            grasp_info.pose.pose.position.x,
            grasp_info.pose.pose.position.y,
            grasp_info.pose.pose.position.z,
            grasp_info.score);
          
          auto s = std::make_unique<mtc::stages::GeneratePose>("generate grasp pose");
          s->properties().configureInitFrom(mtc::Stage::PARENT);
          s->properties().set("marker_ns", "grasp_pose_" + std::to_string(grasp_info.index));
          s->setPose(grasp_info.pose);
          s->setMonitoredStage(current_state);

          auto w = std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(s));
          w->setMaxIKSolutions(max_ik);
          w->setMinSolutionDistance(min_dist);
          w->setIKFrame(vectorToEigen(grasp_tf), gripper_frame);
          w->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group"});
          w->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
          grasp_option->insert(std::move(w));
        }

        alternatives->insert(std::move(grasp_option));
      }

      pick->insert(std::move(alternatives));
    }

    // Close gripper
    {
      auto s = std::make_unique<mtc::stages::MoveTo>("close gripper", interp);
      s->setGroup(gripper);
      s->setGoal(close_pose);
      s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
      pick->insert(std::move(s));
    }

    // Allow collision object-surface
    {
      auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (object,support)");
      s->allowCollisions({object}, {support_surface_id_}, true);
      pick->insert(std::move(s));
    }

    // Attach
    {
      auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
      s->attachObject(object, gripper_frame);
      attach_stage = s.get();
      pick->insert(std::move(s));
    }

    // Lift
    {
      auto s = std::make_unique<mtc::stages::MoveRelative>("lift object", cart);
      s->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
      s->setMinMaxDistance(lift_min, lift_max);
      s->setIKFrame(gripper_frame);
      s->properties().set("marker_ns", "lift");
      s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
      geometry_msgs::msg::Vector3Stamped v;
      v.header.frame_id = world;
      v.vector.z = lift_z;
      s->setDirection(v);
      pick->insert(std::move(s));
    }

    // Forbid collision
    {
      auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (object,support)");
      s->allowCollisions({object}, {support_surface_id_}, false);
      pick->insert(std::move(s));
    }

    task.add(std::move(pick));
  }

  // Move to Place
  {
    auto s = std::make_unique<mtc::stages::Connect>("move to place",
      mtc::stages::Connect::GroupPlannerVector{{arm, ompl}, {gripper, interp}});
    s->setTimeout(place_timeout);
    s->properties().configureInitFrom(mtc::Stage::PARENT);
    task.add(std::move(s));
  }

  // Place Object
  {
    auto place_c = std::make_unique<mtc::SerialContainer>("place object");
    task.properties().exposeTo(place_c->properties(), {"eef", "group", "ik_frame"});
    place_c->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group", "ik_frame"});

    // Lower
    {
      auto s = std::make_unique<mtc::stages::MoveRelative>("lower object", cart);
      s->properties().set("marker_ns", "lower");
      s->properties().set("link", gripper_frame);
      s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
      s->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
      s->setMinMaxDistance(lower_min, lower_max);
      geometry_msgs::msg::Vector3Stamped v;
      v.header.frame_id = world;
      v.vector.z = lower_z;
      s->setDirection(v);
      place_c->insert(std::move(s));
    }

    // Generate place pose
    {
      auto s = std::make_unique<mtc::stages::GeneratePlacePose>("generate place pose");
      s->properties().configureInitFrom(mtc::Stage::PARENT, {"ik_frame"});
      s->properties().set("marker_ns", "place_pose");
      s->setObject(object);
      geometry_msgs::msg::PoseStamped p;
      p.header.frame_id = world;
      p.pose = vectorToPose(place);
      p.pose.position.z += place_z_off * obj_dims[0];
      s->setPose(p);
      s->setMonitoredStage(attach_stage);

      auto w = std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(s));
      w->setMaxIKSolutions(place_ik);
      w->setIKFrame(vectorToEigen(grasp_tf), gripper_frame);
      w->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group"});
      w->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
      place_c->insert(std::move(w));
    }

    // Open gripper
    {
      auto s = std::make_unique<mtc::stages::MoveTo>("open gripper", interp);
      s->setGroup(gripper);
      s->setGoal(open_pose);
      s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
      place_c->insert(std::move(s));
    }

    // Forbid collision
    {
      auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (gripper,object)");
      s->allowCollisions(object, *task.getRobotModel()->getJointModelGroup(gripper), false);
      place_c->insert(std::move(s));
    }

    // Detach
    {
      auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
      s->detachObject(object, gripper_frame);
      place_c->insert(std::move(s));
    }

    // Retreat
    {
      auto s = std::make_unique<mtc::stages::MoveRelative>("retreat", cart);
      s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
      s->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
      s->setMinMaxDistance(retreat_min, retreat_max);
      s->setIKFrame(gripper_frame);
      s->properties().set("marker_ns", "retreat");
      geometry_msgs::msg::Vector3Stamped v;
      v.header.frame_id = gripper_frame;
      v.vector.z = retreat_z;
      s->setDirection(v);
      place_c->insert(std::move(s));
    }

    task.add(std::move(place_c));
  }

  // Home
  {
    auto s = std::make_unique<mtc::stages::MoveTo>("move home", ompl);
    s->properties().set("trajectory_execution_info", mtc::TrajectoryExecutionInfo().set__controller_names(controllers));
    s->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
    s->setGoal(home_pose);
    task.add(std::move(s));
  }

  return task;
}

// ============================================================================
// Task Execution
// ============================================================================

void MTCGraspNetNode::doTask() {
  RCLCPP_INFO(this->get_logger(), "============================================");
  RCLCPP_INFO(this->get_logger(), "Starting MTC Pick and Place Task (Multi-Grasp)");
  RCLCPP_INFO(this->get_logger(), "============================================");

  auto grasp_poses = getBestGraspPoses();

  if (grasp_poses.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No valid grasp poses available!");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Using %zu grasp pose alternatives", grasp_poses.size());
  for (const auto& g : grasp_poses) {
    RCLCPP_INFO(this->get_logger(), "  Grasp %d: [%.3f, %.3f, %.3f], score=%.3f",
      g.index, g.pose.pose.position.x, g.pose.pose.position.y, g.pose.pose.position.z, g.score);
  }

  task_ = createTask(grasp_poses);

  try {
    task_.init();
  } catch (mtc::InitStageException& e) {
    RCLCPP_ERROR(this->get_logger(), "Init failed: %s", e.what());
    return;
  }

  auto max_sol = this->get_parameter("max_solutions").as_int();
  RCLCPP_INFO(this->get_logger(), "Planning (max_solutions: %ld, trying %zu grasp alternatives)...", 
    max_sol, grasp_poses.size());

  if (!task_.plan(max_sol)) {
    RCLCPP_ERROR(this->get_logger(), "Planning failed!");
    RCLCPP_INFO(this->get_logger(), "Printing detailed stage info:");
    task_.printState();
    return;
  }

  auto solutions = task_.solutions();
  RCLCPP_INFO(this->get_logger(), "============================================");
  RCLCPP_INFO(this->get_logger(), "Found %zu solutions across all grasp alternatives", solutions.size());
  RCLCPP_INFO(this->get_logger(), "============================================");

  int idx = 0;
  for (const auto& sol : solutions) {
    RCLCPP_INFO(this->get_logger(), "Solution %d: cost = %.4f", idx++, sol->cost());
  }

  // Publish solution for visualization in RViz
  task_.introspection().publishSolution(*task_.solutions().front());

  if (this->get_parameter("execute").as_bool()) {
    RCLCPP_INFO(this->get_logger(), "Executing best solution...");
    auto result = task_.execute(*task_.solutions().front());
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Execution failed!");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Execution complete!");
  } else {
    RCLCPP_INFO(this->get_logger(), "Set execute:=true to run");
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<MTCGraspNetNode>(options);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread task_thread([node]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    try {
      node->setupPlanningScene();
      node->doTask();
      RCLCPP_INFO(node->get_logger(), "Task complete. Press Ctrl+C to exit.");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Error: %s", e.what());
    }
  });

  executor.spin();

  if (task_thread.joinable()) {
    task_thread.join();
  }

  rclcpp::shutdown();
  return 0;
}
