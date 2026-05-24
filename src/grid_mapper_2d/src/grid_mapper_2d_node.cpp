// grid_mapper_2d_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::placeholders::_1;
using std::placeholders::_2;

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

namespace {

Eigen::Quaterniond axisAngleToQuat(const Eigen::Vector3d & w) {
  const double theta = w.norm();

  if (theta < 1e-9) {
    Eigen::Quaterniond q(1.0, 0.5 * w.x(), 0.5 * w.y(), 0.5 * w.z());
    q.normalize();
    return q;
  }

  return Eigen::Quaterniond(Eigen::AngleAxisd(theta, w / theta));
}

uint64_t makeCellKey(int32_t x, int32_t y) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
         static_cast<uint32_t>(y);
}

}  // namespace

struct ImuState {
  double t = 0.0;
  Eigen::Quaterniond q_wi = Eigen::Quaterniond::Identity();
};

class GridMapper2DNode : public rclcpp::Node {
public:
  GridMapper2DNode() : Node("grid_mapper_2d") {
    declareAllParameters();
    loadAllParameters();
    initGrid();

    gicp_.setMaxCorrespondenceDistance(gicp_max_corr_dist_);
    gicp_.setMaximumIterations(gicp_max_iters_);
    gicp_.setTransformationEpsilon(1e-6);
    gicp_.setEuclideanFitnessEpsilon(0.01);

    local_map_.reset(new CloudT);

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GridMapper2DNode::onImu, this, _1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GridMapper2DNode::onCloud, this, _1));

    map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "map",
      rclcpp::QoS(1).transient_local());

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odom", 20);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 20);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_world", 5);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    save_srv_ = create_service<std_srvs::srv::Trigger>(
      "save_map",
      std::bind(&GridMapper2DNode::onSaveMap, this, _1, _2));

    process_timer_ = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&GridMapper2DNode::processData, this));

    publish_timer_ = create_wall_timer(
      std::chrono::milliseconds(publish_period_ms_),
      std::bind(&GridMapper2DNode::publishMap, this));

    path_.header.frame_id = map_frame_;

    RCLCPP_INFO(
      get_logger(),
      "GridMapper2D: original mapping logic kept; only /cloud_world z visualization is corrected.");
  }

  ~GridMapper2DNode() override {
    std::string message;
    saveMapInternal(message);
  }

private:
  struct Grid {
    int width = 0;
    int height = 0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    std::vector<float> log_odds;
  };

  struct DynamicMask2D {
    bool valid = false;
    double res = 0.12;
    double origin_x = 0.0;
    double origin_y = 0.0;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;

    bool contains(double x, double y) const {
      if (!valid || width <= 0 || height <= 0 || data.empty()) {
        return false;
      }

      const int gx = static_cast<int>(std::floor((x - origin_x) / res));
      const int gy = static_cast<int>(std::floor((y - origin_y) / res));

      if (gx < 0 || gy < 0 || gx >= width || gy >= height) {
        return false;
      }

      return data[static_cast<size_t>(gy) * width + gx] != 0;
    }
  };

private:
  std::string cloud_topic_;
  std::string imu_topic_;
  std::string map_frame_;
  std::string body_frame_;
  std::string save_dir_;

  double resolution_ = 0.05;
  double global_map_init_size_m_ = 20.0;
  double global_map_expand_padding_m_ = 10.0;

  double range_max_ = 50.0;
  double raycast_max_range_ = 10.0;
  int publish_period_ms_ = 500;
  double scan_duration_ = 0.1;

  bool carto_2d_optimization_ = true;

  bool map_front_only_en_ = true;
  double map_front_min_x_ = 0.0;
  double map_lateral_limit_y_ = 0.0;

  double obstacle_z_min_ = 0.0;
  double obstacle_z_max_ = 0.5;

  bool use_elevation_ground_filter_ = true;
  double sensor_mount_height_ = 4.0;
  double elevation_grid_res_ = 0.5;
  double elevation_z_tolerance_ = 0.30;
  double elevation_max_slope_ = 0.15;

  double voxel_leaf_scan_ = 0.30;
  double voxel_leaf_map_ = 0.40;
  double voxel_leaf_grid_ = 0.05;

  bool exclude_box_en_ = true;
  double exclude_box_min_x_ = -1.2;
  double exclude_box_max_x_ = 1.2;
  double exclude_box_min_y_ = -1.0;
  double exclude_box_max_y_ = 1.0;
  double exclude_box_min_z_ = -3.0;
  double exclude_box_max_z_ = 0.8;

  bool self_filter_en_ = true;
  double self_filter_min_x_ = -0.2;
  double self_filter_max_x_ = 0.2;
  double self_filter_min_y_ = -0.2;
  double self_filter_max_y_ = 0.2;
  double self_filter_min_z_ = -6.5;
  double self_filter_max_z_ = 1.2;
  double self_filter_clear_margin_ = 0.35;
  double self_filter_clear_log_odds_ = -2.0;

  bool dynamic_self_mask_en_ = true;
  double dynamic_self_mask_res_ = 0.12;
  double dynamic_self_mask_max_range_ = 12.0;
  double dynamic_self_mask_half_width_ = 5.0;
  double dynamic_self_mask_front_x_ = 0.8;
  double dynamic_self_mask_z_min_ = -6.5;
  double dynamic_self_mask_z_max_ = 1.5;
  double dynamic_self_mask_seed_min_x_ = -2.5;
  double dynamic_self_mask_seed_max_x_ = 0.8;
  double dynamic_self_mask_seed_half_width_ = 2.5;
  double dynamic_self_mask_bridge_m_ = 0.45;
  double dynamic_self_mask_margin_m_ = 0.35;
  int dynamic_self_mask_min_component_cells_ = 4;

  bool map_postprocess_en_ = true;
  bool map_boundary_only_en_ = true;
  int map_morph_open_iters_ = 0;
  int map_morph_close_iters_ = 1;
  int map_min_component_cells_ = 6;
  int map_boundary_erode_radius_ = 1;
  int map_boundary_dilate_iters_ = 0;
  double map_publish_occ_prob_ = 0.62;

  double log_odds_hit_ = 0.85;
  double log_odds_miss_ = -0.40;
  double log_odds_min_ = -2.0;
  double log_odds_max_ = 3.5;
  double log_odds_decay_ = 0.0;
  int decay_period_frames_ = 0;

  int imu_init_frames_ = 30;
  bool align_world_to_gravity_ = true;
  bool mapping_use_yaw_only_ = true;

  double gicp_max_corr_dist_ = 1.0;
  int gicp_max_iters_ = 15;
  double local_map_max_size_ = 30.0;
  bool publish_world_cloud_ = true;

  // 仅修复 RViz 中 /cloud_world 的 z 显示，不参与配准、不参与地面分割、不参与二维栅格更新。
  // 原始建图逻辑保持不变，避免重新引入地面拖影或边界漏建。
  bool cloud_world_ground_aligned_ = true;
  bool cloud_world_use_auto_ground_ = true;
  double cloud_world_target_ground_z_ = 0.0;
  double cloud_world_z_offset_ = 0.0;
  double cloud_world_ground_estimate_grid_res_ = 0.50;
  double cloud_world_ground_estimate_percentile_ = 0.25;
  double cloud_world_ground_estimate_smooth_alpha_ = 0.30;
  double cloud_world_ground_rel_z_estimate_ = std::numeric_limits<double>::quiet_NaN();

  Eigen::Matrix3d R_il_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_il_ = Eigen::Vector3d::Zero();

  std::mutex data_mutex_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_buf_;
  std::deque<ImuState> imu_history_;

  Eigen::Quaterniond q_wi_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_wl_last_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_wl_last_ = Eigen::Vector3d::Zero();

  double last_cloud_time_ = -1.0;
  bool first_cloud_ = true;

  bool imu_inited_ = false;
  int imu_init_count_ = 0;
  Eigen::Vector3d acc_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_bias_ = Eigen::Vector3d::Zero();
  double last_imu_t_ = -1.0;

  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp_;
  CloudT::Ptr local_map_;
  std::mutex local_map_mutex_;

  Grid grid_;
  bool grid_initialized_ = false;
  int frame_counter_ = 0;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  std::mutex grid_mutex_;

  nav_msgs::msg::Path path_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
  rclcpp::TimerBase::SharedPtr process_timer_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

private:
  void declareAllParameters() {
    declare_parameter("cloud_topic", "/rslidar_points");
    declare_parameter("imu_topic", "/imu/data");
    declare_parameter("map_frame", "map");
    declare_parameter("body_frame", "base_link");
    declare_parameter("save_dir", "/tmp/grid_map");

    declare_parameter("resolution", 0.05);
    declare_parameter("global_map_init_size_m", 20.0);
    declare_parameter("global_map_expand_padding_m", 10.0);

    declare_parameter("range_max", 50.0);
    declare_parameter("raycast_max_range", 10.0);
    declare_parameter("publish_period_ms", 500);
    declare_parameter("scan_duration", 0.1);

    declare_parameter("carto_2d_optimization", true);

    declare_parameter("map_front_only_en", true);
    declare_parameter("map_front_min_x", 0.0);
    declare_parameter("map_lateral_limit_y", 0.0);

    declare_parameter("obstacle_z_min", 0.0);
    declare_parameter("obstacle_z_max", 0.5);

    declare_parameter("use_elevation_ground_filter", true);
    declare_parameter("sensor_mount_height", 4.0);
    declare_parameter("elevation_grid_res", 0.5);
    declare_parameter("elevation_z_tolerance", 0.30);
    declare_parameter("elevation_max_slope", 0.15);

    declare_parameter("voxel_leaf_scan", 0.30);
    declare_parameter("voxel_leaf_map", 0.40);
    declare_parameter("voxel_leaf_grid", 0.05);

    declare_parameter("exclude_box_en", true);
    declare_parameter("exclude_box_min_x", -1.2);
    declare_parameter("exclude_box_max_x", 1.2);
    declare_parameter("exclude_box_min_y", -1.0);
    declare_parameter("exclude_box_max_y", 1.0);
    declare_parameter("exclude_box_min_z", -3.0);
    declare_parameter("exclude_box_max_z", 0.8);

    declare_parameter("self_filter_en", true);
    declare_parameter("self_filter_min_x", -0.2);
    declare_parameter("self_filter_max_x", 0.2);
    declare_parameter("self_filter_min_y", -0.2);
    declare_parameter("self_filter_max_y", 0.2);
    declare_parameter("self_filter_min_z", -6.5);
    declare_parameter("self_filter_max_z", 1.2);
    declare_parameter("self_filter_clear_margin", 0.35);
    declare_parameter("self_filter_clear_log_odds", -2.0);

    declare_parameter("dynamic_self_mask_en", true);
    declare_parameter("dynamic_self_mask_res", 0.12);
    declare_parameter("dynamic_self_mask_max_range", 12.0);
    declare_parameter("dynamic_self_mask_half_width", 5.0);
    declare_parameter("dynamic_self_mask_front_x", 0.8);
    declare_parameter("dynamic_self_mask_z_min", -6.5);
    declare_parameter("dynamic_self_mask_z_max", 1.5);
    declare_parameter("dynamic_self_mask_seed_min_x", -2.5);
    declare_parameter("dynamic_self_mask_seed_max_x", 0.8);
    declare_parameter("dynamic_self_mask_seed_half_width", 2.5);
    declare_parameter("dynamic_self_mask_bridge_m", 0.45);
    declare_parameter("dynamic_self_mask_margin_m", 0.35);
    declare_parameter("dynamic_self_mask_min_component_cells", 4);

    declare_parameter("map_postprocess_en", true);
    declare_parameter("map_boundary_only_en", true);
    declare_parameter("map_morph_open_iters", 0);
    declare_parameter("map_morph_close_iters", 1);
    declare_parameter("map_min_component_cells", 6);
    declare_parameter("map_boundary_erode_radius", 1);
    declare_parameter("map_boundary_dilate_iters", 0);
    declare_parameter("map_publish_occ_prob", 0.62);

    declare_parameter("log_odds_hit", 0.85);
    declare_parameter("log_odds_miss", -0.40);
    declare_parameter("log_odds_min", -2.0);
    declare_parameter("log_odds_max", 3.5);
    declare_parameter("log_odds_decay", 0.0);
    declare_parameter("decay_period_frames", 0);

    declare_parameter("imu_init_frames", 30);
    declare_parameter("align_world_to_gravity", true);
    declare_parameter("mapping_use_yaw_only", true);

    declare_parameter("gicp_max_correspondence_distance", 1.0);
    declare_parameter("gicp_max_iterations", 15);
    declare_parameter("local_map_max_size", 30.0);
    declare_parameter("publish_world_cloud", true);

    // 只用于 /cloud_world 可视化高度修正，不影响建图。
    declare_parameter("cloud_world_ground_aligned", true);
    declare_parameter("cloud_world_use_auto_ground", true);
    declare_parameter("cloud_world_target_ground_z", 0.0);
    declare_parameter("cloud_world_z_offset", 0.0);
    declare_parameter("cloud_world_ground_estimate_grid_res", 0.50);
    declare_parameter("cloud_world_ground_estimate_percentile", 0.25);
    declare_parameter("cloud_world_ground_estimate_smooth_alpha", 0.30);

    declare_parameter("extrinsic_T_lidar_in_imu", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter(
      "extrinsic_R_lidar_in_imu",
      std::vector<double>{1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0});
  }

  void loadAllParameters() {
    cloud_topic_ = get_parameter("cloud_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    map_frame_ = get_parameter("map_frame").as_string();
    body_frame_ = get_parameter("body_frame").as_string();
    save_dir_ = get_parameter("save_dir").as_string();

    resolution_ = std::max(0.01, get_parameter("resolution").as_double());
    global_map_init_size_m_ = std::max(5.0, get_parameter("global_map_init_size_m").as_double());
    global_map_expand_padding_m_ = std::max(1.0, get_parameter("global_map_expand_padding_m").as_double());

    range_max_ = get_parameter("range_max").as_double();
    raycast_max_range_ = get_parameter("raycast_max_range").as_double();
    publish_period_ms_ = static_cast<int>(get_parameter("publish_period_ms").as_int());
    scan_duration_ = get_parameter("scan_duration").as_double();

    carto_2d_optimization_ = get_parameter("carto_2d_optimization").as_bool();

    map_front_only_en_ = get_parameter("map_front_only_en").as_bool();
    map_front_min_x_ = get_parameter("map_front_min_x").as_double();
    map_lateral_limit_y_ = std::max(0.0, get_parameter("map_lateral_limit_y").as_double());

    obstacle_z_min_ = get_parameter("obstacle_z_min").as_double();
    obstacle_z_max_ = get_parameter("obstacle_z_max").as_double();

    use_elevation_ground_filter_ = get_parameter("use_elevation_ground_filter").as_bool();
    sensor_mount_height_ = get_parameter("sensor_mount_height").as_double();
    elevation_grid_res_ = std::max(0.05, get_parameter("elevation_grid_res").as_double());
    elevation_z_tolerance_ = get_parameter("elevation_z_tolerance").as_double();
    elevation_max_slope_ = get_parameter("elevation_max_slope").as_double();

    voxel_leaf_scan_ = get_parameter("voxel_leaf_scan").as_double();
    voxel_leaf_map_ = get_parameter("voxel_leaf_map").as_double();
    voxel_leaf_grid_ = get_parameter("voxel_leaf_grid").as_double();

    exclude_box_en_ = get_parameter("exclude_box_en").as_bool();
    exclude_box_min_x_ = get_parameter("exclude_box_min_x").as_double();
    exclude_box_max_x_ = get_parameter("exclude_box_max_x").as_double();
    exclude_box_min_y_ = get_parameter("exclude_box_min_y").as_double();
    exclude_box_max_y_ = get_parameter("exclude_box_max_y").as_double();
    exclude_box_min_z_ = get_parameter("exclude_box_min_z").as_double();
    exclude_box_max_z_ = get_parameter("exclude_box_max_z").as_double();

    self_filter_en_ = get_parameter("self_filter_en").as_bool();
    self_filter_min_x_ = get_parameter("self_filter_min_x").as_double();
    self_filter_max_x_ = get_parameter("self_filter_max_x").as_double();
    self_filter_min_y_ = get_parameter("self_filter_min_y").as_double();
    self_filter_max_y_ = get_parameter("self_filter_max_y").as_double();
    self_filter_min_z_ = get_parameter("self_filter_min_z").as_double();
    self_filter_max_z_ = get_parameter("self_filter_max_z").as_double();
    self_filter_clear_margin_ = std::max(0.0, get_parameter("self_filter_clear_margin").as_double());
    self_filter_clear_log_odds_ = get_parameter("self_filter_clear_log_odds").as_double();

    dynamic_self_mask_en_ = get_parameter("dynamic_self_mask_en").as_bool();
    dynamic_self_mask_res_ = std::max(0.03, get_parameter("dynamic_self_mask_res").as_double());
    dynamic_self_mask_max_range_ = std::max(1.0, get_parameter("dynamic_self_mask_max_range").as_double());
    dynamic_self_mask_half_width_ = std::max(0.5, get_parameter("dynamic_self_mask_half_width").as_double());
    dynamic_self_mask_front_x_ = get_parameter("dynamic_self_mask_front_x").as_double();
    dynamic_self_mask_z_min_ = get_parameter("dynamic_self_mask_z_min").as_double();
    dynamic_self_mask_z_max_ = get_parameter("dynamic_self_mask_z_max").as_double();
    dynamic_self_mask_seed_min_x_ = get_parameter("dynamic_self_mask_seed_min_x").as_double();
    dynamic_self_mask_seed_max_x_ = get_parameter("dynamic_self_mask_seed_max_x").as_double();
    dynamic_self_mask_seed_half_width_ =
      std::max(0.1, get_parameter("dynamic_self_mask_seed_half_width").as_double());
    dynamic_self_mask_bridge_m_ =
      std::max(0.0, get_parameter("dynamic_self_mask_bridge_m").as_double());
    dynamic_self_mask_margin_m_ =
      std::max(0.0, get_parameter("dynamic_self_mask_margin_m").as_double());
    dynamic_self_mask_min_component_cells_ =
      std::max(1, static_cast<int>(get_parameter("dynamic_self_mask_min_component_cells").as_int()));

    map_postprocess_en_ = get_parameter("map_postprocess_en").as_bool();
    map_boundary_only_en_ = get_parameter("map_boundary_only_en").as_bool();
    map_morph_open_iters_ =
      std::max(0, static_cast<int>(get_parameter("map_morph_open_iters").as_int()));
    map_morph_close_iters_ =
      std::max(0, static_cast<int>(get_parameter("map_morph_close_iters").as_int()));
    map_min_component_cells_ =
      std::max(1, static_cast<int>(get_parameter("map_min_component_cells").as_int()));
    map_boundary_erode_radius_ =
      std::max(1, static_cast<int>(get_parameter("map_boundary_erode_radius").as_int()));
    map_boundary_dilate_iters_ =
      std::max(0, static_cast<int>(get_parameter("map_boundary_dilate_iters").as_int()));
    map_publish_occ_prob_ =
      std::clamp(get_parameter("map_publish_occ_prob").as_double(), 0.50, 0.95);

    log_odds_hit_ = get_parameter("log_odds_hit").as_double();
    log_odds_miss_ = get_parameter("log_odds_miss").as_double();
    log_odds_min_ = get_parameter("log_odds_min").as_double();
    log_odds_max_ = get_parameter("log_odds_max").as_double();
    log_odds_decay_ = std::max(0.0, get_parameter("log_odds_decay").as_double());
    decay_period_frames_ =
      std::max(0, static_cast<int>(get_parameter("decay_period_frames").as_int()));

    imu_init_frames_ = static_cast<int>(get_parameter("imu_init_frames").as_int());
    align_world_to_gravity_ = get_parameter("align_world_to_gravity").as_bool();
    mapping_use_yaw_only_ = get_parameter("mapping_use_yaw_only").as_bool();

    gicp_max_corr_dist_ = get_parameter("gicp_max_correspondence_distance").as_double();
    gicp_max_iters_ = static_cast<int>(get_parameter("gicp_max_iterations").as_int());
    local_map_max_size_ = get_parameter("local_map_max_size").as_double();
    publish_world_cloud_ = get_parameter("publish_world_cloud").as_bool();

    cloud_world_ground_aligned_ = get_parameter("cloud_world_ground_aligned").as_bool();
    cloud_world_use_auto_ground_ = get_parameter("cloud_world_use_auto_ground").as_bool();
    cloud_world_target_ground_z_ = get_parameter("cloud_world_target_ground_z").as_double();
    cloud_world_z_offset_ = get_parameter("cloud_world_z_offset").as_double();
    cloud_world_ground_estimate_grid_res_ =
      std::max(0.10, get_parameter("cloud_world_ground_estimate_grid_res").as_double());
    cloud_world_ground_estimate_percentile_ =
      std::clamp(get_parameter("cloud_world_ground_estimate_percentile").as_double(), 0.02, 0.80);
    cloud_world_ground_estimate_smooth_alpha_ =
      std::clamp(get_parameter("cloud_world_ground_estimate_smooth_alpha").as_double(), 0.0, 1.0);

    const auto et = get_parameter("extrinsic_T_lidar_in_imu").as_double_array();
    if (et.size() == 3) {
      t_il_ << et[0], et[1], et[2];
    }

    const auto eR = get_parameter("extrinsic_R_lidar_in_imu").as_double_array();
    if (eR.size() == 9) {
      R_il_ << eR[0], eR[1], eR[2],
               eR[3], eR[4], eR[5],
               eR[6], eR[7], eR[8];
    }
  }

  void initGrid() {
    grid_.width = std::max(20, static_cast<int>(std::ceil(global_map_init_size_m_ / resolution_)));
    grid_.height = grid_.width;
    grid_.origin_x = -0.5 * grid_.width * resolution_;
    grid_.origin_y = -0.5 * grid_.height * resolution_;
    grid_.log_odds.assign(static_cast<size_t>(grid_.width) * grid_.height, 0.0f);
    grid_initialized_ = false;
  }

  static double yawFromRotation(const Eigen::Matrix3d & R) {
    return std::atan2(R(1, 0), R(0, 0));
  }

  static Eigen::Quaterniond yawToQuat(double yaw) {
    Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    q.normalize();
    return q;
  }

  Eigen::Quaterniond getImuQuat(double t) const {
    if (imu_history_.empty()) {
      return Eigen::Quaterniond::Identity();
    }

    if (t <= imu_history_.front().t) {
      return imu_history_.front().q_wi.normalized();
    }

    if (t >= imu_history_.back().t) {
      return imu_history_.back().q_wi.normalized();
    }

    int left = 0;
    int right = static_cast<int>(imu_history_.size()) - 1;

    while (left < right - 1) {
      const int mid = left + (right - left) / 2;
      if (imu_history_[mid].t <= t) {
        left = mid;
      } else {
        right = mid;
      }
    }

    const double dt = imu_history_[right].t - imu_history_[left].t;
    if (dt <= 1e-9) {
      return imu_history_[left].q_wi.normalized();
    }

    const double r = (t - imu_history_[left].t) / dt;
    return imu_history_[left].q_wi.slerp(r, imu_history_[right].q_wi).normalized();
  }

  Eigen::Quaterniond getMappingQuat(double t) const {
    const Eigen::Quaterniond q = getImuQuat(t);

    if (!mapping_use_yaw_only_) {
      return q.normalized();
    }

    return yawToQuat(yawFromRotation(q.toRotationMatrix()));
  }

  void onImu(sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    const Eigen::Vector3d acc(
      msg->linear_acceleration.x,
      msg->linear_acceleration.y,
      msg->linear_acceleration.z);

    const Eigen::Vector3d gyr(
      msg->angular_velocity.x,
      msg->angular_velocity.y,
      msg->angular_velocity.z);

    const double t = rclcpp::Time(msg->header.stamp).seconds();

    std::lock_guard<std::mutex> lk(data_mutex_);

    if (!imu_inited_) {
      acc_acc_ += acc;
      gyr_acc_ += gyr;
      imu_init_count_++;

      if (imu_init_count_ >= imu_init_frames_) {
        gyr_bias_ = gyr_acc_ / static_cast<double>(imu_init_count_);

        if (align_world_to_gravity_ && acc_acc_.norm() > 1e-6) {
          const Eigen::Vector3d acc_mean =
            acc_acc_ / static_cast<double>(imu_init_count_);

          q_wi_ = Eigen::Quaterniond::FromTwoVectors(
            acc_mean.normalized(),
            Eigen::Vector3d::UnitZ());

          q_wi_.normalize();
        } else {
          q_wi_.setIdentity();
        }

        last_imu_t_ = t;
        imu_inited_ = true;
        imu_history_.clear();
        imu_history_.push_back({t, q_wi_});

        RCLCPP_INFO(
          get_logger(),
          "IMU initialized. mapping_use_yaw_only=%s",
          mapping_use_yaw_only_ ? "true" : "false");
      }

      return;
    }

    const double dt = t - last_imu_t_;

    if (dt > 0.0 && dt <= 0.2) {
      q_wi_ = (q_wi_ * axisAngleToQuat((gyr - gyr_bias_) * dt)).normalized();

      imu_history_.push_back({t, q_wi_});
      while (imu_history_.size() > 500) {
        imu_history_.pop_front();
      }
    }

    last_imu_t_ = t;
  }

  void onCloud(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(data_mutex_);
    cloud_buf_.push_back(msg);
  }

  bool inExcludeBoxLidar(const PointT & p) const {
    if (!exclude_box_en_) {
      return false;
    }

    return p.x >= exclude_box_min_x_ && p.x <= exclude_box_max_x_ &&
           p.y >= exclude_box_min_y_ && p.y <= exclude_box_max_y_ &&
           p.z >= exclude_box_min_z_ && p.z <= exclude_box_max_z_;
  }

  bool inSelfBoxLidar(const PointT & p, double margin = 0.0) const {
    if (!self_filter_en_) {
      return false;
    }

    return p.x >= self_filter_min_x_ - margin &&
           p.x <= self_filter_max_x_ + margin &&
           p.y >= self_filter_min_y_ - margin &&
           p.y <= self_filter_max_y_ + margin &&
           p.z >= self_filter_min_z_ &&
           p.z <= self_filter_max_z_;
  }

  static std::vector<uint8_t> dilateMask(
    const std::vector<uint8_t> & in,
    int w,
    int h,
    int r) {
    if (r <= 0 || in.empty()) {
      return in;
    }

    std::vector<uint8_t> out(static_cast<size_t>(w) * h, 0);

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        bool any = false;

        for (int dy = -r; dy <= r && !any; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= h) {
            continue;
          }

          for (int dx = -r; dx <= r; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= w) {
              continue;
            }

            if (dx * dx + dy * dy > r * r) {
              continue;
            }

            if (in[static_cast<size_t>(yy) * w + xx]) {
              any = true;
              break;
            }
          }
        }

        out[static_cast<size_t>(y) * w + x] = any ? 1 : 0;
      }
    }

    return out;
  }

  static std::vector<uint8_t> erodeMask(
    const std::vector<uint8_t> & in,
    int w,
    int h,
    int r) {
    if (r <= 0 || in.empty()) {
      return in;
    }

    std::vector<uint8_t> out(static_cast<size_t>(w) * h, 0);

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        bool all = true;

        for (int dy = -r; dy <= r && all; ++dy) {
          const int yy = y + dy;

          for (int dx = -r; dx <= r; ++dx) {
            const int xx = x + dx;

            if (dx * dx + dy * dy > r * r) {
              continue;
            }

            if (xx < 0 || yy < 0 || xx >= w || yy >= h ||
                !in[static_cast<size_t>(yy) * w + xx]) {
              all = false;
              break;
            }
          }
        }

        out[static_cast<size_t>(y) * w + x] = all ? 1 : 0;
      }
    }

    return out;
  }

  static void removeSmallComponents(
    std::vector<uint8_t> & mask,
    int w,
    int h,
    int min_cells) {
    if (min_cells <= 1 || mask.empty()) {
      return;
    }

    std::vector<uint8_t> visited(static_cast<size_t>(w) * h, 0);
    std::queue<int> q;
    std::vector<int> comp;

    const int nx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int ny[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int start = y * w + x;

        if (!mask[start] || visited[start]) {
          continue;
        }

        visited[start] = 1;
        q.push(start);
        comp.clear();

        while (!q.empty()) {
          const int id = q.front();
          q.pop();
          comp.push_back(id);

          const int cx = id % w;
          const int cy = id / w;

          for (int k = 0; k < 8; ++k) {
            const int xx = cx + nx[k];
            const int yy = cy + ny[k];

            if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
              continue;
            }

            const int nid = yy * w + xx;

            if (mask[nid] && !visited[nid]) {
              visited[nid] = 1;
              q.push(nid);
            }
          }
        }

        if (static_cast<int>(comp.size()) < min_cells) {
          for (const int id : comp) {
            mask[id] = 0;
          }
        }
      }
    }
  }

  DynamicMask2D buildDynamicSelfMask(const CloudT::Ptr & raw) const {
    DynamicMask2D mask;

    if (!dynamic_self_mask_en_ || raw->empty()) {
      return mask;
    }

    mask.res = dynamic_self_mask_res_;
    mask.origin_x = -dynamic_self_mask_max_range_;
    mask.origin_y = -dynamic_self_mask_half_width_;

    mask.width = std::max(
      1,
      static_cast<int>(
        std::ceil((dynamic_self_mask_front_x_ + dynamic_self_mask_max_range_) / mask.res)));

    mask.height = std::max(
      1,
      static_cast<int>(std::ceil((2.0 * dynamic_self_mask_half_width_) / mask.res)));

    const size_t total = static_cast<size_t>(mask.width) * mask.height;
    std::vector<uint8_t> occ(total, 0);

    const double r2_max = dynamic_self_mask_max_range_ * dynamic_self_mask_max_range_;

    for (const auto & p : raw->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }

      if (p.z < dynamic_self_mask_z_min_ || p.z > dynamic_self_mask_z_max_) {
        continue;
      }

      if (p.x > dynamic_self_mask_front_x_) {
        continue;
      }

      if (std::abs(p.y) > dynamic_self_mask_half_width_) {
        continue;
      }

      if (p.x * p.x + p.y * p.y + p.z * p.z > r2_max) {
        continue;
      }

      const int gx = static_cast<int>(std::floor((p.x - mask.origin_x) / mask.res));
      const int gy = static_cast<int>(std::floor((p.y - mask.origin_y) / mask.res));

      if (gx >= 0 && gy >= 0 && gx < mask.width && gy < mask.height) {
        occ[static_cast<size_t>(gy) * mask.width + gx] = 1;
      }
    }

    const int bridge_r =
      static_cast<int>(std::ceil(dynamic_self_mask_bridge_m_ / mask.res));

    std::vector<uint8_t> connected =
      dilateMask(occ, mask.width, mask.height, bridge_r);

    std::vector<uint8_t> selected(total, 0);
    std::vector<uint8_t> visited(total, 0);
    std::queue<int> q;
    std::vector<int> comp;

    const int nx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int ny[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    for (int y = 0; y < mask.height; ++y) {
      for (int x = 0; x < mask.width; ++x) {
        const int start = y * mask.width + x;

        if (!connected[start] || visited[start]) {
          continue;
        }

        visited[start] = 1;
        q.push(start);
        comp.clear();

        bool touch_seed = false;

        while (!q.empty()) {
          const int id = q.front();
          q.pop();
          comp.push_back(id);

          const int cx = id % mask.width;
          const int cy = id / mask.width;

          const double lx =
            mask.origin_x + (static_cast<double>(cx) + 0.5) * mask.res;
          const double ly =
            mask.origin_y + (static_cast<double>(cy) + 0.5) * mask.res;

          if (lx >= dynamic_self_mask_seed_min_x_ &&
              lx <= dynamic_self_mask_seed_max_x_ &&
              std::abs(ly) <= dynamic_self_mask_seed_half_width_) {
            touch_seed = true;
          }

          for (int k = 0; k < 8; ++k) {
            const int xx = cx + nx[k];
            const int yy = cy + ny[k];

            if (xx < 0 || yy < 0 || xx >= mask.width || yy >= mask.height) {
              continue;
            }

            const int nid = yy * mask.width + xx;

            if (connected[nid] && !visited[nid]) {
              visited[nid] = 1;
              q.push(nid);
            }
          }
        }

        if (touch_seed &&
            static_cast<int>(comp.size()) >= dynamic_self_mask_min_component_cells_) {
          for (const int id : comp) {
            selected[id] = 1;
          }
        }
      }
    }

    const int margin_r =
      static_cast<int>(std::ceil(dynamic_self_mask_margin_m_ / mask.res));

    mask.data = dilateMask(selected, mask.width, mask.height, margin_r);

    mask.valid = std::any_of(mask.data.begin(), mask.data.end(), [](uint8_t v) {
      return v != 0;
    });

    return mask;
  }

  bool inDynamicSelfMaskLidar(const PointT & p, const DynamicMask2D & mask) const {
    if (!dynamic_self_mask_en_ || !mask.valid) {
      return false;
    }

    if (p.z < dynamic_self_mask_z_min_ || p.z > dynamic_self_mask_z_max_) {
      return false;
    }

    return mask.contains(p.x, p.y);
  }

  bool allowedForGlobalGridLidar(const PointT & p, const DynamicMask2D & mask) const {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      return false;
    }

    if (inExcludeBoxLidar(p)) {
      return false;
    }

    if (inSelfBoxLidar(p)) {
      return false;
    }

    if (inDynamicSelfMaskLidar(p, mask)) {
      return false;
    }

    if (map_front_only_en_ && p.x < map_front_min_x_) {
      return false;
    }

    if (map_lateral_limit_y_ > 0.0 && std::abs(p.y) > map_lateral_limit_y_) {
      return false;
    }

    return true;
  }

  void voxelDownsample(CloudT::Ptr & cloud, double leaf) const {
    if (leaf <= 0.0 || cloud->empty()) {
      return;
    }

    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(leaf, leaf, leaf);

    CloudT::Ptr tmp(new CloudT);
    vg.setInputCloud(cloud);
    vg.filter(*tmp);
    cloud->swap(*tmp);
  }

  void cropLocalMap(const Eigen::Vector3d & center) {
    if (local_map_->empty()) {
      return;
    }

    const double r2 = local_map_max_size_ * local_map_max_size_;

    CloudT::Ptr cropped(new CloudT);
    cropped->reserve(local_map_->size());

    for (const auto & p : local_map_->points) {
      const double dx = p.x - center.x();
      const double dy = p.y - center.y();

      if (dx * dx + dy * dy <= r2) {
        cropped->push_back(p);
      }
    }

    local_map_->swap(*cropped);
  }

  void processData() {
    std::lock_guard<std::mutex> lk(data_mutex_);

    if (cloud_buf_.empty() || !imu_inited_) {
      return;
    }

    auto cloud_msg = cloud_buf_.front();

    const double t_start = rclcpp::Time(cloud_msg->header.stamp).seconds();
    const double t_end = t_start + scan_duration_;

    if (imu_history_.empty() || imu_history_.back().t < t_end) {
      return;
    }

    cloud_buf_.pop_front();

    CloudT::Ptr raw(new CloudT);
    pcl::fromROSMsg(*cloud_msg, *raw);

    if (raw->empty()) {
      return;
    }

    const DynamicMask2D dynamic_mask = buildDynamicSelfMask(raw);

    CloudT::Ptr clean_for_pose(new CloudT);
    clean_for_pose->reserve(raw->size());

    const double r2_max = range_max_ * range_max_;

    for (const auto & p : raw->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }

      if (p.x * p.x + p.y * p.y + p.z * p.z > r2_max) {
        continue;
      }

      if (inExcludeBoxLidar(p)) {
        continue;
      }

      if (inSelfBoxLidar(p)) {
        continue;
      }

      if (inDynamicSelfMaskLidar(p, dynamic_mask)) {
        continue;
      }

      clean_for_pose->push_back(p);
    }

    if (clean_for_pose->empty()) {
      return;
    }

    const double dt_pred =
      last_cloud_time_ > 0.0 ? std::max(0.0, t_end - last_cloud_time_) : 0.0;

    const Eigen::Quaterniond q_wi_end = getMappingQuat(t_end);
    const Eigen::Vector3d p_wl_pred = p_wl_last_ + v_wl_last_ * dt_pred;

    Eigen::Matrix4d T_wl_pred = Eigen::Matrix4d::Identity();
    T_wl_pred.block<3,3>(0,0) = q_wi_end.toRotationMatrix() * R_il_;
    T_wl_pred.block<3,1>(0,3) = p_wl_pred;

    const Eigen::Matrix4d T_lw_pred = T_wl_pred.inverse();

    CloudT::Ptr deskewed(new CloudT);
    deskewed->reserve(clean_for_pose->size());

    for (size_t i = 0; i < clean_for_pose->size(); ++i) {
      const auto & pt = clean_for_pose->points[i];

      const double r =
        static_cast<double>(i) / std::max<size_t>(1, clean_for_pose->size());

      const double tj = t_start + r * scan_duration_;
      const double dtj =
        last_cloud_time_ > 0.0 ? std::max(0.0, tj - last_cloud_time_) : 0.0;

      const Eigen::Quaterniond qj = getMappingQuat(tj);
      const Eigen::Vector3d pj = p_wl_last_ + v_wl_last_ * dtj;

      Eigen::Matrix4d T_wl_j = Eigen::Matrix4d::Identity();
      T_wl_j.block<3,3>(0,0) = qj.toRotationMatrix() * R_il_;
      T_wl_j.block<3,1>(0,3) = pj;

      const Eigen::Vector3d pw =
        T_wl_j.block<3,3>(0,0) * Eigen::Vector3d(pt.x, pt.y, pt.z) +
        T_wl_j.block<3,1>(0,3);

      const Eigen::Vector3d pl =
        T_lw_pred.block<3,3>(0,0) * pw +
        T_lw_pred.block<3,1>(0,3);

      PointT out = pt;
      out.x = static_cast<float>(pl.x());
      out.y = static_cast<float>(pl.y());
      out.z = static_cast<float>(pl.z());
      deskewed->push_back(out);
    }

    CloudT::Ptr scan_ds(new CloudT(*deskewed));
    voxelDownsample(scan_ds, voxel_leaf_scan_);

    Eigen::Matrix4d T_wl_final = T_wl_pred;
    double gicp_delta_yaw_for_imu_feedback = 0.0;

    if (!first_cloud_) {
      CloudT::Ptr source_world(new CloudT);
      pcl::transformPointCloud(*scan_ds, *source_world, T_wl_pred.cast<float>());

      std::lock_guard<std::mutex> lk_map(local_map_mutex_);

      if (!local_map_->empty() && source_world->size() > 20) {
        gicp_.setInputSource(source_world);
        gicp_.setInputTarget(local_map_);

        CloudT aligned;
        gicp_.align(aligned, Eigen::Matrix4f::Identity());

        if (gicp_.hasConverged()) {
          Eigen::Matrix4d T_delta = gicp_.getFinalTransformation().cast<double>();

          if (carto_2d_optimization_) {
            const double delta_yaw =
              yawFromRotation(T_delta.block<3,3>(0,0));

            const double tx = T_delta(0, 3);
            const double ty = T_delta(1, 3);

            T_delta.setIdentity();
            T_delta(0, 0) = std::cos(delta_yaw);
            T_delta(0, 1) = -std::sin(delta_yaw);
            T_delta(1, 0) = std::sin(delta_yaw);
            T_delta(1, 1) = std::cos(delta_yaw);
            T_delta(0, 3) = tx;
            T_delta(1, 3) = ty;
            T_delta(2, 3) = 0.0;
          }

          gicp_delta_yaw_for_imu_feedback =
            yawFromRotation(T_delta.block<3,3>(0,0));

          T_wl_final = T_delta * T_wl_pred;
        }
      }
    } else {
      T_wl_final.block<3,1>(0,3) = Eigen::Vector3d::Zero();
    }

    if (last_cloud_time_ > 0.0) {
      const double dt = std::max(1e-3, t_end - last_cloud_time_);
      v_wl_last_ = (T_wl_final.block<3,1>(0,3) - p_wl_last_) / dt;
    } else {
      v_wl_last_.setZero();
    }

    if (carto_2d_optimization_) {
      v_wl_last_.z() = 0.0;
      T_wl_final(2, 3) = 0.0;
    }

    p_wl_last_ = T_wl_final.block<3,1>(0,3);
    last_cloud_time_ = t_end;

    if (!first_cloud_) {
      Eigen::Quaterniond q_err = Eigen::Quaterniond::Identity();

      if (mapping_use_yaw_only_) {
        q_err = yawToQuat(gicp_delta_yaw_for_imu_feedback);
      }

      for (auto & state : imu_history_) {
        state.q_wi = (q_err * state.q_wi).normalized();
      }

      q_wi_ = (q_err * q_wi_).normalized();
    }

    CloudT::Ptr scan_world(new CloudT);
    pcl::transformPointCloud(*scan_ds, *scan_world, T_wl_final.cast<float>());

    {
      std::lock_guard<std::mutex> lk_map(local_map_mutex_);
      *local_map_ += *scan_world;
      cropLocalMap(T_wl_final.block<3,1>(0,3));
      voxelDownsample(local_map_, voxel_leaf_map_);
    }

    CloudT::Ptr grid_lidar(new CloudT);
    grid_lidar->reserve(deskewed->size());

    for (const auto & p : deskewed->points) {
      if (!allowedForGlobalGridLidar(p, dynamic_mask)) {
        continue;
      }

      grid_lidar->push_back(p);
    }

    voxelDownsample(grid_lidar, voxel_leaf_grid_);

    CloudT::Ptr grid_world(new CloudT);
    pcl::transformPointCloud(*grid_lidar, *grid_world, T_wl_final.cast<float>());

    const Eigen::Vector3d sensor_pos = T_wl_final.block<3,1>(0,3);
    const double rviz_ground_rel_z = estimateCloudWorldGroundRelZForRviz(grid_world, sensor_pos);

    CloudT::Ptr nonground_world(new CloudT);
    segmentGround(grid_world, sensor_pos, nonground_world);

    last_stamp_ = cloud_msg->header.stamp;

    // 建图仍然使用原始 nonground_world，z 不做任何平移。
    updateGlobalGridFromCurrentScan(nonground_world, T_wl_final);
    publishPoseAndPath(T_wl_final);

    if (publish_world_cloud_) {
      CloudT::Ptr cloud_for_rviz =
        makeCloudWorldForRvizOnly(nonground_world, sensor_pos, rviz_ground_rel_z);

      sensor_msgs::msg::PointCloud2 msg_out;
      pcl::toROSMsg(*cloud_for_rviz, msg_out);
      msg_out.header.stamp = last_stamp_;
      msg_out.header.frame_id = map_frame_;
      cloud_pub_->publish(msg_out);
    }

    first_cloud_ = false;
  }

  double estimateCloudWorldGroundRelZForRviz(
    const CloudT::Ptr & cloud_world,
    const Eigen::Vector3d & sensor_pos) {
    const double nominal_ground_rel_z = -std::max(0.1, sensor_mount_height_);

    if (!cloud_world_ground_aligned_) {
      return nominal_ground_rel_z;
    }

    if (!cloud_world_use_auto_ground_ || cloud_world->empty()) {
      cloud_world_ground_rel_z_estimate_ = nominal_ground_rel_z;
      return cloud_world_ground_rel_z_estimate_;
    }

    std::unordered_map<uint64_t, double> cell_min_z;
    cell_min_z.reserve(cloud_world->size());

    // 估计范围放宽一点，避免 sensor_mount_height 不精确时自动估计失败。
    const double min_valid_rel_z = nominal_ground_rel_z - 4.0;
    const double max_valid_rel_z = nominal_ground_rel_z + 4.0;

    for (const auto & p : cloud_world->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }

      const double rel_z = static_cast<double>(p.z) - sensor_pos.z();

      if (rel_z < min_valid_rel_z || rel_z > max_valid_rel_z) {
        continue;
      }

      const int32_t cx = static_cast<int32_t>(
        std::floor((static_cast<double>(p.x) - sensor_pos.x()) /
                   cloud_world_ground_estimate_grid_res_));

      const int32_t cy = static_cast<int32_t>(
        std::floor((static_cast<double>(p.y) - sensor_pos.y()) /
                   cloud_world_ground_estimate_grid_res_));

      const uint64_t key = makeCellKey(cx, cy);
      const auto it = cell_min_z.find(key);

      if (it == cell_min_z.end()) {
        cell_min_z[key] = rel_z;
      } else {
        it->second = std::min(it->second, rel_z);
      }
    }

    if (cell_min_z.size() < 8) {
      if (std::isfinite(cloud_world_ground_rel_z_estimate_)) {
        return cloud_world_ground_rel_z_estimate_;
      }

      cloud_world_ground_rel_z_estimate_ = nominal_ground_rel_z;
      return cloud_world_ground_rel_z_estimate_;
    }

    std::vector<double> mins;
    mins.reserve(cell_min_z.size());

    for (const auto & kv : cell_min_z) {
      mins.push_back(kv.second);
    }

    const size_t k = std::min(
      mins.size() - 1,
      static_cast<size_t>(
        std::floor(cloud_world_ground_estimate_percentile_ *
                   static_cast<double>(mins.size() - 1))));

    std::nth_element(mins.begin(), mins.begin() + k, mins.end());
    const double measured_ground_rel_z = mins[k];

    if (!std::isfinite(cloud_world_ground_rel_z_estimate_)) {
      cloud_world_ground_rel_z_estimate_ = measured_ground_rel_z;
    } else {
      cloud_world_ground_rel_z_estimate_ =
        (1.0 - cloud_world_ground_estimate_smooth_alpha_) *
          cloud_world_ground_rel_z_estimate_ +
        cloud_world_ground_estimate_smooth_alpha_ *
          measured_ground_rel_z;
    }

    return cloud_world_ground_rel_z_estimate_;
  }

  CloudT::Ptr makeCloudWorldForRvizOnly(
    const CloudT::Ptr & cloud_world,
    const Eigen::Vector3d & sensor_pos,
    double ground_rel_z) const {
    if (!cloud_world_ground_aligned_) {
      return cloud_world;
    }

    double dz = cloud_world_z_offset_;

    if (cloud_world_use_auto_ground_) {
      const double world_ground_z = sensor_pos.z() + ground_rel_z;
      dz = cloud_world_target_ground_z_ - world_ground_z;
    }

    if (std::abs(dz) <= 1e-9) {
      return cloud_world;
    }

    CloudT::Ptr out(new CloudT(*cloud_world));

    for (auto & p : out->points) {
      if (std::isfinite(p.z)) {
        p.z = static_cast<float>(static_cast<double>(p.z) + dz);
      }
    }

    return out;
  }

  void segmentGround(
    const CloudT::Ptr & in,
    const Eigen::Vector3d & sensor_pos,
    CloudT::Ptr & non_ground) const {
    non_ground->clear();
    non_ground->reserve(in->size());

    if (!use_elevation_ground_filter_) {
      for (const auto & p : in->points) {
        const double rel_z = p.z - sensor_pos.z();

        if (rel_z >= obstacle_z_min_ && rel_z <= obstacle_z_max_) {
          non_ground->push_back(p);
        }
      }

      return;
    }

    std::unordered_map<uint64_t, double> cell_min_z;

    const double expected_ground_z = -sensor_mount_height_;

    for (const auto & p : in->points) {
      const double rel_z = p.z - sensor_pos.z();

      if (rel_z > obstacle_z_max_) {
        continue;
      }

      if (rel_z < expected_ground_z - 1.5) {
        continue;
      }

      const int32_t cx =
        static_cast<int32_t>(std::floor((p.x - sensor_pos.x()) / elevation_grid_res_));

      const int32_t cy =
        static_cast<int32_t>(std::floor((p.y - sensor_pos.y()) / elevation_grid_res_));

      const uint64_t key = makeCellKey(cx, cy);

      auto it = cell_min_z.find(key);
      if (it == cell_min_z.end()) {
        cell_min_z[key] = rel_z;
      } else {
        it->second = std::min(it->second, rel_z);
      }
    }

    for (const auto & p : in->points) {
      const double rel_z = p.z - sensor_pos.z();

      if (rel_z > obstacle_z_max_) {
        continue;
      }

      if (rel_z < expected_ground_z - 1.5) {
        continue;
      }

      const int32_t cx =
        static_cast<int32_t>(std::floor((p.x - sensor_pos.x()) / elevation_grid_res_));

      const int32_t cy =
        static_cast<int32_t>(std::floor((p.y - sensor_pos.y()) / elevation_grid_res_));

      const uint64_t key = makeCellKey(cx, cy);
      const auto it = cell_min_z.find(key);

      if (it == cell_min_z.end()) {
        continue;
      }

      const double min_z = it->second;

      const double dx = p.x - sensor_pos.x();
      const double dy = p.y - sensor_pos.y();
      const double dist = std::sqrt(dx * dx + dy * dy);

      const double dynamic_max_ground_z =
        expected_ground_z + 0.5 + dist * elevation_max_slope_;

      const bool obstacle_cell = min_z > dynamic_max_ground_z;

      if (obstacle_cell) {
        if (rel_z >= obstacle_z_min_) {
          non_ground->push_back(p);
        }
      } else {
        if (rel_z > min_z + elevation_z_tolerance_) {
          non_ground->push_back(p);
        }
      }
    }
  }

  bool worldToGrid(double x, double y, int & gx, int & gy) const {
    gx = static_cast<int>(std::floor((x - grid_.origin_x) / resolution_));
    gy = static_cast<int>(std::floor((y - grid_.origin_y) / resolution_));

    return gx >= 0 && gy >= 0 && gx < grid_.width && gy < grid_.height;
  }

  void expandToInclude(double x, double y) {
    const double min_x = grid_.origin_x;
    const double min_y = grid_.origin_y;
    const double max_x = grid_.origin_x + grid_.width * resolution_;
    const double max_y = grid_.origin_y + grid_.height * resolution_;

    int add_left = 0;
    int add_right = 0;
    int add_bottom = 0;
    int add_top = 0;

    const int pad =
      std::max(1, static_cast<int>(std::ceil(global_map_expand_padding_m_ / resolution_)));

    if (x < min_x) {
      add_left = static_cast<int>(std::ceil((min_x - x) / resolution_)) + pad;
    }

    if (x >= max_x) {
      add_right = static_cast<int>(std::ceil((x - max_x) / resolution_)) + pad;
    }

    if (y < min_y) {
      add_bottom = static_cast<int>(std::ceil((min_y - y) / resolution_)) + pad;
    }

    if (y >= max_y) {
      add_top = static_cast<int>(std::ceil((y - max_y) / resolution_)) + pad;
    }

    if (add_left == 0 && add_right == 0 && add_bottom == 0 && add_top == 0) {
      return;
    }

    const int old_w = grid_.width;
    const int old_h = grid_.height;
    const int new_w = old_w + add_left + add_right;
    const int new_h = old_h + add_bottom + add_top;

    std::vector<float> new_lo(static_cast<size_t>(new_w) * new_h, 0.0f);

    for (int y_old = 0; y_old < old_h; ++y_old) {
      std::copy(
        grid_.log_odds.begin() + static_cast<size_t>(y_old) * old_w,
        grid_.log_odds.begin() + static_cast<size_t>(y_old) * old_w + old_w,
        new_lo.begin() + static_cast<size_t>(y_old + add_bottom) * new_w + add_left);
    }

    grid_.log_odds.swap(new_lo);
    grid_.width = new_w;
    grid_.height = new_h;
    grid_.origin_x -= static_cast<double>(add_left) * resolution_;
    grid_.origin_y -= static_cast<double>(add_bottom) * resolution_;
  }

  void updateLogOdds(int gx, int gy, float delta) {
    if (gx < 0 || gy < 0 || gx >= grid_.width || gy >= grid_.height) {
      return;
    }

    const size_t idx = static_cast<size_t>(gy) * grid_.width + gx;

    grid_.log_odds[idx] = std::clamp(
      grid_.log_odds[idx] + delta,
      static_cast<float>(log_odds_min_),
      static_cast<float>(log_odds_max_));
  }

  void collectRayFreeCells(
    int x0,
    int y0,
    int x1,
    int y1,
    std::unordered_set<int64_t> & cells) const {
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    int x = x0;
    int y = y0;

    const int max_steps =
      static_cast<int>(std::ceil(raycast_max_range_ / resolution_));

    int step = 0;

    while (x != x1 || y != y1) {
      if (step++ >= max_steps) {
        break;
      }

      if (x >= 0 && y >= 0 && x < grid_.width && y < grid_.height) {
        cells.insert(static_cast<int64_t>(y) * grid_.width + x);
      }

      const int e2 = 2 * err;

      if (e2 > -dy) {
        err -= dy;
        x += sx;
      }

      if (e2 < dx) {
        err += dx;
        y += sy;
      }
    }
  }

  void clearCurrentSelfFootprint(const Eigen::Matrix4d & T_wl) {
    if (!self_filter_en_ || !grid_initialized_) {
      return;
    }

    const Eigen::Matrix3d R = T_wl.block<3,3>(0,0);
    const Eigen::Vector3d t = T_wl.block<3,1>(0,3);

    const double min_x = self_filter_min_x_ - self_filter_clear_margin_;
    const double max_x = self_filter_max_x_ + self_filter_clear_margin_;
    const double min_y = self_filter_min_y_ - self_filter_clear_margin_;
    const double max_y = self_filter_max_y_ + self_filter_clear_margin_;

    const Eigen::Vector3d corners_l[4] = {
      Eigen::Vector3d(min_x, min_y, 0.0),
      Eigen::Vector3d(min_x, max_y, 0.0),
      Eigen::Vector3d(max_x, min_y, 0.0),
      Eigen::Vector3d(max_x, max_y, 0.0)
    };

    double wx_min = std::numeric_limits<double>::infinity();
    double wy_min = std::numeric_limits<double>::infinity();
    double wx_max = -std::numeric_limits<double>::infinity();
    double wy_max = -std::numeric_limits<double>::infinity();

    for (const auto & c : corners_l) {
      const Eigen::Vector3d pw = R * c + t;
      wx_min = std::min(wx_min, pw.x());
      wy_min = std::min(wy_min, pw.y());
      wx_max = std::max(wx_max, pw.x());
      wy_max = std::max(wy_max, pw.y());
    }

    expandToInclude(wx_min, wy_min);
    expandToInclude(wx_max, wy_max);

    int gx0, gy0, gx1, gy1;

    worldToGrid(wx_min, wy_min, gx0, gy0);
    worldToGrid(wx_max, wy_max, gx1, gy1);

    gx0 = std::clamp(gx0, 0, grid_.width - 1);
    gx1 = std::clamp(gx1, 0, grid_.width - 1);
    gy0 = std::clamp(gy0, 0, grid_.height - 1);
    gy1 = std::clamp(gy1, 0, grid_.height - 1);

    if (gx0 > gx1) {
      std::swap(gx0, gx1);
    }

    if (gy0 > gy1) {
      std::swap(gy0, gy1);
    }

    const Eigen::Matrix3d R_lw = R.transpose();
    const float clear_lo =
      static_cast<float>(std::max(log_odds_min_, self_filter_clear_log_odds_));

    for (int gy = gy0; gy <= gy1; ++gy) {
      for (int gx = gx0; gx <= gx1; ++gx) {
        const Eigen::Vector3d pw(
          grid_.origin_x + (static_cast<double>(gx) + 0.5) * resolution_,
          grid_.origin_y + (static_cast<double>(gy) + 0.5) * resolution_,
          t.z());

        const Eigen::Vector3d pl = R_lw * (pw - t);

        if (pl.x() >= min_x && pl.x() <= max_x &&
            pl.y() >= min_y && pl.y() <= max_y) {
          const size_t idx = static_cast<size_t>(gy) * grid_.width + gx;
          grid_.log_odds[idx] = std::min(grid_.log_odds[idx], clear_lo);
        }
      }
    }
  }

  void updateGlobalGridFromCurrentScan(
    const CloudT::Ptr & cloud_world,
    const Eigen::Matrix4d & T_wl) {
    const Eigen::Vector3d sensor_pos = T_wl.block<3,1>(0,3);

    std::lock_guard<std::mutex> lk(grid_mutex_);

    if (!grid_initialized_) {
      grid_.origin_x = sensor_pos.x() - 0.5 * grid_.width * resolution_;
      grid_.origin_y = sensor_pos.y() - 0.5 * grid_.height * resolution_;
      grid_initialized_ = true;
    }

    expandToInclude(sensor_pos.x(), sensor_pos.y());

    for (const auto & p : cloud_world->points) {
      expandToInclude(p.x, p.y);
    }

    clearCurrentSelfFootprint(T_wl);

    int sx, sy;
    if (!worldToGrid(sensor_pos.x(), sensor_pos.y(), sx, sy)) {
      return;
    }

    std::unordered_set<int64_t> hit_cells;
    hit_cells.reserve(cloud_world->size());

    for (const auto & p : cloud_world->points) {
      int gx, gy;
      if (!worldToGrid(p.x, p.y, gx, gy)) {
        continue;
      }

      hit_cells.insert(static_cast<int64_t>(gy) * grid_.width + gx);
    }

    std::unordered_set<int64_t> free_cells;
    free_cells.reserve(std::max<size_t>(hit_cells.size() * 8, 64));

    for (const int64_t key : hit_cells) {
      const int gx = static_cast<int>(key % grid_.width);
      const int gy = static_cast<int>(key / grid_.width);
      collectRayFreeCells(sx, sy, gx, gy, free_cells);
    }

    for (const int64_t key : hit_cells) {
      free_cells.erase(key);
    }

    for (const int64_t key : free_cells) {
      const int gx = static_cast<int>(key % grid_.width);
      const int gy = static_cast<int>(key / grid_.width);
      updateLogOdds(gx, gy, static_cast<float>(log_odds_miss_));
    }

    for (const int64_t key : hit_cells) {
      const int gx = static_cast<int>(key % grid_.width);
      const int gy = static_cast<int>(key / grid_.width);
      updateLogOdds(gx, gy, static_cast<float>(log_odds_hit_));
    }

    clearCurrentSelfFootprint(T_wl);

    frame_counter_++;

    if (decay_period_frames_ > 0 &&
        log_odds_decay_ > 0.0 &&
        frame_counter_ % decay_period_frames_ == 0) {
      applyDecay();
    }
  }

  void applyDecay() {
    const float d_pos = static_cast<float>(log_odds_decay_);
    const float d_neg = static_cast<float>(log_odds_decay_ * 0.5);

    for (auto & v : grid_.log_odds) {
      if (v > 0.0f) {
        v = std::max(0.0f, v - d_pos);
      } else if (v < 0.0f) {
        v = std::min(0.0f, v + d_neg);
      }
    }
  }

  std::vector<uint8_t> buildOccMask(double occ_prob) const {
    std::vector<uint8_t> occ(static_cast<size_t>(grid_.width) * grid_.height, 0);

    for (size_t i = 0; i < occ.size(); ++i) {
      const float lo = grid_.log_odds[i];

      if (lo <= 0.0f) {
        continue;
      }

      const float prob = 1.0f - 1.0f / (1.0f + std::exp(lo));

      if (prob >= static_cast<float>(occ_prob)) {
        occ[i] = 1;
      }
    }

    return occ;
  }

  std::vector<uint8_t> postprocessOcc(std::vector<uint8_t> occ) const {
    if (!map_postprocess_en_ || occ.empty()) {
      return occ;
    }

    for (int i = 0; i < map_morph_close_iters_; ++i) {
      occ = dilateMask(occ, grid_.width, grid_.height, 1);
      occ = erodeMask(occ, grid_.width, grid_.height, 1);
    }

    for (int i = 0; i < map_morph_open_iters_; ++i) {
      occ = erodeMask(occ, grid_.width, grid_.height, 1);
      occ = dilateMask(occ, grid_.width, grid_.height, 1);
    }

    removeSmallComponents(occ, grid_.width, grid_.height, map_min_component_cells_);

    if (map_boundary_only_en_) {
      const std::vector<uint8_t> eroded =
        erodeMask(occ, grid_.width, grid_.height, map_boundary_erode_radius_);

      for (size_t i = 0; i < occ.size(); ++i) {
        occ[i] = (occ[i] && !eroded[i]) ? 1 : 0;
      }

      for (int i = 0; i < map_boundary_dilate_iters_; ++i) {
        occ = dilateMask(occ, grid_.width, grid_.height, 1);
      }
    }

    return occ;
  }

  void publishMap() {
    std::lock_guard<std::mutex> lk(grid_mutex_);

    if (!grid_initialized_) {
      return;
    }

    const std::vector<uint8_t> occ =
      postprocessOcc(buildOccMask(map_publish_occ_prob_));

    nav_msgs::msg::OccupancyGrid msg;
    msg.header.frame_id = map_frame_;
    msg.header.stamp = last_stamp_.nanoseconds() ? last_stamp_ : now();

    msg.info.resolution = static_cast<float>(resolution_);
    msg.info.width = grid_.width;
    msg.info.height = grid_.height;
    msg.info.origin.position.x = grid_.origin_x;
    msg.info.origin.position.y = grid_.origin_y;
    msg.info.origin.position.z = 0.0;
    msg.info.origin.orientation.w = 1.0;

    msg.data.resize(static_cast<size_t>(grid_.width) * grid_.height);

    for (size_t i = 0; i < msg.data.size(); ++i) {
      if (occ[i]) {
        msg.data[i] = 100;
      } else if (grid_.log_odds[i] < 0.0f) {
        msg.data[i] = 0;
      } else {
        msg.data[i] = -1;
      }
    }

    map_pub_->publish(msg);
  }

  void publishPoseAndPath(const Eigen::Matrix4d & T_wl) {
    Eigen::Matrix4d T_wi = Eigen::Matrix4d::Identity();
    T_wi.block<3,3>(0,0) = T_wl.block<3,3>(0,0) * R_il_.transpose();
    T_wi.block<3,1>(0,3) =
      T_wl.block<3,1>(0,3) -
      T_wi.block<3,3>(0,0) * t_il_;

    Eigen::Quaterniond q(T_wi.block<3,3>(0,0));
    q.normalize();

    const Eigen::Vector3d t = T_wi.block<3,1>(0,3);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = last_stamp_;
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = body_frame_;

    odom.pose.pose.position.x = t.x();
    odom.pose.pose.position.y = t.y();
    odom.pose.pose.position.z = t.z();

    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = last_stamp_;
    tf.header.frame_id = map_frame_;
    tf.child_frame_id = body_frame_;
    tf.transform.translation.x = t.x();
    tf.transform.translation.y = t.y();
    tf.transform.translation.z = t.z();
    tf.transform.rotation = odom.pose.pose.orientation;

    tf_broadcaster_->sendTransform(tf);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = odom.header;
    ps.pose = odom.pose.pose;

    path_.header.stamp = last_stamp_;
    path_.poses.push_back(ps);

    if (path_.poses.size() > 5000) {
      path_.poses.erase(path_.poses.begin());
    }

    path_pub_->publish(path_);
  }

  bool saveMapInternal(std::string & message) {
    std::lock_guard<std::mutex> lk(grid_mutex_);

    if (!grid_initialized_) {
      message = "Map not initialized.";
      return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(save_dir_, ec);

    const std::filesystem::path pgm =
      std::filesystem::path(save_dir_) / "grid_map_2d.pgm";

    const std::filesystem::path yaml =
      std::filesystem::path(save_dir_) / "grid_map_2d.yaml";

    std::ofstream of(pgm, std::ios::binary);

    if (!of) {
      message = "Cannot open " + pgm.string();
      return false;
    }

    const double occ_th = map_publish_occ_prob_;
    const double free_th = 0.196;

    const std::vector<uint8_t> occ =
      postprocessOcc(buildOccMask(occ_th));

    of << "P5\n# grid_mapper_2d\n"
       << grid_.width << " " << grid_.height << "\n255\n";

    for (int y = grid_.height - 1; y >= 0; --y) {
      for (int x = 0; x < grid_.width; ++x) {
        const size_t idx = static_cast<size_t>(y) * grid_.width + x;

        uint8_t pixel = 205;

        if (occ[idx]) {
          pixel = 0;
        } else if (grid_.log_odds[idx] < 0.0f) {
          pixel = 254;
        }

        of.write(reinterpret_cast<const char *>(&pixel), 1);
      }
    }

    of.close();

    std::ofstream oy(yaml);

    if (!oy) {
      message = "Cannot open " + yaml.string();
      return false;
    }

    oy << "image: " << pgm.filename().string() << "\n"
       << "resolution: " << resolution_ << "\n"
       << "origin: [" << grid_.origin_x << ", " << grid_.origin_y << ", 0.0]\n"
       << "negate: 0\n"
       << "occupied_thresh: " << occ_th << "\n"
       << "free_thresh: " << free_th << "\n";

    oy.close();

    message = "Saved global map to " + save_dir_;
    return true;
  }

  void onSaveMap(
    std_srvs::srv::Trigger::Request::ConstSharedPtr,
    std_srvs::srv::Trigger::Response::SharedPtr res) {
    std::string msg;
    res->success = saveMapInternal(msg);
    res->message = msg;
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridMapper2DNode>());
  rclcpp::shutdown();
  return 0;
}