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
#include <memory>
#include <mutex>
#include <limits>
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
    q.normalize(); return q;
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(theta, w / theta));
}
}

struct ImuState {
  double t;
  Eigen::Quaterniond q_wi;
};

class GridMapper2DNode : public rclcpp::Node {
public:
  GridMapper2DNode() : Node("grid_mapper_2d") {
    declare_all_parameters();
    load_all_parameters();
    init_grid();

    gicp_.setMaxCorrespondenceDistance(gicp_max_corr_dist_);
    gicp_.setMaximumIterations(gicp_max_iters_);
    gicp_.setTransformationEpsilon(1e-6);
    gicp_.setEuclideanFitnessEpsilon(0.01);
    local_map_.reset(new CloudT);

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(), std::bind(&GridMapper2DNode::on_imu, this, _1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(), std::bind(&GridMapper2DNode::on_cloud, this, _1));

    map_pub_   = create_publisher<nav_msgs::msg::OccupancyGrid>("map", rclcpp::QoS(1).transient_local());
    odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("odom", 20);
    path_pub_  = create_publisher<nav_msgs::msg::Path>("path", 20);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_world", 5);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    save_srv_ = create_service<std_srvs::srv::Trigger>("save_map", std::bind(&GridMapper2DNode::on_save_map, this, _1, _2));
    
    process_timer_ = create_wall_timer(std::chrono::milliseconds(10), std::bind(&GridMapper2DNode::process_data, this));
    publish_timer_ = create_wall_timer(std::chrono::milliseconds(publish_period_ms_), std::bind(&GridMapper2DNode::publish_map, this));

    path_.header.frame_id = map_frame_;
    RCLCPP_INFO(get_logger(), "GridMapper2D 3.0: Cartographer 2D Optimization & 3m Filter Activated.");
  }

  ~GridMapper2DNode() override {
    std::string message; save_map_internal(message);
  }

private:
  std::string cloud_topic_, imu_topic_, map_frame_, body_frame_, save_dir_;
  double resolution_, range_max_, raycast_max_range_;
  int publish_period_ms_;
  double obstacle_z_min_, obstacle_z_max_;
  
  bool   use_elevation_ground_filter_;
  double sensor_mount_height_;
  double elevation_grid_res_;
  double elevation_z_tolerance_;
  double elevation_max_slope_;

  double voxel_leaf_scan_, voxel_leaf_map_, voxel_leaf_grid_;
  bool exclude_box_en_;
  double exclude_box_min_x_, exclude_box_max_x_, exclude_box_min_y_, exclude_box_max_y_, exclude_box_min_z_, exclude_box_max_z_;

  bool self_filter_en_;
  double self_filter_min_x_, self_filter_max_x_, self_filter_min_y_, self_filter_max_y_, self_filter_min_z_, self_filter_max_z_;
  double self_filter_clear_margin_, self_filter_clear_log_odds_;
  double self_filter_path_sample_step_, self_filter_path_min_distance_;
  int self_filter_history_max_poses_;

  // 新增：Cartographer 2D 约束开关
  bool carto_2d_optimization_;

  struct SelfPose2D {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
  };
  std::deque<SelfPose2D> self_pose_history_;
  std::unordered_set<uint64_t> self_trail_global_keys_;

  double log_odds_hit_, log_odds_miss_, log_odds_min_, log_odds_max_, log_odds_decay_;
  int decay_period_frames_, imu_init_frames_;
  bool align_world_to_gravity_;
  double gicp_max_corr_dist_;
  int gicp_max_iters_;
  double local_map_max_size_;
  bool publish_world_cloud_;
  double scan_duration_;

  std::vector<double> ext_t_param_, ext_R_param_;
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
  Eigen::Vector3d acc_acc_ = Eigen::Vector3d::Zero(), gyr_acc_ = Eigen::Vector3d::Zero(), gyr_bias_ = Eigen::Vector3d::Zero();
  double last_imu_t_ = -1.0;

  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp_;
  CloudT::Ptr local_map_;
  std::mutex local_map_mutex_;

  struct Grid {
    int width = 0, height = 0;
    double origin_x = 0.0, origin_y = 0.0;
    std::vector<float> log_odds;
  } grid_;
  bool grid_initialized_ = false;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  std::mutex grid_mutex_;
  int frame_counter_ = 0;
  nav_msgs::msg::Path path_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
  rclcpp::TimerBase::SharedPtr process_timer_, publish_timer_;

  void declare_all_parameters() {
    declare_parameter("cloud_topic", "/rslidar_points"); declare_parameter("imu_topic", "/imu/data");
    declare_parameter("map_frame", "map"); declare_parameter("body_frame", "base_link");
    declare_parameter("save_dir", "/tmp/grid_map");
    declare_parameter("resolution", 0.05); declare_parameter("range_max", 50.0);
    declare_parameter("raycast_max_range", 10.0); declare_parameter("publish_period_ms", 500);
    declare_parameter("scan_duration", 0.1); 
    declare_parameter("obstacle_z_min", -2.5); declare_parameter("obstacle_z_max", 1.0);
    
    // Cartographer 2D 级别优化开关
    declare_parameter("carto_2d_optimization", true);

    declare_parameter("use_elevation_ground_filter", true);
    declare_parameter("sensor_mount_height", 3.0);
    declare_parameter("elevation_grid_res", 0.5);
    declare_parameter("elevation_z_tolerance", 0.30);
    declare_parameter("elevation_max_slope", 0.15);

    declare_parameter("voxel_leaf_scan", 0.3); declare_parameter("voxel_leaf_map", 0.4);
    declare_parameter("voxel_leaf_grid", 0.05);
    declare_parameter("exclude_box_en", true); declare_parameter("exclude_box_min_x", -1.2); declare_parameter("exclude_box_max_x", 1.2);
    declare_parameter("exclude_box_min_y", -1.0); declare_parameter("exclude_box_max_y", 1.0); declare_parameter("exclude_box_min_z", -3.0); declare_parameter("exclude_box_max_z", 0.8);

    declare_parameter("self_filter_en", true);
    declare_parameter("self_filter_min_x", -5.0); declare_parameter("self_filter_max_x", 2.5);
    declare_parameter("self_filter_min_y", -2.2); declare_parameter("self_filter_max_y", 2.2);
    declare_parameter("self_filter_min_z", -6.5); declare_parameter("self_filter_max_z", 1.2);
    declare_parameter("self_filter_clear_margin", 0.35);
    declare_parameter("self_filter_clear_log_odds", -2.0);
    declare_parameter("self_filter_path_sample_step", 0.20);
    declare_parameter("self_filter_path_min_distance", 0.10);
    declare_parameter("self_filter_history_max_poses", 30000);

    declare_parameter("log_odds_hit", 0.85); declare_parameter("log_odds_miss", -0.40); declare_parameter("log_odds_min", -2.0); declare_parameter("log_odds_max", 3.5);
    declare_parameter("log_odds_decay", 0.02); declare_parameter("decay_period_frames", 50);
    declare_parameter("imu_init_frames", 30); declare_parameter("align_world_to_gravity", true);
    declare_parameter("gicp_max_correspondence_distance", 1.0); declare_parameter("gicp_max_iterations", 15);
    declare_parameter("local_map_max_size", 30.0); declare_parameter("publish_world_cloud", true);
    declare_parameter("extrinsic_T_lidar_in_imu", std::vector<double>{0,0,0});
    declare_parameter("extrinsic_R_lidar_in_imu", std::vector<double>{1,0,0, 0,1,0, 0,0,1});
  }

  void load_all_parameters() {
    cloud_topic_ = get_parameter("cloud_topic").as_string(); imu_topic_ = get_parameter("imu_topic").as_string();
    map_frame_ = get_parameter("map_frame").as_string(); body_frame_ = get_parameter("body_frame").as_string();
    save_dir_ = get_parameter("save_dir").as_string();
    resolution_ = get_parameter("resolution").as_double(); range_max_ = get_parameter("range_max").as_double();
    raycast_max_range_ = get_parameter("raycast_max_range").as_double(); publish_period_ms_ = get_parameter("publish_period_ms").as_int();
    scan_duration_ = get_parameter("scan_duration").as_double();
    obstacle_z_min_ = get_parameter("obstacle_z_min").as_double(); obstacle_z_max_ = get_parameter("obstacle_z_max").as_double();
    
    carto_2d_optimization_ = get_parameter("carto_2d_optimization").as_bool();

    use_elevation_ground_filter_ = get_parameter("use_elevation_ground_filter").as_bool();
    sensor_mount_height_ = get_parameter("sensor_mount_height").as_double();
    elevation_grid_res_ = get_parameter("elevation_grid_res").as_double();
    elevation_z_tolerance_ = get_parameter("elevation_z_tolerance").as_double();
    elevation_max_slope_ = get_parameter("elevation_max_slope").as_double();

    voxel_leaf_scan_ = get_parameter("voxel_leaf_scan").as_double(); voxel_leaf_map_ = get_parameter("voxel_leaf_map").as_double(); voxel_leaf_grid_ = get_parameter("voxel_leaf_grid").as_double();
    exclude_box_en_ = get_parameter("exclude_box_en").as_bool();
    exclude_box_min_x_ = get_parameter("exclude_box_min_x").as_double(); exclude_box_max_x_ = get_parameter("exclude_box_max_x").as_double();
    exclude_box_min_y_ = get_parameter("exclude_box_min_y").as_double(); exclude_box_max_y_ = get_parameter("exclude_box_max_y").as_double();
    exclude_box_min_z_ = get_parameter("exclude_box_min_z").as_double(); exclude_box_max_z_ = get_parameter("exclude_box_max_z").as_double();

    self_filter_en_ = get_parameter("self_filter_en").as_bool();
    self_filter_min_x_ = get_parameter("self_filter_min_x").as_double(); self_filter_max_x_ = get_parameter("self_filter_max_x").as_double();
    self_filter_min_y_ = get_parameter("self_filter_min_y").as_double(); self_filter_max_y_ = get_parameter("self_filter_max_y").as_double();
    self_filter_min_z_ = get_parameter("self_filter_min_z").as_double(); self_filter_max_z_ = get_parameter("self_filter_max_z").as_double();
    self_filter_clear_margin_ = get_parameter("self_filter_clear_margin").as_double();
    self_filter_clear_log_odds_ = get_parameter("self_filter_clear_log_odds").as_double();
    self_filter_path_sample_step_ = std::max(0.02, get_parameter("self_filter_path_sample_step").as_double());
    self_filter_path_min_distance_ = std::max(0.0, get_parameter("self_filter_path_min_distance").as_double());
    const int64_t history_max_param = get_parameter("self_filter_history_max_poses").as_int();
    self_filter_history_max_poses_ = static_cast<int>(std::max<int64_t>(2, history_max_param));

    log_odds_hit_ = get_parameter("log_odds_hit").as_double(); log_odds_miss_ = get_parameter("log_odds_miss").as_double();
    log_odds_min_ = get_parameter("log_odds_min").as_double(); log_odds_max_ = get_parameter("log_odds_max").as_double();
    log_odds_decay_ = get_parameter("log_odds_decay").as_double(); decay_period_frames_ = get_parameter("decay_period_frames").as_int();
    imu_init_frames_ = get_parameter("imu_init_frames").as_int(); align_world_to_gravity_ = get_parameter("align_world_to_gravity").as_bool();
    gicp_max_corr_dist_ = get_parameter("gicp_max_correspondence_distance").as_double(); gicp_max_iters_ = get_parameter("gicp_max_iterations").as_int();
    local_map_max_size_ = get_parameter("local_map_max_size").as_double(); publish_world_cloud_ = get_parameter("publish_world_cloud").as_bool();
    auto et = get_parameter("extrinsic_T_lidar_in_imu").as_double_array(); if(et.size()==3) t_il_ << et[0],et[1],et[2];
    auto eR = get_parameter("extrinsic_R_lidar_in_imu").as_double_array();
    if(eR.size()==9) R_il_ << eR[0],eR[1],eR[2], eR[3],eR[4],eR[5], eR[6],eR[7],eR[8];
  }

  void init_grid() {
    grid_.width = 200; grid_.height = 200;
    grid_.origin_x = -grid_.width * resolution_ * 0.5; grid_.origin_y = -grid_.height * resolution_ * 0.5;
    grid_.log_odds.assign(grid_.width * grid_.height, 0.0f); grid_initialized_ = false;
  }

  Eigen::Quaterniond get_q_wi(double t) {
    if (imu_history_.empty()) return Eigen::Quaterniond::Identity();
    if (t <= imu_history_.front().t) return imu_history_.front().q_wi;
    if (t >= imu_history_.back().t) return imu_history_.back().q_wi;
    
    int left = 0, right = imu_history_.size() - 1;
    while (left < right - 1) {
      int mid = left + (right - left) / 2;
      if (imu_history_[mid].t <= t) left = mid; else right = mid;
    }
    double dt = imu_history_[right].t - imu_history_[left].t;
    double ratio = (t - imu_history_[left].t) / dt;
    return imu_history_[left].q_wi.slerp(ratio, imu_history_[right].q_wi);
  }

  void on_imu(sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    const Eigen::Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    const Eigen::Vector3d gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    const double t = rclcpp::Time(msg->header.stamp).seconds();

    std::lock_guard<std::mutex> lk(data_mutex_);
    if (!imu_inited_) {
      acc_acc_ += acc; gyr_acc_ += gyr; imu_init_count_++;
      if (imu_init_count_ >= imu_init_frames_) {
        gyr_bias_ = gyr_acc_ / imu_init_count_;
        if (align_world_to_gravity_) {
          q_wi_ = Eigen::Quaterniond::FromTwoVectors((acc_acc_ / imu_init_count_).normalized(), Eigen::Vector3d(0.0, 0.0, -1.0));
        } else {
          q_wi_.setIdentity();
        }
        last_imu_t_ = t; imu_inited_ = true;
      }
      return;
    }
    double dt = t - last_imu_t_;
    if (dt > 0.0 && dt <= 0.2) {
      q_wi_ = (q_wi_ * axisAngleToQuat((gyr - gyr_bias_) * dt)).normalized();
      imu_history_.push_back({t, q_wi_});
      while(imu_history_.size() > 500) imu_history_.pop_front();
    }
    last_imu_t_ = t;
  }

  void on_cloud(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(data_mutex_);
    cloud_buf_.push_back(msg);
  }

  void process_data() {
    std::lock_guard<std::mutex> lk(data_mutex_);
    if (cloud_buf_.empty() || !imu_inited_) return;

    auto cloud_msg = cloud_buf_.front();
    double t_start = rclcpp::Time(cloud_msg->header.stamp).seconds();
    double t_end = t_start + scan_duration_;

    if (imu_history_.empty() || imu_history_.back().t < t_end) return;
    
    cloud_buf_.pop_front();
    
    CloudT::Ptr cloud_raw(new CloudT);
    pcl::fromROSMsg(*cloud_msg, *cloud_raw);
    if (cloud_raw->empty()) return;

    CloudT::Ptr cloud_clean(new CloudT);
    cloud_clean->reserve(cloud_raw->size());
    const double r2_max = range_max_ * range_max_;
    for (const auto & p : cloud_raw->points) {
      if (!std::isfinite(p.x)) continue;
      if (exclude_box_en_ && p.x >= exclude_box_min_x_ && p.x <= exclude_box_max_x_ && p.y >= exclude_box_min_y_ && p.y <= exclude_box_max_y_ && p.z >= exclude_box_min_z_ && p.z <= exclude_box_max_z_) continue;
      if (point_in_self_filter_box_lidar(Eigen::Vector3d(p.x, p.y, p.z))) continue;
      if (p.x*p.x + p.y*p.y + p.z*p.z > r2_max) continue;
      cloud_clean->push_back(p);
    }
    if (cloud_clean->empty()) return;

    CloudT::Ptr cloud_deskewed(new CloudT);
    cloud_deskewed->reserve(cloud_clean->size());
    
    Eigen::Quaterniond q_wi_end = get_q_wi(t_end);
    Eigen::Vector3d p_wl_end_pred = p_wl_last_ + v_wl_last_ * (t_end - last_cloud_time_);
    Eigen::Matrix4d T_wl_end_pred = Eigen::Matrix4d::Identity();
    T_wl_end_pred.block<3,3>(0,0) = q_wi_end.toRotationMatrix() * R_il_;
    T_wl_end_pred.block<3,1>(0,3) = p_wl_end_pred;
    Eigen::Matrix4d T_lw_end_pred = T_wl_end_pred.inverse();

    for (size_t i = 0; i < cloud_clean->size(); ++i) {
      const auto & pt = cloud_clean->points[i];
      double t_j = t_start + (static_cast<double>(i) / cloud_clean->size()) * scan_duration_;
      
      Eigen::Quaterniond q_wi_j = get_q_wi(t_j);
      Eigen::Vector3d p_wl_j = p_wl_last_ + v_wl_last_ * (t_j - last_cloud_time_);
      
      Eigen::Matrix4d T_wl_j = Eigen::Matrix4d::Identity();
      T_wl_j.block<3,3>(0,0) = q_wi_j.toRotationMatrix() * R_il_;
      T_wl_j.block<3,1>(0,3) = p_wl_j;
      
      Eigen::Vector3d pt_w = T_wl_j.block<3,3>(0,0) * Eigen::Vector3d(pt.x, pt.y, pt.z) + T_wl_j.block<3,1>(0,3);
      Eigen::Vector3d pt_deskewed = T_lw_end_pred.block<3,3>(0,0) * pt_w + T_lw_end_pred.block<3,1>(0,3);
      
      PointT pt_out = pt;
      pt_out.x = pt_deskewed.x(); pt_out.y = pt_deskewed.y(); pt_out.z = pt_deskewed.z();
      cloud_deskewed->push_back(pt_out);
    }

    CloudT::Ptr cloud_ds(new CloudT); 
    pcl::VoxelGrid<PointT> vg_scan; vg_scan.setLeafSize(voxel_leaf_scan_, voxel_leaf_scan_, voxel_leaf_scan_);
    vg_scan.setInputCloud(cloud_deskewed); vg_scan.filter(*cloud_ds);

    CloudT::Ptr cloud_grid(new CloudT); 
    pcl::VoxelGrid<PointT> vg_grid; vg_grid.setLeafSize(voxel_leaf_grid_, voxel_leaf_grid_, voxel_leaf_grid_);
    vg_grid.setInputCloud(cloud_deskewed); vg_grid.filter(*cloud_grid);

    Eigen::Matrix4d T_wl_final = T_wl_end_pred; 

    if (first_cloud_) {
      last_cloud_time_ = t_end;
      p_wl_last_ = Eigen::Vector3d::Zero();
      v_wl_last_ = Eigen::Vector3d::Zero();
      T_wl_final.block<3,1>(0,3) = p_wl_last_;
      
      CloudT::Ptr cloud_world(new CloudT);
      pcl::transformPointCloud(*cloud_ds, *cloud_world, T_wl_final.cast<float>());
      CloudT::Ptr cloud_world_self_filtered(new CloudT);
      remove_self_points_from_cloud(cloud_world, T_wl_final, cloud_world_self_filtered);
      {
        std::lock_guard<std::mutex> lk_map(local_map_mutex_);
        *local_map_ = *cloud_world_self_filtered;
      }
      if (self_filter_en_ && self_pose_history_.empty()) {
        self_pose_history_.push_back(self_pose_from_T(T_wl_final));
      }
      first_cloud_ = false;
      return; 
    }

    CloudT::Ptr cloud_ds_world(new CloudT);
    pcl::transformPointCloud(*cloud_ds, *cloud_ds_world, T_wl_end_pred.cast<float>());

    {
      std::lock_guard<std::mutex> lk_map(local_map_mutex_);
      if (!local_map_->empty() && cloud_ds_world->size() > 20) {
        gicp_.setInputSource(cloud_ds_world);
        gicp_.setInputTarget(local_map_);
        CloudT::Ptr cloud_aligned(new CloudT);
        gicp_.align(*cloud_aligned, Eigen::Matrix4f::Identity());
        
        if (gicp_.hasConverged()) {
          Eigen::Matrix4d T_delta = gicp_.getFinalTransformation().cast<double>();

          // =========================================================================
          // Cartographer 2D 核心优化：降维约束
          // =========================================================================
          if (carto_2d_optimization_) {
            // 1. 强制去除点云匹配中 Z 轴的位移以及 Roll / Pitch 旋转变化
            // 提取匹配计算出的 Yaw 角度
            double delta_yaw = std::atan2(T_delta(1,0), T_delta(0,0));
            Eigen::Vector3d delta_t = T_delta.block<3,1>(0,3);
            
            // 2. 重新构建绝对干净的 2D 变换矩阵
            T_delta.setIdentity();
            T_delta(0,3) = delta_t.x();
            T_delta(1,3) = delta_t.y();
            T_delta(2,3) = 0.0; // 绝对锁死 Z 轴匹配漂移
            
            T_delta(0,0) = std::cos(delta_yaw); T_delta(0,1) = -std::sin(delta_yaw);
            T_delta(1,0) = std::sin(delta_yaw); T_delta(1,1) =  std::cos(delta_yaw);
          }

          T_wl_final = T_delta * T_wl_end_pred;
        }
      }
    }

    // =========================================================================
    // 阻断积分系统级 Z 漂移
    // =========================================================================
    v_wl_last_ = (T_wl_final.block<3,1>(0,3) - p_wl_last_) / (t_end - last_cloud_time_);
    
    if (carto_2d_optimization_) {
      v_wl_last_.z() = 0.0;     // 速度 Z 置零，彻底切断物理预测积分漂移
      T_wl_final(2, 3) = 0.0;   // 世界系坐标 Z 强制归零（建图高度恒定）
    }

    p_wl_last_ = T_wl_final.block<3,1>(0,3);
    last_cloud_time_ = t_end;

    // 因为前面使用了 carto_2d_optimization_ 剔除了 Roll/Pitch 匹配修正
    // 这里的 q_err 计算出来的将是纯粹的 Yaw 修正。
    // 这意味着你的机器人在 Roll/Pitch 上永远 100% 服从并信任 IMU 算出的重力向量！极其稳定。
    Eigen::Quaterniond q_wl_final(T_wl_final.block<3,3>(0,0));
    Eigen::Quaterniond q_wi_final = q_wl_final * Eigen::Quaterniond(R_il_).inverse();
    Eigen::Quaterniond q_err = q_wi_final * get_q_wi(t_end).inverse();
    
    for (auto & state : imu_history_) { state.q_wi = q_err * state.q_wi; }
    q_wi_ = q_err * q_wi_; 

    CloudT::Ptr cloud_world_final(new CloudT);
    pcl::transformPointCloud(*cloud_ds, *cloud_world_final, T_wl_final.cast<float>());
    CloudT::Ptr cloud_world_final_self_filtered(new CloudT);
    remove_self_points_from_cloud(cloud_world_final, T_wl_final, cloud_world_final_self_filtered);
    {
      std::lock_guard<std::mutex> lk_map(local_map_mutex_);
      *local_map_ += *cloud_world_final_self_filtered;
      crop_local_map(T_wl_final.block<3,1>(0,3));
      voxel_downsample(local_map_, voxel_leaf_map_);
    }

    CloudT::Ptr cloud_grid_world(new CloudT);
    pcl::transformPointCloud(*cloud_grid, *cloud_grid_world, T_wl_final.cast<float>());
    
    CloudT::Ptr cloud_grid_ng(new CloudT);
    segment_ground(cloud_grid_world, T_wl_final.block<3,1>(0,3), cloud_grid_ng);

    CloudT::Ptr cloud_grid_ng_self_filtered(new CloudT);
    remove_self_points_from_cloud(cloud_grid_ng, T_wl_final, cloud_grid_ng_self_filtered);
    update_grid_from_cloud(cloud_grid_ng_self_filtered, T_wl_final);

    last_stamp_ = cloud_msg->header.stamp;
    publish_pose_and_path(T_wl_final);
    if (publish_world_cloud_) {
      sensor_msgs::msg::PointCloud2 msg_out; pcl::toROSMsg(*cloud_grid_ng_self_filtered, msg_out);
      msg_out.header.stamp = last_stamp_; msg_out.header.frame_id = map_frame_; cloud_pub_->publish(msg_out);
    }
  }

  void segment_ground(const CloudT::Ptr & in, const Eigen::Vector3d & sensor_pos, CloudT::Ptr & non_ground) {
    non_ground->reserve(in->size());

    if (!use_elevation_ground_filter_) {
      for (const auto & p : in->points) {
        double rel_z = p.z - sensor_pos.z();
        if (rel_z >= obstacle_z_min_ && rel_z <= obstacle_z_max_) non_ground->push_back(p);
      }
      return;
    }

    std::unordered_map<uint64_t, double> cell_min_z;
    const double grid_res = elevation_grid_res_; 
    const double expected_ground_z = -sensor_mount_height_;

    for (const auto & p : in->points) {
      double rel_z = p.z - sensor_pos.z();
      if (rel_z > obstacle_z_max_) continue; 
      if (rel_z < expected_ground_z - 1.5) continue; 

      int32_t cx = static_cast<int32_t>(std::floor((p.x - sensor_pos.x()) / grid_res));
      int32_t cy = static_cast<int32_t>(std::floor((p.y - sensor_pos.y()) / grid_res));
      uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32) | static_cast<uint32_t>(cy);

      if (cell_min_z.find(key) == cell_min_z.end()) {
        cell_min_z[key] = rel_z;
      } else {
        if (rel_z < cell_min_z[key]) cell_min_z[key] = rel_z;
      }
    }

    for (const auto & p : in->points) {
      double rel_z = p.z - sensor_pos.z();
      if (rel_z > obstacle_z_max_) continue;
      if (rel_z < expected_ground_z - 1.5) continue;

      int32_t cx = static_cast<int32_t>(std::floor((p.x - sensor_pos.x()) / grid_res));
      int32_t cy = static_cast<int32_t>(std::floor((p.y - sensor_pos.y()) / grid_res));
      uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32) | static_cast<uint32_t>(cy);

      if (cell_min_z.find(key) != cell_min_z.end()) {
        double min_z = cell_min_z[key];

        double dx = p.x - sensor_pos.x();
        double dy = p.y - sensor_pos.y();
        double dist = std::sqrt(dx*dx + dy*dy);

        double dynamic_max_ground_z = expected_ground_z + 0.5 + dist * elevation_max_slope_;
        bool is_obstacle_cell = (min_z > dynamic_max_ground_z);

        if (is_obstacle_cell) {
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
  }

  bool point_in_self_filter_box_lidar(const Eigen::Vector3d & p_l) const {
    if (!self_filter_en_) return false;
    return p_l.x() >= self_filter_min_x_ && p_l.x() <= self_filter_max_x_ &&
           p_l.y() >= self_filter_min_y_ && p_l.y() <= self_filter_max_y_ &&
           p_l.z() >= self_filter_min_z_ && p_l.z() <= self_filter_max_z_;
  }

  void remove_self_points_from_cloud(const CloudT::Ptr & in_world,
                                     const Eigen::Matrix4d & T_wl,
                                     CloudT::Ptr & out_world) const {
    out_world->clear();
    out_world->reserve(in_world->size());

    if (!self_filter_en_) {
      *out_world = *in_world;
      return;
    }

    const Eigen::Matrix3d R_lw = T_wl.block<3,3>(0,0).transpose();
    const Eigen::Vector3d t_wl = T_wl.block<3,1>(0,3);

    for (const auto & p : in_world->points) {
      const Eigen::Vector3d p_w(p.x, p.y, p.z);
      const Eigen::Vector3d p_l = R_lw * (p_w - t_wl);
      if (point_in_self_filter_box_lidar(p_l)) continue;
      out_world->push_back(p);
    }
    out_world->width = static_cast<uint32_t>(out_world->size());
    out_world->height = 1;
    out_world->is_dense = false;
  }

  static double normalize_angle(double a) {
    constexpr double kPi = 3.14159265358979323846;
    while (a > kPi) a -= 2.0 * kPi;
    while (a < -kPi) a += 2.0 * kPi;
    return a;
  }

  SelfPose2D self_pose_from_T(const Eigen::Matrix4d & T_wl) const {
    SelfPose2D p;
    p.x = T_wl(0, 3);
    p.y = T_wl(1, 3);
    p.yaw = std::atan2(T_wl(1, 0), T_wl(0, 0));
    return p;
  }

  SelfPose2D interpolate_self_pose(const SelfPose2D & a, const SelfPose2D & b, double r) const {
    SelfPose2D out;
    out.x = a.x + r * (b.x - a.x);
    out.y = a.y + r * (b.y - a.y);
    out.yaw = normalize_angle(a.yaw + r * normalize_angle(b.yaw - a.yaw));
    return out;
  }

  static uint64_t make_global_cell_key(int32_t gx, int32_t gy) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(gx)) << 32) |
           static_cast<uint32_t>(gy);
  }

  uint64_t world_to_global_cell_key(double x, double y) const {
    const int32_t gx = static_cast<int32_t>(std::floor(x / resolution_));
    const int32_t gy = static_cast<int32_t>(std::floor(y / resolution_));
    return make_global_cell_key(gx, gy);
  }

  uint64_t local_grid_to_global_cell_key(int gx, int gy) const {
    const double wx = grid_.origin_x + (static_cast<double>(gx) + 0.5) * resolution_;
    const double wy = grid_.origin_y + (static_cast<double>(gy) + 0.5) * resolution_;
    return world_to_global_cell_key(wx, wy);
  }

  bool is_self_trail_cell_locked(int gx, int gy) const {
    if (!self_filter_en_) return false;
    return self_trail_global_keys_.find(local_grid_to_global_cell_key(gx, gy)) != self_trail_global_keys_.end();
  }

  void self_pose_xy_axes(const SelfPose2D & pose, Eigen::Vector2d & x_axis, Eigen::Vector2d & y_axis) const {
    x_axis = Eigen::Vector2d(std::cos(pose.yaw), std::sin(pose.yaw));
    y_axis = Eigen::Vector2d(-x_axis.y(), x_axis.x());
  }

  void expand_to_include_self_footprint_locked(const SelfPose2D & pose) {
    if (!self_filter_en_) return;

    Eigen::Vector2d x_axis, y_axis;
    self_pose_xy_axes(pose, x_axis, y_axis);
    const Eigen::Vector2d origin(pose.x, pose.y);

    const double min_x = self_filter_min_x_ - self_filter_clear_margin_;
    const double max_x = self_filter_max_x_ + self_filter_clear_margin_;
    const double min_y = self_filter_min_y_ - self_filter_clear_margin_;
    const double max_y = self_filter_max_y_ + self_filter_clear_margin_;

    const double xs[2] = {min_x, max_x};
    const double ys[2] = {min_y, max_y};
    for (double lx : xs) {
      for (double ly : ys) {
        const Eigen::Vector2d p_w = origin + lx * x_axis + ly * y_axis;
        expand_to_include(p_w.x(), p_w.y());
      }
    }
  }

  void clear_self_footprint_pose_locked(const SelfPose2D & pose) {
    if (!self_filter_en_ || !grid_initialized_) return;

    expand_to_include_self_footprint_locked(pose);

    Eigen::Vector2d x_axis, y_axis;
    self_pose_xy_axes(pose, x_axis, y_axis);
    const Eigen::Vector2d origin(pose.x, pose.y);

    const double min_x = self_filter_min_x_ - self_filter_clear_margin_;
    const double max_x = self_filter_max_x_ + self_filter_clear_margin_;
    const double min_y = self_filter_min_y_ - self_filter_clear_margin_;
    const double max_y = self_filter_max_y_ + self_filter_clear_margin_;

    const double xs[2] = {min_x, max_x};
    const double ys[2] = {min_y, max_y};
    double wx_min = std::numeric_limits<double>::infinity();
    double wy_min = std::numeric_limits<double>::infinity();
    double wx_max = -std::numeric_limits<double>::infinity();
    double wy_max = -std::numeric_limits<double>::infinity();
    for (double lx : xs) {
      for (double ly : ys) {
        const Eigen::Vector2d p_w = origin + lx * x_axis + ly * y_axis;
        wx_min = std::min(wx_min, p_w.x()); wx_max = std::max(wx_max, p_w.x());
        wy_min = std::min(wy_min, p_w.y()); wy_max = std::max(wy_max, p_w.y());
      }
    }

    int gx0 = static_cast<int>(std::floor((wx_min - grid_.origin_x) / resolution_));
    int gx1 = static_cast<int>(std::floor((wx_max - grid_.origin_x) / resolution_));
    int gy0 = static_cast<int>(std::floor((wy_min - grid_.origin_y) / resolution_));
    int gy1 = static_cast<int>(std::floor((wy_max - grid_.origin_y) / resolution_));
    gx0 = std::clamp(gx0, 0, grid_.width - 1);  gx1 = std::clamp(gx1, 0, grid_.width - 1);
    gy0 = std::clamp(gy0, 0, grid_.height - 1); gy1 = std::clamp(gy1, 0, grid_.height - 1);

    const float clear_lo = static_cast<float>(std::max(log_odds_min_, self_filter_clear_log_odds_));
    for (int gy = gy0; gy <= gy1; ++gy) {
      for (int gx = gx0; gx <= gx1; ++gx) {
        const Eigen::Vector2d p_w(grid_.origin_x + (gx + 0.5) * resolution_,
                                  grid_.origin_y + (gy + 0.5) * resolution_);
        const Eigen::Vector2d d = p_w - origin;
        const double lx = d.dot(x_axis);
        const double ly = d.dot(y_axis);
        if (lx >= min_x && lx <= max_x && ly >= min_y && ly <= max_y) {
          const size_t idx = static_cast<size_t>(gy) * grid_.width + gx;
          grid_.log_odds[idx] = std::min(grid_.log_odds[idx], clear_lo);
          self_trail_global_keys_.insert(local_grid_to_global_cell_key(gx, gy));
        }
      }
    }
  }

  void clear_self_swept_segment_locked(const SelfPose2D & a, const SelfPose2D & b) {
    if (!self_filter_en_ || !grid_initialized_) return;

    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double dist = std::sqrt(dx * dx + dy * dy);
    const double dyaw = std::abs(normalize_angle(b.yaw - a.yaw));
    const int n_pos = static_cast<int>(std::ceil(dist / self_filter_path_sample_step_));
    const int n_yaw = static_cast<int>(std::ceil(dyaw / 0.15));
    const int n = std::max(1, std::max(n_pos, n_yaw));

    for (int i = 0; i <= n; ++i) {
      const double r = static_cast<double>(i) / static_cast<double>(n);
      clear_self_footprint_pose_locked(interpolate_self_pose(a, b, r));
    }
  }

  void record_and_clear_self_trail_locked(const Eigen::Matrix4d & T_wl) {
    if (!self_filter_en_ || !grid_initialized_) return;

    const SelfPose2D cur = self_pose_from_T(T_wl);
    if (self_pose_history_.empty()) {
      self_pose_history_.push_back(cur);
      clear_self_footprint_pose_locked(cur);
      return;
    }

    const SelfPose2D last = self_pose_history_.back();
    const double dx = cur.x - last.x;
    const double dy = cur.y - last.y;
    const double dist = std::sqrt(dx * dx + dy * dy);
    const double dyaw = std::abs(normalize_angle(cur.yaw - last.yaw));

    if (dist >= self_filter_path_min_distance_ || dyaw >= 0.05) {
      clear_self_swept_segment_locked(last, cur);
      self_pose_history_.push_back(cur);
      while (static_cast<int>(self_pose_history_.size()) > self_filter_history_max_poses_) {
        self_pose_history_.pop_front();
      }
    } else {
      clear_self_footprint_pose_locked(cur);
    }
  }

  void voxel_downsample(CloudT::Ptr & cloud, double leaf) {
    if (leaf <= 0.0 || cloud->empty()) return;
    pcl::VoxelGrid<PointT> vg; vg.setLeafSize(leaf, leaf, leaf);
    CloudT::Ptr tmp(new CloudT); vg.setInputCloud(cloud); vg.filter(*tmp); cloud->swap(*tmp);
  }

  void crop_local_map(const Eigen::Vector3d & center) {
    if (local_map_->empty()) return;
    const double r2 = local_map_max_size_ * local_map_max_size_;
    CloudT::Ptr cropped(new CloudT); cropped->reserve(local_map_->size());
    for (const auto & p : local_map_->points) {
      double dx = p.x - center.x(), dy = p.y - center.y();
      if (dx*dx + dy*dy < r2) cropped->push_back(p);
    }
    local_map_->swap(*cropped);
  }

  void update_grid_from_cloud(const CloudT::Ptr & cloud_ng_world, const Eigen::Matrix4d & T_wl) {
    const Eigen::Vector3d sensor_pos = T_wl.block<3,1>(0,3);

    std::lock_guard<std::mutex> lk(grid_mutex_);
    if (!grid_initialized_) {
      grid_.origin_x = sensor_pos.x() - grid_.width * resolution_ * 0.5;
      grid_.origin_y = sensor_pos.y() - grid_.height * resolution_ * 0.5;
      grid_initialized_ = true;
    }

    expand_to_include(sensor_pos.x(), sensor_pos.y());
    for (const auto & p : cloud_ng_world->points) expand_to_include(p.x, p.y);

    record_and_clear_self_trail_locked(T_wl);

    if (cloud_ng_world->empty()) {
      frame_counter_++;
      if (decay_period_frames_ > 0 && frame_counter_ % decay_period_frames_ == 0) apply_decay();
      return;
    }

    int sx, sy; if (!world_to_grid(sensor_pos.x(), sensor_pos.y(), sx, sy)) return;

    std::unordered_set<long long> hit_cells; hit_cells.reserve(cloud_ng_world->size());
    for (const auto & p : cloud_ng_world->points) {
      int gx, gy; if (!world_to_grid(p.x, p.y, gx, gy)) continue;
      if (is_self_trail_cell_locked(gx, gy)) continue;
      hit_cells.insert(static_cast<long long>(gy) * grid_.width + gx);
    }

    std::unordered_set<long long> free_cells; free_cells.reserve(std::max<size_t>(hit_cells.size() * 8, 64));
    for (long long key : hit_cells) {
      collect_ray_free_cells(sx, sy, static_cast<int>(key % grid_.width), static_cast<int>(key / grid_.width), free_cells);
    }
    for (long long key : hit_cells) free_cells.erase(key);

    const float hit_delta = static_cast<float>(log_odds_hit_), miss_delta = static_cast<float>(log_odds_miss_);
    for (long long key : hit_cells) {
      const int gx = static_cast<int>(key % grid_.width);
      const int gy = static_cast<int>(key / grid_.width);
      if (!is_self_trail_cell_locked(gx, gy)) update_log_odds(gx, gy, hit_delta);
    }
    for (long long key : free_cells) update_log_odds(key % grid_.width, key / grid_.width, miss_delta);

    record_and_clear_self_trail_locked(T_wl);

    frame_counter_++;
    if (decay_period_frames_ > 0 && frame_counter_ % decay_period_frames_ == 0) apply_decay();
  }

  inline void update_log_odds(int gx, int gy, float delta) {
    if (gx < 0 || gy < 0 || gx >= grid_.width || gy >= grid_.height) return;
    const size_t idx = static_cast<size_t>(gy) * grid_.width + gx;
    grid_.log_odds[idx] = std::clamp(grid_.log_odds[idx] + delta, static_cast<float>(log_odds_min_), static_cast<float>(log_odds_max_));
  }

  void collect_ray_free_cells(int x0, int y0, int x1, int y1, std::unordered_set<long long> & cells) const {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy, x = x0, y = y0;
    int max_steps = static_cast<int>(std::ceil(raycast_max_range_ / resolution_));
    int step = 0;
    while (x != x1 || y != y1) {
      if (step++ >= max_steps) break;
      if (x >= 0 && y >= 0 && x < grid_.width && y < grid_.height) cells.insert(static_cast<long long>(y) * grid_.width + x);
      int e2 = 2 * err;
      if (e2 > -dy) { err -= dy; x += sx; }
      if (e2 <  dx) { err += dx; y += sy; }
    }
  }

  bool world_to_grid(double x, double y, int & gx, int & gy) const {
    gx = static_cast<int>(std::floor((x - grid_.origin_x) / resolution_));
    gy = static_cast<int>(std::floor((y - grid_.origin_y) / resolution_));
    return gx >= 0 && gy >= 0 && gx < grid_.width && gy < grid_.height;
  }

  void expand_to_include(double x, double y) {
    const double min_x = grid_.origin_x, min_y = grid_.origin_y;
    const double max_x = grid_.origin_x + grid_.width * resolution_, max_y = grid_.origin_y + grid_.height * resolution_;
    int add_left = 0, add_right = 0, add_bottom = 0, add_top = 0, pad = 200;
    if (x < min_x) add_left = static_cast<int>(std::ceil((min_x - x) / resolution_)) + pad;
    if (x >= max_x) add_right = static_cast<int>(std::ceil((x - max_x) / resolution_)) + pad;
    if (y < min_y) add_bottom = static_cast<int>(std::ceil((min_y - y) / resolution_)) + pad;
    if (y >= max_y) add_top = static_cast<int>(std::ceil((y - max_y) / resolution_)) + pad;
    if (add_left == 0 && add_right == 0 && add_bottom == 0 && add_top == 0) return;

    const int new_w = grid_.width + add_left + add_right, new_h = grid_.height + add_bottom + add_top;
    std::vector<float> new_lo(static_cast<size_t>(new_w) * new_h, 0.0f);
    for (int oy = 0; oy < grid_.height; ++oy) {
      std::copy(grid_.log_odds.begin() + static_cast<size_t>(oy) * grid_.width,
                grid_.log_odds.begin() + static_cast<size_t>(oy) * grid_.width + grid_.width,
                new_lo.begin() + static_cast<size_t>(oy + add_bottom) * new_w + add_left);
    }
    grid_.log_odds.swap(new_lo); grid_.width = new_w; grid_.height = new_h;
    grid_.origin_x -= add_left * resolution_; grid_.origin_y -= add_bottom * resolution_;
  }

  void apply_decay() {
    const float d_pos = static_cast<float>(log_odds_decay_), d_neg = static_cast<float>(log_odds_decay_) * 0.5f;
    for (auto & v : grid_.log_odds) {
      if (v > 0.0f) v = std::max(0.0f, v - d_pos); else if (v < 0.0f) v = std::min(0.0f, v + d_neg);
    }
  }

  void publish_pose_and_path(const Eigen::Matrix4d & T_wl) {
    Eigen::Matrix4d T_wi = T_wl * Eigen::Matrix4d::Identity();
    T_wi.block<3,3>(0,0) = T_wl.block<3,3>(0,0) * R_il_.transpose();
    T_wi.block<3,1>(0,3) = T_wl.block<3,1>(0,3) - T_wi.block<3,3>(0,0) * t_il_;

    Eigen::Quaterniond q(T_wi.block<3,3>(0,0)); q.normalize();
    const Eigen::Vector3d t = T_wi.block<3,1>(0,3);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = last_stamp_; odom.header.frame_id = map_frame_; odom.child_frame_id = body_frame_;
    odom.pose.pose.position.x = t.x(); odom.pose.pose.position.y = t.y(); odom.pose.pose.position.z = t.z();
    odom.pose.pose.orientation.x = q.x(); odom.pose.pose.orientation.y = q.y(); odom.pose.pose.orientation.z = q.z(); odom.pose.pose.orientation.w = q.w();
    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = last_stamp_; tf.header.frame_id = map_frame_; tf.child_frame_id = body_frame_;
    tf.transform.translation.x = t.x(); tf.transform.translation.y = t.y(); tf.transform.translation.z = t.z();
    tf.transform.rotation.x = q.x(); tf.transform.rotation.y = q.y(); tf.transform.rotation.z = q.z(); tf.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf);

    geometry_msgs::msg::PoseStamped ps; ps.header = odom.header; ps.pose = odom.pose.pose;
    path_.header.stamp = last_stamp_; path_.poses.push_back(ps);
    if (path_.poses.size() > 5000) path_.poses.erase(path_.poses.begin());
    path_pub_->publish(path_);
  }

  void publish_map() {
    std::lock_guard<std::mutex> lk(grid_mutex_);
    if (!grid_initialized_) return;
    nav_msgs::msg::OccupancyGrid msg;
    msg.header.frame_id = map_frame_; msg.header.stamp = last_stamp_.nanoseconds() ? last_stamp_ : now();
    msg.info.resolution = static_cast<float>(resolution_);
    msg.info.width = grid_.width; msg.info.height = grid_.height;
    msg.info.origin.position.x = grid_.origin_x; msg.info.origin.position.y = grid_.origin_y; msg.info.origin.position.z = 0.0;
    msg.info.origin.orientation.w = 1.0;
    msg.data.resize(static_cast<size_t>(grid_.width) * grid_.height);
    for (size_t i = 0; i < msg.data.size(); ++i) {
      const int gx = static_cast<int>(i % static_cast<size_t>(grid_.width));
      const int gy = static_cast<int>(i / static_cast<size_t>(grid_.width));
      if (is_self_trail_cell_locked(gx, gy)) { msg.data[i] = 0; continue; }
      const float lo = grid_.log_odds[i];
      if (lo == 0.0f) { msg.data[i] = -1; continue; }
      const float prob = 1.0f - 1.0f / (1.0f + std::exp(lo));
      msg.data[i] = static_cast<int8_t>(std::clamp(static_cast<int>(std::round(100.0f * prob)), 0, 100));
    }
    map_pub_->publish(msg);
  }

  bool save_map_internal(std::string & message) {
    std::lock_guard<std::mutex> lk(grid_mutex_);
    if (!grid_initialized_) { message = "Map not yet initialized."; return false; }
    std::error_code ec; std::filesystem::create_directories(save_dir_, ec);
    const std::filesystem::path pgm = std::filesystem::path(save_dir_) / "grid_map_2d.pgm";
    const std::filesystem::path yaml = std::filesystem::path(save_dir_) / "grid_map_2d.yaml";
    std::ofstream of(pgm, std::ios::binary);
    if (!of) { message = "Cannot open " + pgm.string(); return false; }
    of << "P5\n# grid_mapper_2d\n" << grid_.width << " " << grid_.height << "\n255\n";
    const double occ_th = 0.65, free_th = 0.196;
    for (int y = grid_.height - 1; y >= 0; --y) {
      for (int x = 0; x < grid_.width; ++x) {
        const float lo = grid_.log_odds[static_cast<size_t>(y) * grid_.width + x];
        uint8_t pixel = 205;
        if (is_self_trail_cell_locked(x, y)) {
          pixel = 254;
          of.write(reinterpret_cast<const char *>(&pixel), 1);
          continue;
        }
        if (lo != 0.0f) {
          const float prob = 1.0f - 1.0f / (1.0f + std::exp(lo));
          if (prob >= occ_th) pixel = 0; else if (prob <= free_th) pixel = 254;
        }
        of.write(reinterpret_cast<const char *>(&pixel), 1);
      }
    }
    of.close();
    std::ofstream oy(yaml);
    if (!oy) { message = "Cannot open " + yaml.string(); return false; }
    oy << "image: " << pgm.filename().string() << "\nresolution: " << resolution_ << "\norigin: [" << grid_.origin_x << ", " << grid_.origin_y << ", 0.0]\nnegate: 0\noccupied_thresh: " << occ_th << "\nfree_thresh: " << free_th << "\n";
    message = "Saved to " + save_dir_; return true;
  }

  void on_save_map(std_srvs::srv::Trigger::Request::ConstSharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
    std::string msg; res->success = save_map_internal(msg); res->message = msg;
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridMapper2DNode>());
  rclcpp::shutdown();
  return 0;
}
