/*
 * Copyright (c) 2015-2016, Fetch Robotics Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Fetch Robotics Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL FETCH ROBOTICS INC. BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Author: Anuj Pasricha, Michael Ferguson

#include <pluginlib/class_list_macros.h>
#include <fetch_depth_layer/depth_layer.h>
#include <limits>

PLUGINLIB_EXPORT_CLASS(costmap_2d::FetchDepthLayer, costmap_2d::Layer)

namespace costmap_2d
{

FetchDepthLayer::FetchDepthLayer()
{
}

void FetchDepthLayer::onInitialize()
{
  VoxelLayer::onInitialize();

  double observation_keep_time;
  double expected_update_rate;
  double transform_tolerance;
  double obstacle_range;
  double raytrace_range;
  double min_obstacle_height;
  double max_obstacle_height;
  double min_clearing_height;
  double max_clearing_height;
  std::string topic = "";
  std::string sensor_frame = "";

  ros::NodeHandle private_nh("~/" + name_);

  private_nh.param("publish_observations", publish_observations_, false);
  private_nh.param("observations_separation_threshold", observations_threshold_, 0.06);

  // Optionally detect the ground plane
  private_nh.param("find_ground_plane", find_ground_plane_, true);
  private_nh.param("ground_orientation_threshold", ground_threshold_, 0.9);

  // Should NANs be used as clearing observations?
  private_nh.param("clear_nans", clear_nans_, false);

  // Observation range values for both marking and clearing
  private_nh.param("min_obstacle_height", min_obstacle_height, 0.0);
  private_nh.param("max_obstacle_height", max_obstacle_height, 2.0);
  private_nh.param("min_clearing_height", min_clearing_height, -std::numeric_limits<double>::infinity());
  private_nh.param("max_clearing_height", max_clearing_height, std::numeric_limits<double>::infinity());

  // Skipping of potentially noisy rays near the edge of the image
  private_nh.param("skip_rays_bottom", skip_rays_bottom_, 20);
  private_nh.param("skip_rays_top",    skip_rays_top_,    20);
  private_nh.param("skip_rays_left",   skip_rays_left_,   20);
  private_nh.param("skip_rays_right",  skip_rays_right_,  20);

  // Should skipped edge rays be used for clearing?
  private_nh.param("clear_with_skipped_rays", clear_with_skipped_rays_, false);

  // How long should observations persist?
  private_nh.param("observation_persistence", observation_keep_time, 0.0);

  // How often should we expect to get sensor updates?
  private_nh.param("expected_update_rate", expected_update_rate, 0.0);

  // How long to wait for transforms to be available?
  private_nh.param("transform_tolerance", transform_tolerance, 0.5);

  std::string raytrace_range_param_name, obstacle_range_param_name;

  // get the obstacle range for the sensor
  obstacle_range = 2.5;
  if (private_nh.searchParam("obstacle_range", obstacle_range_param_name))
  {
    private_nh.getParam(obstacle_range_param_name, obstacle_range);
  }

  // get the raytrace range for the sensor
  raytrace_range = 3.0;
  if (private_nh.searchParam("raytrace_range", raytrace_range_param_name))
  {
    private_nh.getParam(raytrace_range_param_name, raytrace_range);
  }

  marking_buf_ = boost::shared_ptr<costmap_2d::ObservationBuffer> (
  	new costmap_2d::ObservationBuffer(topic, observation_keep_time,
  	  expected_update_rate, min_obstacle_height, max_obstacle_height,
  	  obstacle_range, raytrace_range, *tf_, global_frame_,
  	  sensor_frame, transform_tolerance));
  marking_buffers_.push_back(marking_buf_);
  observation_buffers_.push_back(marking_buf_);

  min_obstacle_height = 0.0;

  clearing_buf_ =  boost::shared_ptr<costmap_2d::ObservationBuffer> (
  	new costmap_2d::ObservationBuffer(topic, observation_keep_time,
  	  expected_update_rate, min_clearing_height, max_clearing_height,
  	  obstacle_range, raytrace_range, *tf_, global_frame_,
  	  sensor_frame, transform_tolerance));
  clearing_buffers_.push_back(clearing_buf_);
  observation_buffers_.push_back(clearing_buf_);

  if (publish_observations_)
  {
    clearing_pub_ = private_nh.advertise<sensor_msgs::PointCloud>("clearing_obs", 1);
    marking_pub_ = private_nh.advertise<sensor_msgs::PointCloud>("marking_obs", 1);
  }

  // subscribe to camera/info topics
  std::string camera_depth_topic, camera_info_topic;
  private_nh.param("depth_topic", camera_depth_topic,
                   std::string("/head_camera/depth_downsample/image_raw"));
  private_nh.param("info_topic", camera_info_topic,
                   std::string("/head_camera/depth_downsample/camera_info"));
  camera_info_sub_ = private_nh.subscribe<sensor_msgs::CameraInfo>(
    camera_info_topic, 10, &FetchDepthLayer::cameraInfoCallback, this);

  depth_image_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(private_nh, camera_depth_topic, 10));
  depth_image_filter_ = boost::shared_ptr< tf::MessageFilter<sensor_msgs::Image> >(
    new tf::MessageFilter<sensor_msgs::Image>(*depth_image_sub_, *tf_, global_frame_, 10));
  depth_image_filter_->registerCallback(boost::bind(&FetchDepthLayer::depthImageCallback, this, _1));
  observation_subscribers_.push_back(depth_image_sub_);
  observation_notifiers_.push_back(depth_image_filter_);
  observation_notifiers_.back()->setTolerance(ros::Duration(0.05));
}

FetchDepthLayer::~FetchDepthLayer()
{
}

void FetchDepthLayer::cameraInfoCallback(
  const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  // Lock mutex before updating K
  boost::unique_lock<boost::mutex> lock(mutex_K_);

  float focal_pixels_ = msg->P[0];
  float center_x_ = msg->P[2];
  float center_y_ = msg->P[6];

  if (msg->binning_x == msg->binning_y)
  {
    if (msg->binning_x > 0)
    {
      K_ = (cv::Mat_<double>(3, 3) <<
        focal_pixels_/msg->binning_x, 0.0, center_x_/msg->binning_x,
        0.0, focal_pixels_/msg->binning_x, center_y_/msg->binning_x,
        0.0, 0.0, 1.0);
    }
    else
    {
      K_ = (cv::Mat_<double>(3, 3) <<
        focal_pixels_, 0.0, center_x_,
        0.0, focal_pixels_, center_y_,
        0.0, 0.0, 1.0);
    }
  }
  else
  {
    ROS_ERROR("binning_x is not equal to binning_y");
  }
}

void FetchDepthLayer::depthImageCallback(
  const sensor_msgs::Image::ConstPtr& msg)
{
  // Lock mutex before using K
  boost::unique_lock<boost::mutex> lock(mutex_K_);

  if (K_.empty())
  {
    ROS_DEBUG_NAMED("depth_layer", "Camera info not yet received.");
    return;
  }

  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Clear with NANs?
  if (clear_nans_)
  {
    for (int i = 0; i < cv_ptr->image.rows * cv_ptr->image.cols; i++)
    {
      if (std::isnan(cv_ptr->image.at<float>(i)))
        cv_ptr->image.at<float>(i) = 25.0;
    }
  }

  // Convert to 3d
  cv::Mat points3d;
  depthTo3d(cv_ptr->image, K_, points3d);

  // Determine ground plane, either through camera or TF
  cv::Vec4f ground_plane;
  if (find_ground_plane_)
  {
    // Get normals
    if (normals_estimator_.empty())
    {
      normals_estimator_ = new RgbdNormals(cv_ptr->image.rows,
                                           cv_ptr->image.cols,
                                           cv_ptr->image.depth(),
                                           K_);
    }
    cv::Mat normals;
    (*normals_estimator_)(points3d, normals);

    // Find plane(s)
    if (plane_estimator_.empty())
    {
#if CV_MAJOR_VERSION == 3
      plane_estimator_.reset(new RgbdPlane());
      // Model parameters are based on notes in opencv_candidate
      plane_estimator_->setSensorErrorA(0.0075);
      plane_estimator_->setSensorErrorB(0.0);
      plane_estimator_->setSensorErrorC(0.0);
      // Image/cloud height/width must be multiple of block size
      plane_estimator_->setBlockSize(40);
      // Distance a point can be from plane and still be part of it
      plane_estimator_->setThreshold(observations_threshold_);
      // Minimum cluster size to be a plane
      plane_estimator_->setMinSize(1000);
#else
      plane_estimator_ = cv::Algorithm::create<RgbdPlane>("RGBD.RgbdPlane");
      // Model parameters are based on notes in opencv_candidate
      plane_estimator_->set("sensor_error_a", 0.0075);
      plane_estimator_->set("sensor_error_b", 0.0);
      plane_estimator_->set("sensor_error_c", 0.0);
      // Image/cloud height/width must be multiple of block size
      plane_estimator_->set("block_size", 40);
      // Distance a point can be from plane and still be part of it
      plane_estimator_->set("threshold", observations_threshold_);
      // Minimum cluster size to be a plane
      plane_estimator_->set("min_size", 1000);
#endif
    }
    cv::Mat planes_mask;
    std::vector<cv::Vec4f> plane_coefficients;
    (*plane_estimator_)(points3d, normals, planes_mask, plane_coefficients);

    for (size_t i = 0; i < plane_coefficients.size(); i++)
    {
      // check plane orientation
      if ((fabs(0.0 - plane_coefficients[i][0]) <= ground_threshold_) &&
          (fabs(1.0 + plane_coefficients[i][1]) <= ground_threshold_) &&
          (fabs(0.0 - plane_coefficients[i][2]) <= ground_threshold_))
      {
        ground_plane = plane_coefficients[i];
        break;
      }
    }
  }
  else
  {
    // find ground plane in camera coordinates using tf
    // transform normal axis
    tf::Stamped<tf::Vector3> vector(tf::Vector3(0, 0, 1), ros::Time(0), "base_link");
    tf_->transformVector(msg->header.frame_id, vector, vector);
    ground_plane[0] = vector.getX();
    ground_plane[1] = vector.getY();
    ground_plane[2] = vector.getZ();

    // find offset
    tf::StampedTransform transform;
    tf_->lookupTransform("base_link", msg->header.frame_id, ros::Time(0), transform);
    ground_plane[3] = transform.getOrigin().getZ();
  }

  // check that ground plane actually exists, so it doesn't count as marking observations
  if (ground_plane[0] == 0.0 && ground_plane[1] == 0.0 &&
      ground_plane[2] == 0.0 && ground_plane[3] == 0.0)
  {
    ROS_DEBUG_NAMED("depth_layer", "Invalid ground plane.");
    return;
  }

  cv::Mat channels[3];
  cv::split(points3d, channels);

  sensor_msgs::PointCloud clearing_points;
  clearing_points.header.stamp = msg->header.stamp;
  clearing_points.header.frame_id = msg->header.frame_id;

  sensor_msgs::PointCloud marking_points;
  marking_points.header.stamp = msg->header.stamp;
  marking_points.header.frame_id = msg->header.frame_id;

  // Put points in clearing/marking clouds
  for (size_t i = 0; i < points3d.rows; i++)
  {
    for (size_t j = 0; j < points3d.cols; j++)
    {
      // Get next point
      geometry_msgs::Point32 current_point;
      current_point.x = channels[0].at<float>(i, j);
      current_point.y = channels[1].at<float>(i, j);
      current_point.z = channels[2].at<float>(i, j);
      // Check point validity
      if (current_point.x != 0.0 &&
          current_point.y != 0.0 &&
          current_point.z != 0.0 &&
          !std::isnan(current_point.x) &&
          !std::isnan(current_point.y) &&
          !std::isnan(current_point.z))
      {
        if (clear_with_skipped_rays_)
        {
          // If edge rays are to be used for clearing, go ahead and add them now.
          clearing_points.points.push_back(current_point);
        }

        // Do not consider boundary points for obstacles marking since they are very noisy.
        if (i < skip_rays_top_ ||
            i >= points3d.rows - skip_rays_bottom_ ||
            j < skip_rays_left_ ||
            j >= points3d.cols - skip_rays_right_)
        {
          continue;
        }

        if (!clear_with_skipped_rays_)
        {
          // If edge rays are not to be used for clearing, only add them after the edge check.
          clearing_points.points.push_back(current_point);
        }

        // Check if point is part of the ground plane
        if (fabs(ground_plane[0] * current_point.x +
                 ground_plane[1] * current_point.y +
                 ground_plane[2] * current_point.z +
                 ground_plane[3]) <= observations_threshold_)
        {
          continue;  // Do not mark points near the floor.
        }

        // Check for outliers, mark non-outliers as obstacles.
        int num_valid = 0;
        for (int x = -1; x < 2; x++)
        {
          for (int y = -1; y < 2; y++)
          {
            if (x == 0 && y == 0)
            {
              continue;
            }
            float px = channels[0].at<float>(i+x, j+y);
            float py = channels[1].at<float>(i+x, j+y);
            float pz = channels[2].at<float>(i+x, j+y);
            if (px != 0.0 && py != 0.0 && pz != 0.0 &&
                !std::isnan(px) && !std::isnan(py) && !std::isnan(pz))
            {
              if ( fabs(px - current_point.x) < 0.1 &&
                    fabs(py - current_point.y) < 0.1 &&
                    fabs(pz - current_point.z) < 0.1)
              {
                num_valid++;
              }
            }
          }  // for y
        }  // for x

        if (num_valid >= 7)
        {
          marking_points.points.push_back(current_point);
        }
      }  // for j (y)
    }  // for i (x)
  }

  // Publish and buffer our clearing point cloud
  if (publish_observations_)
  {
    clearing_pub_.publish(clearing_points);
  }

  sensor_msgs::PointCloud2 clearing_cloud2;
  if (!sensor_msgs::convertPointCloudToPointCloud2(clearing_points, clearing_cloud2))
  {
    ROS_ERROR("Failed to convert a PointCloud to a PointCloud2, dropping message");
    return;
  }

  // buffer the ground plane observation
  clearing_buf_->lock();
  clearing_buf_->bufferCloud(clearing_cloud2);
  clearing_buf_->unlock();

  // Publish and buffer our marking point cloud
  if (publish_observations_)
  {
    marking_pub_.publish(marking_points);
  }

  sensor_msgs::PointCloud2 marking_cloud2;
  if (!sensor_msgs::convertPointCloudToPointCloud2(marking_points, marking_cloud2))
  {
    ROS_ERROR("Failed to convert a PointCloud to a PointCloud2, dropping message");
    return;
  }

  marking_buf_->lock();
  marking_buf_->bufferCloud(marking_cloud2);
  marking_buf_->unlock();
}

}  // namespace costmap_2d
