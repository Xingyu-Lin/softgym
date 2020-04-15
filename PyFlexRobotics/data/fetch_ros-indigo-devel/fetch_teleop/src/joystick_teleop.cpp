/*
 * Copyright (c) 2015, Fetch Robotics Inc.
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

// Author: Michael Ferguson, Hanjun Song

/*
 * This is still a work in progress
 * In the future, each teleop component would probably be a plugin
 */

#include <algorithm>
#include <boost/thread/mutex.hpp>

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

#include <control_msgs/FollowJointTrajectoryAction.h>
#include <control_msgs/GripperCommandAction.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Joy.h>
#include <topic_tools/MuxSelect.h>

double integrate(double desired, double present, double max_rate, double dt)
{
  if (desired > present)
    return std::min(desired, present + max_rate * dt);
  else
    return std::max(desired, present - max_rate * dt);
}


class TeleopComponent
{
public:
  TeleopComponent() : active_(false) {}
  virtual ~TeleopComponent() {}

  // This gets called whenever new joy message comes in
  // returns whether lower priority teleop components should be stopped
  virtual bool update(const sensor_msgs::Joy::ConstPtr& joy,
                      const sensor_msgs::JointState::ConstPtr& state) = 0;

  // This gets called at set frequency
  virtual void publish(const ros::Duration& dt) = 0;

  // Start the component. Must be idempotent.
  virtual bool start()
  {
    active_ = true;
    return active_;
  }

  // Stop the component. Must be idempotent.
  virtual bool stop()
  {
    active_ = false;
    return active_;
  }

protected:
  bool active_;
};


class BaseTeleop : public TeleopComponent
{
public:
  BaseTeleop(const std::string& name, ros::NodeHandle& nh)
  {
    ros::NodeHandle pnh(nh, name);

    // Button mapping
    pnh.param("button_deadman", deadman_, 10);
    pnh.param("axis_x", axis_x_, 3);
    pnh.param("axis_w", axis_w_, 0);

    // Base limits
    pnh.param("max_vel_x", max_vel_x_, 1.0);
    pnh.param("min_vel_x", min_vel_x_, -0.5);
    pnh.param("max_vel_w", max_vel_w_, 3.0);
    pnh.param("max_acc_x", max_acc_x_, 1.0);
    pnh.param("max_acc_w", max_acc_w_, 3.0);

    // Maximum windup of acceleration ramping
    pnh.param("max_windup_time", max_windup_time, 0.25);

    // Mux for overriding navigation, etc.
    pnh.param("use_mux", use_mux_, true);
    if (use_mux_)
    {
      mux_ = nh.serviceClient<topic_tools::MuxSelect>("/cmd_vel_mux/select");
    }

    cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/teleop/cmd_vel", 1);
    odom_sub_ = nh.subscribe("/odom", 1, &BaseTeleop::odomCallback, this);
  }

  virtual bool update(const sensor_msgs::Joy::ConstPtr& joy,
                      const sensor_msgs::JointState::ConstPtr& state)
  {
    bool deadman_pressed = joy->buttons[deadman_];

    if (!deadman_pressed)
    {
      stop();
      return false;
    }

    start();

    if (joy->axes[axis_x_] > 0.0)
      desired_.linear.x = joy->axes[axis_x_] * max_vel_x_;
    else
      desired_.linear.x = joy->axes[axis_x_] * -min_vel_x_;
    desired_.angular.z = joy->axes[axis_w_] * max_vel_w_;

    // We are active, don't process lower priority components
    return true;
  }

  virtual void publish(const ros::Duration& dt)
  {
    if (active_)
    {
      {
        boost::mutex::scoped_lock lock(odom_mutex_);
        // Make sure base is actually keeping up with commands
        // When accelerating (in either direction) do not continue to ramp our
        //   acceleration more than max_windup_time ahead of actually attained speeds.
        // This is especially important if robot gets stuck.
        if (last_.linear.x >= 0)
          last_.linear.x = std::min(last_.linear.x, odom_.twist.twist.linear.x + max_acc_x_ * max_windup_time);
        else
          last_.linear.x = std::max(last_.linear.x, odom_.twist.twist.linear.x - max_acc_x_ * max_windup_time);
        if (last_.angular.z >= 0)
          last_.angular.z = std::min(last_.angular.z, odom_.twist.twist.angular.z + max_acc_w_ * max_windup_time);
        else
          last_.angular.z = std::max(last_.angular.z, odom_.twist.twist.angular.z - max_acc_w_ * max_windup_time);
      }
      // Ramp commands based on acceleration limits
      last_.linear.x = integrate(desired_.linear.x, last_.linear.x, max_acc_x_, dt.toSec());
      last_.angular.z = integrate(desired_.angular.z, last_.angular.z, max_acc_w_, dt.toSec());
      cmd_vel_pub_.publish(last_);
    }
  }

  virtual bool start()
  {
    if (!active_ && use_mux_)
    {
      // Connect mux
      topic_tools::MuxSelect req;
      req.request.topic = cmd_vel_pub_.getTopic();
      if (mux_.call(req))
      {
        prev_mux_topic_ = req.response.prev_topic;
      }
      else
      {
        ROS_ERROR("Unable to switch mux");
      }
    }
    active_ = true;
    return active_;
  }

  virtual bool stop()
  {
    // Publish stop message
    last_ = desired_ = geometry_msgs::Twist();
    cmd_vel_pub_.publish(last_);
    // Disconnect mux
    if (active_ && use_mux_)
    {
      topic_tools::MuxSelect req;
      req.request.topic = prev_mux_topic_;
      if (!mux_.call(req))
      {
        ROS_ERROR("Unable to switch mux");
        return active_;
      }
    }
    active_ = false;
    return active_;
  }

private:
  void odomCallback(const nav_msgs::OdometryConstPtr& odom)
  {
    // Lock mutex on state message
    boost::mutex::scoped_lock lock(odom_mutex_);
    odom_ = *odom;
  }

  // Buttons from params
  int deadman_, axis_x_, axis_w_;

  // Limits from params
  double max_vel_x_, min_vel_x_, max_vel_w_;
  double max_acc_x_, max_acc_w_;

  // Support for multiplexor between teleop and application base commands
  bool use_mux_;
  std::string prev_mux_topic_;
  ros::ServiceClient mux_;

  // Twist output, odometry feedback
  ros::Publisher cmd_vel_pub_;
  ros::Subscriber odom_sub_;

  // Latest feedback, mutex around it
  boost::mutex odom_mutex_;
  nav_msgs::Odometry odom_;
  // Maximum timestep that our ramping can get ahead of actual velocities
  double max_windup_time;

  geometry_msgs::Twist desired_;
  geometry_msgs::Twist last_;
};


// This controls a single joint through a follow controller (for instance, torso)
class FollowTeleop : public TeleopComponent
{
  typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> client_t;

public:
  FollowTeleop(const std::string& name, ros::NodeHandle& nh)
  {
    ros::NodeHandle pnh(nh, name);

    // Button mapping
    pnh.param("button_deadman", deadman_, 10);
    pnh.param("button_increase", inc_button_, 12);
    pnh.param("button_decrease", dec_button_, 14);

    // Joint Limits
    pnh.param("min_position", min_position_, 0.0);
    pnh.param("max_position", max_position_, 0.4);
    pnh.param("max_velocity", max_velocity_, 0.075);
    pnh.param("max_accel", max_acceleration_, 0.25);

    // Should we inhibit lower priority components if running?
    pnh.param("inhibit", inhibit_, false);

    // Load topic/joint info
    pnh.param<std::string>("joint_name", joint_name_, "torso_lift_joint");
    std::string action_name;
    pnh.param<std::string>("action_name", action_name, "torso_controller/follow_joint_trajectory");

    client_.reset(new client_t(action_name, true));
    if (!client_->waitForServer(ros::Duration(2.0)))
    {
      ROS_ERROR("%s may not be connected.", action_name.c_str());
    }
  }

  // This gets called whenever new joy message comes in
  virtual bool update(const sensor_msgs::Joy::ConstPtr& joy,
                      const sensor_msgs::JointState::ConstPtr& state)
  {
    bool deadman_pressed = joy->buttons[deadman_];

    if (!deadman_pressed)
    {
      stop();
      // Update joint position
      for (size_t i = 0; i < state->name.size(); i++)
      {
        if (state->name[i] == joint_name_)
        {
          actual_position_ = state->position[i];
          break;
        }
      }
      return false;
    }

    if (joy->buttons[inc_button_])
    {
      desired_velocity_ = max_velocity_;
      start();
    }
    else if (joy->buttons[dec_button_])
    {
      desired_velocity_ = -max_velocity_;
      start();
    }
    else
    {
      desired_velocity_ = 0.0;
    }

    return inhibit_;
  }

  // This gets called at set frequency
  virtual void publish(const ros::Duration& dt)
  {
    if (active_)
    {
      // Fill in a message (future dated at fixed time step)
      double step = 0.25;
      double vel = integrate(desired_velocity_, last_velocity_, max_acceleration_, step);
      double travel = step * (vel + last_velocity_) / 2.0;
      double pos = std::max(min_position_, std::min(max_position_, actual_position_ + travel));
      // Send message
      control_msgs::FollowJointTrajectoryGoal goal;
      goal.trajectory.joint_names.push_back(joint_name_);
      trajectory_msgs::JointTrajectoryPoint p;
      p.positions.push_back(pos);
      p.velocities.push_back(vel);
      p.time_from_start = ros::Duration(step);
      goal.trajectory.points.push_back(p);
      goal.goal_time_tolerance = ros::Duration(0.0);
      client_->sendGoal(goal);
      // Update based on actual timestep
      vel = integrate(desired_velocity_, last_velocity_, max_acceleration_, dt.toSec());
      travel = dt.toSec() * (vel + last_velocity_) / 2.0;
      actual_position_ = std::max(min_position_, std::min(max_position_, actual_position_ + travel));
      last_velocity_ = vel;
    }
  }

  virtual bool stop()
  {
    active_ = false;
    last_velocity_ = 0.0;
    return active_;
  }

private:
  int deadman_, inc_button_, dec_button_;
  double min_position_, max_position_, max_velocity_, max_acceleration_;
  bool inhibit_;
  std::string joint_name_;
  double actual_position_;
  double desired_velocity_, last_velocity_;
  boost::shared_ptr<client_t> client_;
};

// Gripper Teleop
class GripperTeleop : public TeleopComponent
{
  typedef actionlib::SimpleActionClient<control_msgs::GripperCommandAction> client_t;

public:
  GripperTeleop(const std::string& name, ros::NodeHandle& nh) :
    req_close_(false),
    req_open_(false)
  {
    ros::NodeHandle pnh(nh, name);

    // Button mapping
    pnh.param("button_deadman", deadman_, 10);
    pnh.param("button_open", open_button_, 0);
    pnh.param("button_close", close_button_, 3);

    // Joint Limits
    pnh.param("closed_position", min_position_, 0.0);
    pnh.param("open_position", max_position_, 0.115);
    pnh.param("max_effort", max_effort_, 100.0);

    std::string action_name = "gripper_controller/gripper_action";
    client_.reset(new client_t(action_name, true));
    if (!client_->waitForServer(ros::Duration(2.0)))
    {
      ROS_ERROR("%s may not be connected.", action_name.c_str());
    }
  }

  // This gets called whenever new joy message comes in
  // returns whether lower priority teleop components should be stopped
  virtual bool update(const sensor_msgs::Joy::ConstPtr& joy,
                      const sensor_msgs::JointState::ConstPtr& state)
  {
    bool deadman_pressed = joy->buttons[deadman_];

    if (deadman_pressed)
    {
      if (joy->buttons[open_button_])
        req_open_ = true;
      else if (joy->buttons[close_button_])
        req_close_ = true;
    }

    return false;
  }

  // This gets called at set frequency
  virtual void publish(const ros::Duration& dt)
  {
    if (req_open_)
    {
      control_msgs::GripperCommandGoal goal;
      goal.command.position = max_position_;
      goal.command.max_effort = max_effort_;
      client_->sendGoal(goal);
      req_open_ = false;
    }
    else if (req_close_)
    {
      control_msgs::GripperCommandGoal goal;
      goal.command.position = min_position_;
      goal.command.max_effort = max_effort_;
      client_->sendGoal(goal);
      req_close_ = false;
    }
  }

private:
  int deadman_, open_button_, close_button_;
  double min_position_, max_position_, max_effort_;
  bool req_close_, req_open_;
  boost::shared_ptr<client_t> client_;
};


// Head Teleop
class HeadTeleop : public TeleopComponent
{
  typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> client_t;

public:
  HeadTeleop(const std::string& name, ros::NodeHandle& nh) :
    last_pan_(0.0),
    last_tilt_(0.0)
  {
    ros::NodeHandle pnh(nh, name);

    // Button mapping
    pnh.param("button_deadman", deadman_, 8);
    pnh.param("axis_pan", axis_pan_, 0);
    pnh.param("axis_tilt", axis_tilt_, 3);

    // Joint limits
    pnh.param("max_vel_pan", max_vel_pan_, 1.5);
    pnh.param("max_vel_tilt", max_vel_tilt_, 1.5);
    pnh.param("max_acc_pan", max_acc_pan_, 3.0);
    pnh.param("max_acc_tilt", max_acc_tilt_, 3.0);
    pnh.param("min_pos_pan", min_pos_pan_, -1.57);
    pnh.param("max_pos_pan", max_pos_pan_, 1.57);
    pnh.param("min_pos_tilt", min_pos_tilt_, -0.76);
    pnh.param("max_pos_tilt", max_pos_tilt_, 1.45);

    // TODO: load topic from params
    head_pan_joint_ = "head_pan_joint";
    head_tilt_joint_ = "head_tilt_joint";

    std::string action_name = "head_controller/follow_joint_trajectory";
    client_.reset(new client_t(action_name, true));
    if (!client_->waitForServer(ros::Duration(2.0)))
    {
      ROS_ERROR("%s may not be connected.", action_name.c_str());
    }
  }

  // This gets called whenever new joy message comes in
  virtual bool update(const sensor_msgs::Joy::ConstPtr& joy,
                      const sensor_msgs::JointState::ConstPtr& state)
  {
    bool deadman_pressed = joy->buttons[deadman_];

    if (!deadman_pressed)
    {
      stop();
      // Update joint positions
      for (size_t i = 0; i < state->name.size(); i++)
      {
        if (state->name[i] == head_pan_joint_)
          actual_pos_pan_ = state->position[i];
        if (state->name[i] == head_tilt_joint_)
          actual_pos_tilt_ = state->position[i];
      }
      return false;
    }

    desired_pan_ = joy->axes[axis_pan_] * max_vel_pan_;
    desired_tilt_ = joy->axes[axis_tilt_] * max_vel_tilt_;
    start();

    return true;
  }

  // This gets called at set frequency
  virtual void publish(const ros::Duration& dt)
  {
    if (active_)
    {
      // Fill in message (future dated with fixed time step)
      double step = 0.125;
      double pan_vel = integrate(desired_pan_, last_pan_, max_acc_pan_, step);
      double pan_travel = step * (pan_vel + last_pan_) / 2.0;
      double pan = std::max(min_pos_pan_, std::min(max_pos_pan_, actual_pos_pan_ + pan_travel));
      double tilt_vel = integrate(desired_tilt_, last_tilt_, max_acc_tilt_, step);
      double tilt_travel = step * (tilt_vel + last_tilt_) / 2.0;
      double tilt = std::max(min_pos_tilt_, std::min(max_pos_tilt_, actual_pos_tilt_ + tilt_travel));
      // Publish message
      control_msgs::FollowJointTrajectoryGoal goal;
      goal.trajectory.joint_names.push_back(head_pan_joint_);
      goal.trajectory.joint_names.push_back(head_tilt_joint_);
      trajectory_msgs::JointTrajectoryPoint p;
      p.positions.push_back(pan);
      p.positions.push_back(tilt);
      p.velocities.push_back(pan_vel);
      p.velocities.push_back(tilt_vel);
      p.time_from_start = ros::Duration(step);
      goal.trajectory.points.push_back(p);
      goal.goal_time_tolerance = ros::Duration(0.0);
      client_->sendGoal(goal);
      // Update based on actual timestep
      pan_vel = integrate(desired_pan_, last_pan_, max_acc_pan_, dt.toSec());
      pan_travel = dt.toSec() * (pan_vel + last_pan_) / 2.0;
      actual_pos_pan_ = std::max(min_pos_pan_, std::min(max_pos_pan_, actual_pos_pan_ + pan_travel));
      last_pan_ = pan_vel;
      tilt_vel = integrate(desired_tilt_, last_tilt_, max_acc_tilt_, dt.toSec());
      tilt_travel = dt.toSec() * (tilt_vel + last_tilt_) / 2.0;
      actual_pos_tilt_ = std::max(min_pos_tilt_, std::min(max_pos_tilt_, actual_pos_tilt_ + tilt_travel));
      last_tilt_ = tilt_vel;
    }
  }

  virtual bool stop()
  {
    active_ = false;
    last_pan_ = last_tilt_ = 0.0;  // reset velocities
    return active_;
  }

private:
  int deadman_, axis_pan_, axis_tilt_;
  double max_vel_pan_, max_vel_tilt_;
  double max_acc_pan_, max_acc_tilt_;
  double min_pos_pan_, max_pos_pan_, min_pos_tilt_, max_pos_tilt_;
  std::string head_pan_joint_, head_tilt_joint_;
  double actual_pos_pan_, actual_pos_tilt_;  // actual positions
  double desired_pan_, desired_tilt_;  // desired velocities
  double last_pan_, last_tilt_;
  boost::shared_ptr<client_t> client_;
};

class ArmTeleop : public TeleopComponent
{
public:
  ArmTeleop(const std::string& name, ros::NodeHandle& nh)
  {
    ros::NodeHandle pnh(nh, name);

    pnh.param("axis_x", axis_x_, 3);
    pnh.param("axis_y", axis_y_, 2);
    pnh.param("axis_z", axis_z_, 1);
    pnh.param("axis_roll", axis_roll_, 2);
    pnh.param("axis_pitch", axis_pitch_, 3);
    pnh.param("axis_yaw", axis_yaw_, 0);

    pnh.param("button_arm_linear", button_linear_, 9);
    pnh.param("button_arm_angular", button_angular_, 11);

    // Twist limits
    pnh.param("max_vel_x", max_vel_x_, 1.0);
    pnh.param("max_vel_y", max_vel_y_, 1.0);
    pnh.param("max_vel_z", max_vel_z_, 1.0);
    pnh.param("max_acc_x", max_acc_x_, 10.0);
    pnh.param("max_acc_y", max_acc_y_, 10.0);
    pnh.param("max_acc_z", max_acc_z_, 10.0);

    pnh.param("max_vel_roll", max_vel_roll_, 2.0);
    pnh.param("max_vel_pitch", max_vel_pitch_, 2.0);
    pnh.param("max_vel_yaw", max_vel_yaw_, 2.0);
    pnh.param("max_acc_roll", max_acc_roll_, 10.0);
    pnh.param("max_acc_pitch", max_acc_pitch_, 10.0);
    pnh.param("max_acc_yaw", max_acc_yaw_, 10.0);

    cmd_pub_ = nh.advertise<geometry_msgs::TwistStamped>("/arm_controller/cartesian_twist/command", 10);
  }

  virtual bool update(const sensor_msgs::Joy::ConstPtr& joy,
                      const sensor_msgs::JointState::ConstPtr& state)
  {
    bool button_linear_pressed = joy->buttons[button_linear_];
    bool button_angular_pressed = joy->buttons[button_angular_];

    if (!(button_linear_pressed || button_angular_pressed) &&
        (ros::Time::now() - last_update_ > ros::Duration(0.5)))
    {
      stop();
      return false;
    }

    start();

    if (button_linear_pressed)
    {
      desired_.twist.linear.x = joy->axes[axis_x_] * max_vel_x_;
      desired_.twist.linear.y = joy->axes[axis_y_] * max_vel_y_;
      desired_.twist.linear.z = joy->axes[axis_z_] * max_vel_z_;
      desired_.twist.angular.x = 0.0;
      desired_.twist.angular.y = 0.0;
      desired_.twist.angular.z = 0.0;
      last_update_ = ros::Time::now();
    }
    else if (button_angular_pressed)
    {
      desired_.twist.linear.x = 0.0;
      desired_.twist.linear.y = 0.0;
      desired_.twist.linear.z = 0.0;
      desired_.twist.angular.x = joy->axes[axis_roll_] * max_vel_roll_;
      desired_.twist.angular.y = joy->axes[axis_pitch_] * max_vel_pitch_;
      desired_.twist.angular.z = joy->axes[axis_yaw_] * max_vel_yaw_;
      last_update_ = ros::Time::now();
    }
    else
    {
      desired_.twist.linear.x = 0.0;
      desired_.twist.linear.y = 0.0;
      desired_.twist.linear.z = 0.0;
      desired_.twist.angular.x = 0.0;
      desired_.twist.angular.y = 0.0;
      desired_.twist.angular.z = 0.0;
    }

    return true;
  }

  virtual void publish(const ros::Duration& dt)
  {
    if (active_)
    {
      // Ramp commands based on acceleration limits
      last_.twist.linear.x = integrate(desired_.twist.linear.x, last_.twist.linear.x, max_acc_x_, dt.toSec());
      last_.twist.linear.y = integrate(desired_.twist.linear.y, last_.twist.linear.y, max_acc_y_, dt.toSec());
      last_.twist.linear.z = integrate(desired_.twist.linear.z, last_.twist.linear.z, max_acc_z_, dt.toSec());

      last_.twist.angular.x = integrate(desired_.twist.angular.x, last_.twist.angular.x, max_acc_roll_, dt.toSec());
      last_.twist.angular.y = integrate(desired_.twist.angular.y, last_.twist.angular.y, max_acc_pitch_, dt.toSec());
      last_.twist.angular.z = integrate(desired_.twist.angular.z, last_.twist.angular.z, max_acc_yaw_, dt.toSec());

      last_.header.frame_id = "base_link";

      cmd_pub_.publish(last_);
    }
  }

  virtual bool start()
  {
    active_ = true;
    return active_;
  }


  virtual bool stop()
  {
    // Publish stop message
    if (active_)
    {
      last_ = desired_ = geometry_msgs::TwistStamped();
      cmd_pub_.publish(last_);
    }

    active_ = false;
    return active_;
  }

private:

  // Buttons from params
  int axis_x_, axis_y_, axis_z_, axis_roll_, axis_pitch_, axis_yaw_;
  int button_linear_, button_angular_;

  // Limits from params
  double max_vel_x_, max_vel_y_, max_vel_z_;
  double max_vel_roll_, max_vel_pitch_, max_vel_yaw_;
  double max_acc_x_, max_acc_y_, max_acc_z_;
  double max_acc_roll_, max_acc_pitch_, max_acc_yaw_;

  // Twist output
  ros::Publisher cmd_pub_;

  geometry_msgs::TwistStamped desired_;
  geometry_msgs::TwistStamped last_;
  ros::Time last_update_;
};

// This pulls all the components together
class Teleop
{
  typedef boost::shared_ptr<TeleopComponent> TeleopComponentPtr;

public:
  void init(ros::NodeHandle& nh)
  {
    bool is_fetch;
    nh.param("is_fetch", is_fetch, true);

    // TODO: load these from YAML

    TeleopComponentPtr c;
    if (is_fetch)
    {
      // Torso does not override
      c.reset(new FollowTeleop("torso", nh));
      components_.push_back(c);

      // Gripper does not override
      c.reset(new GripperTeleop("gripper", nh));
      components_.push_back(c);

      // Head overrides base
      c.reset(new HeadTeleop("head", nh));
      components_.push_back(c);

      c.reset(new ArmTeleop("arm", nh));
      components_.push_back(c);
    }

    // BaseTeleop goes last
    c.reset(new BaseTeleop("base", nh));
    components_.push_back(c);

    state_msg_.reset(new sensor_msgs::JointState());
    joy_sub_ = nh.subscribe("/joy", 1, &Teleop::joyCallback, this);
    state_sub_ = nh.subscribe("/joint_states", 10, &Teleop::stateCallback, this);
  }

  void publish(const ros::Duration& dt)
  {
    if (ros::Time::now() - last_update_ > ros::Duration(0.25))
    {
      // Timed out
      for (size_t c = 0; c < components_.size(); c++)
      {
        components_[c]->stop();
      }
    }
    else
    {
      for (size_t c = 0; c < components_.size(); c++)
      {
        components_[c]->publish(dt);
      }
    }
  }

private:
  void joyCallback(const sensor_msgs::Joy::ConstPtr& msg)
  {
    // Lock mutex on state message
    boost::mutex::scoped_lock lock(state_mutex_);

    bool ok = true;
    for (size_t c = 0; c < components_.size(); c++)
    {
      if (ok)
      {
        ok &= !components_[c]->update(msg, state_msg_);
      }
      else
      {
        // supressed by a higher priority component
        components_[c]->stop();
      }
    }
    last_update_ = ros::Time::now();
  }

  void stateCallback(const sensor_msgs::JointStateConstPtr& msg)
  {
    // Lock mutex on state message
    boost::mutex::scoped_lock lock(state_mutex_);

    // Update each joint based on message
    for (size_t msg_j = 0; msg_j < msg->name.size(); msg_j++)
    {
      size_t state_j;
      for (state_j = 0; state_j < state_msg_->name.size(); state_j++)
      {
        if (state_msg_->name[state_j] == msg->name[msg_j])
        {
          state_msg_->position[state_j] = msg->position[msg_j];
          state_msg_->velocity[state_j] = msg->velocity[msg_j];
          break;
        }
      }
      if (state_j == state_msg_->name.size())
      {
        // New joint
        state_msg_->name.push_back(msg->name[msg_j]);
        state_msg_->position.push_back(msg->position[msg_j]);
        state_msg_->velocity.push_back(msg->velocity[msg_j]);
      }
    }
  }

  std::vector<TeleopComponentPtr> components_;
  ros::Time last_update_;
  boost::mutex state_mutex_;
  sensor_msgs::JointStatePtr state_msg_;
  ros::Subscriber joy_sub_, state_sub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "teleop");
  ros::NodeHandle n("~");

  Teleop teleop;
  teleop.init(n);

  ros::Rate r(30.0);
  while (ros::ok())
  {
    ros::spinOnce();
    teleop.publish(ros::Duration(1/30.0));
    r.sleep();
  }

  return 0;
}
