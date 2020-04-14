// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>
#include <chrono>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/gripper.h>
#include "examples_common.h"

#if __has_feature(cxx_deleted_functions)
	#define ZMQ_DELETED_FUNCTION = delete
#else
	#define ZMQ_DELETED_FUNCTION
#endif
#include <zmq.hpp>

namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}  // anonymous namespace


int main(int argc, char** argv) {

  // Check whether the required arguments were passed.
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  // Gripper settings
  franka::Gripper gripper(argv[1]);
  const double grasping_width = 0.022;
  const double grasping_force = 100;
  const double grasping_speed = 0.1;

  const double joint_limits_low[7] = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
  const double joint_limits_high[7] = {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
  const double joint_limit_margin = 1.005;

  franka::GripperState gripper_state = gripper.readOnce();
  const double gripper_max_width = gripper_state.max_width;

  // Set state publishing rate.
  const double publish_rate = 100.0;

  // Initialize data fields for the publish thread.
  struct {
    std::mutex mutex;
    bool has_data;
    std::array<double, 7> tau_d_last;
    franka::RobotState robot_state;
    std::array<double, 7> gravity;
    std::array<double, 7> des_vel;
    int gripper_value;
  } publish_data{};

  std::atomic_bool running{true};

  zmq::context_t context(1);

  // Command subscriber.
  zmq::socket_t zmq_sub(context, ZMQ_SUB);
  zmq_sub.setsockopt(ZMQ_SUBSCRIBE, "fcom ", 5);
  int conflate = 1;
  zmq_sub.setsockopt(ZMQ_CONFLATE, &conflate, sizeof(conflate)); 
  zmq_sub.connect("tcp://10.0.0.1:6001");

  // State publisher.
  zmq::socket_t zmq_pub (context, ZMQ_PUB);
  zmq_pub.bind("tcp://*:6002");


  // Start publish thread.
  std::thread publish_thread([publish_rate, &publish_data, &running, &zmq_pub]() {
    while (running) {
      // Sleep to achieve the desired print rate.
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>((1.0 / publish_rate * 1000.0))));

      // Try to lock data to avoid read write collisions.
      if (publish_data.mutex.try_lock()) {
        if (publish_data.has_data) {
	   zmq::message_t message(150);
           snprintf ((char *) message.data(), 150,
             "fstate %.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf", 
	        publish_data.robot_state.q[0], 
		publish_data.robot_state.q[1], publish_data.robot_state.q[2],
		publish_data.robot_state.q[3], publish_data.robot_state.q[4],
		publish_data.robot_state.q[5], publish_data.robot_state.q[6],
		publish_data.robot_state.q_d[0], 
		publish_data.robot_state.q_d[1], publish_data.robot_state.q_d[2],
		publish_data.robot_state.q_d[3], publish_data.robot_state.q_d[4],
		publish_data.robot_state.q_d[5], publish_data.robot_state.q_d[6]);
	   //printf("SENDING %s\n", (char*) message.data());
	   printf("DES VEL %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
 	   	publish_data.des_vel[0],publish_data.des_vel[1],publish_data.des_vel[2],
	   	publish_data.des_vel[3],publish_data.des_vel[4],publish_data.des_vel[5],
	   	publish_data.des_vel[6]);
           zmq_pub.send(message);

          /*std::array<double, 7> tau_error{};
          double error_rms(0.0);
          std::array<double, 7> tau_d_actual{};
          for (size_t i = 0; i < 7; ++i) {
            tau_d_actual[i] = publish_data.tau_d_last[i] + publish_data.gravity[i];
            tau_error[i] = tau_d_actual[i] - publish_data.robot_state.tau_J[i];
            error_rms += std::pow(tau_error[i], 2.0) / tau_error.size();
          }
          error_rms = std::sqrt(error_rms);

          // Print data to console
          std::cout << "tau_error [Nm]: " << tau_error << std::endl
                    << "tau_commanded [Nm]: " << tau_d_actual << std::endl
                    << "tau_measured [Nm]: " << publish_data.robot_state.tau_J << std::endl
                    << "root mean square of tau_error [Nm]: " << error_rms << std::endl
                    << "-----------------------" << std::endl;
	*/
          
          publish_data.has_data = false;
        }
        publish_data.mutex.unlock();
      }
    }

  });

  int current_gripper_state = 0;
  std::thread gripper_thread([publish_rate, grasping_width, grasping_speed, gripper_max_width,
				grasping_force, &gripper, &publish_data, &running, &current_gripper_state]() {
    while (running) {
      // Sleep to achieve the desired print rate.
      std::this_thread::sleep_for(
         std::chrono::milliseconds(static_cast<int>((1.0 / publish_rate * 1000.0))));
      int gripper_value = 0;

      if (publish_data.mutex.try_lock()) {
	gripper_value = publish_data.gripper_value;
	publish_data.gripper_value = 0;
      	publish_data.mutex.unlock();
      }
      if (gripper_value < 0 && current_gripper_state >= 0) {
        std::cout << "Closing the gripper..." << std::endl;
        gripper.grasp(grasping_width, grasping_speed, grasping_force);
        current_gripper_state = -1;
      } else if (gripper_value > 0 && current_gripper_state <= 0) {
        std::cout << "Openinig the gripper..." << std::endl;
	gripper.move(gripper_max_width, grasping_speed);
	current_gripper_state = 1;
      }
    }
  });

  try {
    // Connect to robot.
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

    // First move the robot to a suitable joint configuration
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator(0.5, q_goal);
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    //robot.setCollisionBehavior(
    //    {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
    //    {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
    //    {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
    //    {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    
    robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                              {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    // Load the kinematics and dynamics model.
    franka::Model model = robot.loadModel();

    double time = 0.0;
    double qvel_d[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int gripper_value = 0;

    auto joint_vel_callback =
        [=, &time, &qvel_d, &zmq_sub, &publish_data, &gripper_value, &joint_limits_high, &joint_limits_low, &joint_limit_margin]
	(const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities {
            time += period.toSec();

            zmq::message_t update;

            if (zmq_sub.recv(&update, ZMQ_DONTWAIT)){
                std::string upd_str =
                    std::string(static_cast<char*>(update.data()), update.size());
                sscanf(upd_str.c_str(), "fcom %lf,%lf,%lf,%lf,%lf,%lf,%lf,%d", &qvel_d[0], &qvel_d[1],
                        &qvel_d[2], &qvel_d[3], &qvel_d[4], &qvel_d[5], &qvel_d[6], &gripper_value);
            }

	    // Cap joint velocities if too close to joint limits.
            for (int i = 0; i < 7; i++){
		if (state.q_d[i] > joint_limit_margin * joint_limits_high[i] && qvel_d[i] > 0.0){
		   qvel_d[i] = 0.0;
		}
		if (state.q_d[i] < joint_limit_margin * joint_limits_low[i] && qvel_d[i] < 0.0){
		   qvel_d[i] = 0.0;
		}
	    }	

            franka::JointVelocities velocities = {{
                qvel_d[0], qvel_d[1], qvel_d[2],
                qvel_d[3], qvel_d[4], qvel_d[5], qvel_d[6] }};

	    if (publish_data.mutex.try_lock()) {
  		for (int i = 0; i < 7; i++){
 		   publish_data.des_vel[i] = qvel_d[i];
		}
 		publish_data.gripper_value = gripper_value;
		publish_data.mutex.unlock();
      	    }
    
            return velocities;
    };

    // Set gains for the joint impedance control.
    // Stiffness
    //const std::array<double, 7> k_gains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    //const std::array<double, 7> k_gains = {{100.0, 100.0, 100.0, 100.0, 50.0, 25.0, 10.0}};
    const std::array<double, 7> k_gains = {{50.0, 50.0, 50.0, 50.0, 25.0, 12.5, 5.0}};

    // Damping
    //const std::array<double, 7> d_gains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
    //const std::array<double, 7> d_gains = {{25.0, 25.0, 25.0, 25.0, 8.0, 7.0, 4.0}};
    const std::array<double, 7> d_gains = {{12.5, 12.5, 12.5, 12.5, 4.0, 3.5, 2.0}};

    // Define callback for the joint torque control loop.
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&publish_data, &model, k_gains, d_gains](
                const franka::RobotState& state, franka::Duration /*period*/) -> franka::Torques {
      // Read current coriolis terms from model.
      std::array<double, 7> coriolis = model.coriolis(state);

      // Compute torque command from joint impedance control law.
      // Note: The answer to our Cartesian pose inverse kinematics is always in state.q_d with one
      // time step delay.
      std::array<double, 7> tau_d_calculated;
      for (size_t i = 0; i < 7; i++) {
        tau_d_calculated[i] =
            k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i];
      }

      // The following line is only necessary for printing the rate limited torque. As we activated
      // rate limiting for the control loop (activated by default), the torque would anyway be
      // adjusted!
      std::array<double, 7> tau_d_rate_limited =
          franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

      // Update data to print.
      if (publish_data.mutex.try_lock()) {
        publish_data.has_data = true;
        publish_data.robot_state = state;
        publish_data.tau_d_last = tau_d_rate_limited;
        publish_data.gravity = model.gravity(state);
        publish_data.mutex.unlock();
      }

      // Send torque command.
      return tau_d_rate_limited;
    };

    // Start real-time control loop.
    robot.control(impedance_control_callback, joint_vel_callback);

  } catch (const franka::Exception& ex) {
    running = false;
    std::cerr << ex.what() << std::endl;
  }

  if (publish_thread.joinable()) {
    publish_thread.join();
  }

  if (gripper_thread.joinable()) {
    gripper_thread.join();
  }
  return 0;
}
