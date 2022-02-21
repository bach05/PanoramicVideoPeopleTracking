#ifndef MONOCULAR_PEOPLE_TRACKING_TRACKSYSTEM_HPP
#define MONOCULAR_PEOPLE_TRACKING_TRACKSYSTEM_HPP

#include <Eigen/Dense>
#include <ros/node_handle.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/CameraInfo.h>
// ADDED
#include <algorithm>
#include <vector>
#include <math.h>

namespace monocular_people_tracking
{

  class TrackSystem
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TrackSystem(ros::NodeHandle &private_nh, const std::shared_ptr<tf::TransformListener> &tf_listener, const std::string &camera_frame_id, const int image_w, const int image_h)
        : camera_frame_id(camera_frame_id),
          tf_listener(tf_listener)
    {
      dt = 0.1;

      measurement_noise = Eigen::Matrix4f::Identity() * private_nh.param<double>("measurement_noise_pix_cov", 100);
      process_noise.setIdentity();
      process_noise.middleRows(0, 2) *= private_nh.param<double>("process_noise_pos_cov", 0.1);
      process_noise(2, 2) = private_nh.param<double>("process_noise_height_cov", 1e-10);
      process_noise.middleRows(3, 2) *= private_nh.param<double>("process_noise_vel_cov", 0.1);

      update_matrices(ros::Time(0));

      // ADDED
      im_height = image_h;
      im_width = image_w;
      CAM_HEIGHT = private_nh.param<double>("camera_height", 1.0);

      ROS_INFO("track system initialized");
    }

    void update_matrices(const ros::Time &stamp)
    {
      odom2camera = lookup_eigen(camera_frame_id, "odom", stamp);
      odom2footprint = lookup_eigen("base_footprint", "odom", stamp);
      footprint2base = lookup_eigen("base_link", "base_footprint", stamp);
      footprint2camera = lookup_eigen(camera_frame_id, "base_footprint", stamp);
    }

    Eigen::Isometry3f lookup_eigen(const std::string &to, const std::string &from, const ros::Time &stamp)
    {
      tf::StampedTransform transform;
      try
      {
        tf_listener->waitForTransform(to, from, stamp, ros::Duration(1.0));
        tf_listener->lookupTransform(to, from, stamp, transform);
      }
      catch (tf::ExtrapolationException &e)
      {
        tf_listener->waitForTransform(to, from, ros::Time(0), ros::Duration(5.0));
        tf_listener->lookupTransform(to, from, ros::Time(0), transform);
      }

      Eigen::Isometry3d iso;
      tf::transformTFToEigen(transform, iso);
      return iso.cast<float>();
    }

    Eigen::Vector3f transform_odom2camera(const Eigen::Vector3f &pos_in_odom) const
    {
      return (odom2camera * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
    }

    Eigen::Vector3f transform_odom2footprint(const Eigen::Vector3f &pos_in_odom) const
    {
      return (odom2footprint * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
    }

    Eigen::Vector3f transform_footprint2odom(const Eigen::Vector3f &pos_in_footprint) const
    {
      return (odom2footprint.inverse() * Eigen::Vector4f(pos_in_footprint.x(), pos_in_footprint.y(), pos_in_footprint.z(), 1.0f)).head<3>();
    }

    void set_dt(double d)
    {
      dt = std::max(d, 1e-9);
    }

    // interface for UKF
    Eigen::VectorXf f(const Eigen::VectorXf &state, const Eigen::VectorXf &control) const
    { // state: x, y, z, vx, vy
      Eigen::VectorXf next_state = state;
      next_state.middleRows(0, 2) += dt * state.middleRows(3, 2); // <x, y> += dt * <vx, vy> Increment 3d position using vx and vy velocities
      return next_state;
    }

    Eigen::MatrixXf processNoiseCov() const
    {
      return process_noise;
    }

    template <typename Measurement>
    Measurement h(const Eigen::VectorXf &state) const;

    template <typename Measurement>
    Eigen::MatrixXf measurementNoiseCov() const;

    double alpha(double x)
    {
      double angle = -1.0 * ((x / (im_width / 400.0)) - 200.0);
      return angle;
    }

    double beta(double y_ankle)
    {
      return ((y_ankle / (im_height / 180.0)) - 90) / -1;
    }

    double gamma(double y_neck)
    {
      return ((y_neck / (im_height / 180.0)) - 90) / -1;
    }

    double distance(double beta)
    {
      beta = abs(beta);
      return (CAM_HEIGHT) / tan(beta * M_PI / 180);
    }

    Eigen::Vector3f XYZw(const Eigen::Vector4f &observation)
    {
      assert(observation[0] >= 0 && observation[0] <= im_width);
      assert(observation[1] >= 0 && observation[1] <= im_height);
      assert(observation[2] >= 0 && observation[2] <= im_width);
      assert(observation[3] >= 0 && observation[3] <= im_height);
      double a = alpha(observation[2]);
      double b = beta(observation[3]);
      double g = gamma(observation[1]);
      double dist = distance(b);
      double Xw = dist * cos(a * M_PI / 180);
      double Yw = dist * sin(a * M_PI / 180);
      double Zw = CAM_HEIGHT + (dist * tan(g * M_PI / 180));
      Eigen::Vector3f pos(Xw, Yw, Zw);
      return pos;
    }

    Eigen::Vector4f xy_image(const Eigen::Vector3f &state) const
    {
      double Xw = state[0];
      double Yw = state[1];
      double Zw = state[2];
      double dist = sqrt(Xw * Xw + Yw * Yw);

      double x_ankle = (acos(Xw / dist) * -180.0 / M_PI + 180.0) * im_width / 360.0;
      if (Yw < 0)
        x_ankle = (acos(Xw / dist) * +180.0 / M_PI + 180.0) * im_width / 360.0;

      double y_ankle = (90.0 - (atan(CAM_HEIGHT / dist) * -180 / M_PI)) * im_height / 180.0;

      double x_neck = x_ankle;
      double y_neck = (90.0 - (atan((Zw - CAM_HEIGHT) / dist) * 180 / M_PI)) * im_height / 180.0;

      Eigen::Vector4f observation(x_neck, y_neck, x_ankle, y_ankle);

      return observation;
    }

  public:
    double dt;

    Eigen::Isometry3f odom2camera;
    Eigen::Isometry3f odom2footprint;
    Eigen::Isometry3f footprint2base;
    Eigen::Isometry3f footprint2camera;
    Eigen::Matrix3f camera_matrix;

    std::string camera_frame_id;
    std::shared_ptr<tf::TransformListener> tf_listener;

    Eigen::Matrix4f measurement_noise;
    Eigen::Matrix<float, 5, 5> process_noise;

    // ADDED
    double CAM_HEIGHT;
    double im_height;
    double im_width;
  };

}

#endif // TRACKSYSTEM_CPP
