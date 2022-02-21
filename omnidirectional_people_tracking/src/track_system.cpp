#include <monocular_people_tracking/track_system.hpp>

namespace monocular_people_tracking
{

  template <>
  Eigen::Vector2f TrackSystem::h(const Eigen::VectorXf &state) const
  {

    double xw = state[0];
    double yw = state[1];
    double zw = state[2];

    Eigen::Vector3f s(xw, yw, zw);
    Eigen::Vector4f full_observation = xy_image(s);

    Eigen::Vector2f observation(full_observation[0], full_observation[1]);
    return observation;
  }

  template <>
  Eigen::MatrixXf TrackSystem::measurementNoiseCov<Eigen::Vector2f>() const
  {
    return measurement_noise.block<2, 2>(0, 0);
  }

  template <>
  Eigen::Vector4f TrackSystem::h(const Eigen::VectorXf &state) const
  { // Given points in 3d space, return the expected measurements of neck and ankle

    //ROS_INFO("h 4f : %d", state.size());
    double xw = state[0];
    double yw = state[1];
    double zw = state[2];

    Eigen::Vector3f s(xw, yw, zw);
    Eigen::Vector4f observation = xy_image(s);

    return observation;
  }

  template <>
  Eigen::MatrixXf TrackSystem::measurementNoiseCov<Eigen::Vector4f>() const
  {
    return measurement_noise;
  }

}
