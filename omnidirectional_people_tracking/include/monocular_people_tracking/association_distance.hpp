#ifndef MONOCULAR_PEOPLE_TRACKING_ASSOCIATION_DISTANCE_HPP
#define MONOCULAR_PEOPLE_TRACKING_ASSOCIATION_DISTANCE_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <monocular_people_tracking/person_tracker.hpp>

namespace monocular_people_tracking
{

  class AssociationDistance
  {
  public:
    AssociationDistance(ros::NodeHandle &private_nh)
        : maha_sq_thresh(private_nh.param<double>("association_maha_sq_thresh", 9.0)),
          neck_ankle_max_dist(private_nh.param<int>("association_neck_ankle_max_dist", 200)),
          neck_max_dist(private_nh.param<int>("association_neck_max_dist", 100))
    {
    }

    double circularEuclideanDistance4(Eigen::Vector4f v1, Eigen::Vector4f v2) const
    {
      int MAX_X = 960;

      double dist = 0.0;

      double x0 = abs(v1[0] - v2[0]);
      double x1 = abs(v1[2] - v2[2]);

      dist += pow(std::min(x0, MAX_X - x0), 2);

      dist += pow(v1[1] - v2[1], 2);

      dist += pow(std::min(x1, MAX_X - x1), 2);

      dist += pow(v1[3] - v2[3], 2);

      dist = sqrt(dist);

      return dist;
    }

    double circularEuclideanDistance(Eigen::Vector2f v1, Eigen::Vector2f v2) const
    {
      int MAX_X = 960;

      double dist1 = 0.0;
      double dist2 = 0.0;

      dist1 += pow(v1[0] - v2[0], 2);
      dist2 += pow(MAX_X - abs(v1[0] - v2[0]), 2);

      dist1 += pow(v1[1] - v2[1], 2);
      dist2 += pow(v1[1] - v2[1], 2);

      dist1 = sqrt(dist1);
      dist2 = sqrt(dist2);

      return std::min(dist1, dist2);
    }

    boost::optional<double> operator()(const PersonTracker::Ptr &tracker, const Observation::Ptr &observation) const
    {
      //ROS_INFO("-------------- ASSOCIATION --------------------------------------------------------------");
      auto expected_measurement = tracker->expected_measurement_distribution();
      Eigen::Vector2f expected_neck = expected_measurement.first.head<2>();
      Eigen::Vector2f expected_ankle = expected_measurement.first.tail<2>();

      if (observation->ankle)
      { // If we have ankle data in the observation

        double distance = circularEuclideanDistance4(observation->neck_ankle_vector(), expected_measurement.first.head<4>());

        if (!observation->min_distance)
        { // Update the minimum distance
          observation->min_distance = distance;
        }
        else
        {
          observation->min_distance = std::min(distance, *observation->min_distance);
        }

        if (distance > neck_ankle_max_dist)
        {
          ROS_WARN("neck_ankle_max_dist : %f", distance);
          return boost::none;
        }

        double sq_maha = tracker->squared_mahalanobis_distance(observation->neck_ankle_vector());
        //ROS_WARN("sq_maha : %f", sq_maha);

        if (sq_maha > maha_sq_thresh)
        {
          ROS_WARN("maha_sq_thresh : %f", sq_maha);
          return boost::none;
        }

        return -tracker->prob(observation->neck_ankle_vector());
      }

      double distance = circularEuclideanDistance(observation->neck_vector(), expected_neck);
      if (distance > neck_max_dist)
      {
        ROS_WARN("neck_max_dist : %f", distance);
        return boost::none;
      }

      double sq_maha = tracker->squared_mahalanobis_distance(observation->neck_vector());
      if (sq_maha > maha_sq_thresh)
      {
        ROS_WARN("maha_sq_thresh : %f", sq_maha);
        return boost::none;
      }

      return -tracker->prob(observation->neck_vector()) + 1.0;
    }

  private:
    double maha_sq_thresh;
    int neck_ankle_max_dist;
    int neck_max_dist;
  };

}

#endif
