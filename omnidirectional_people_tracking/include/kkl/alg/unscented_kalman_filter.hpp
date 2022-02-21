/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <memory>
#include <Eigen/Dense>

namespace kkl
{
  namespace alg
  {

    /**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
    template <typename T, class System>
    class UnscentedKalmanFilterX
    {
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;

    public:
      /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
      UnscentedKalmanFilterX(const std::shared_ptr<System> &system, const VectorXt &mean, const MatrixXt &cov)
          : mean(mean),
            cov(cov),
            system(system),
            lambda(1.0)
      {
      }

      /**
   * @brief predict
   * @param control  input vector
   */
      void predict(const VectorXt &control)
      {
        ROS_INFO("predict");
        const int N = mean.rows();
        VectorXt weights = calcWeights(N);

        // calculate sigma points
        ensurePositiveFinite(cov);
        MatrixXt sigma_points = computeSigmaPoints(mean, cov);
        ROS_INFO("mean (%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",mean.rows(), mean[0], mean[1], mean[2], mean[3], mean[4]);
        for (int i = 0; i < 2 * N + 1; i++)
        {
          ROS_INFO("sigma b(%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",i, sigma_points.row(i)[0], sigma_points.row(i)[1], sigma_points.row(i)[2], sigma_points.row(i)[3], sigma_points.row(i)[4]);
          sigma_points.row(i) = system->f(sigma_points.row(i), control); // Mapping sigma points using the f function
          ROS_INFO("sigma a(%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",i, sigma_points.row(i)[0], sigma_points.row(i)[1], sigma_points.row(i)[2], sigma_points.row(i)[3], sigma_points.row(i)[4]);
        }

        // unscented transform
        VectorXt mean_pred(mean.size());
        MatrixXt cov_pred(cov.rows(), cov.cols());

        mean_pred.setZero();
        cov_pred.setZero();
        for (int i = 0; i < 2 * N + 1; i++)
        {
          mean_pred += weights[i] * sigma_points.row(i); // Computing the mean of the mapped sigma points
        }
        for (int i = 0; i < 2 * N + 1; i++)
        {
          VectorXt diff = sigma_points.row(i).transpose() - mean;
          cov_pred += weights[i] * diff * diff.transpose(); // COmputing the covariance of the mapped sigma points
        }
        cov_pred += system->processNoiseCov(); // Just add some noise

        mean = mean_pred;
        cov = cov_pred;
        ROS_INFO("prediction (%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",mean.rows(), mean[0], mean[1], mean[2], mean[3], mean[4]);
      }

      /**
   * @brief correct
   * @param measurement  measurement vector // x,y neck x,y ankles
   */
      template <typename Measurement>
      void correct(const Measurement &measurement)
      {
        //ROS_INFO("correct");
        const int N = mean.rows();        // This is the state space dimensionality
        const int K = measurement.rows(); // This is two or four depending on the observation if it contains both neck and ankle data
        //ROS_INFO("N : %d, K : %d", N, K);

        VectorXt ext_weights = calcWeights(N + K);

        // create extended state space which includes error variances
        VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
        MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
        ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
        ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
        ext_cov_pred.bottomRightCorner(K, K) = system->template measurementNoiseCov<Measurement>(); // Just adding some noise to the measurement

        ensurePositiveFinite(ext_cov_pred);
        MatrixXt ext_sigma_points = computeSigmaPoints(ext_mean_pred, ext_cov_pred);

        // unscented transform

        MatrixXt expected_measurements(2 * (N + K) + 1, K);
        double mean_x_neck = 0.0;
        double mean_x_ankle = 0.0;
        bool flag = expected_measurements.cols() == 4;
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          expected_measurements.row(i) = system->template h<Measurement>(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
          //ROS_INFO("EXPECTED MEASUREMENT B4 : %2.4f, %2.4f, %2.4f, %2.4f  (from sigma %2.4f, %2.4f)", expected_measurements.row(i)[0], expected_measurements.row(i)[1], expected_measurements.row(i)[2], expected_measurements.row(i)[3], ext_sigma_points.row(i)[0], ext_sigma_points.row(i)[1]);
          expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
          //ROS_INFO("EXPECTED MEASUREMENT AF: %2.4f, %2.4f, %2.4f, %2.4f  (from sigma %2.4f, %2.4f)", expected_measurements.row(i)[0], expected_measurements.row(i)[1], expected_measurements.row(i)[2], expected_measurements.row(i)[3], ext_sigma_points.row(i)[0], ext_sigma_points.row(i)[1]);
          mean_x_neck += expected_measurements.row(i).x();
          if (flag)
            mean_x_ankle += expected_measurements.row(i)[2];
        }
        mean_x_neck /= ext_sigma_points.rows();
        if (flag)
          mean_x_ankle /= ext_sigma_points.rows();
        //ROS_WARN("Mean_ankle SIGMA : %4.3f | Mean_neck SIGMA : %4.3f", mean_x_ankle, mean_x_neck);

        VectorXt expected_measurement_mean = VectorXt::Zero(K);
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          if (abs(expected_measurements.row(i).x() - mean_x_neck) > 480 || (flag && abs(expected_measurements.row(i)[2] - mean_x_ankle) > 480))
          {
            if (mean_x_neck < 480 || (flag && mean_x_ankle < 480))
            {
              expected_measurements.row(i).x() -= 960;
              if (flag)
                expected_measurements.row(i)[2] -= 960;
            }
            else
            {
              expected_measurements.row(i).x() += 960;
              if (flag)
                expected_measurements.row(i)[2] += 960;
            }
            //ext_weights[i] = 0;
          }
          /*
          if (expected_measurements.row(i).x() <= 0 || expected_measurements.row(i).x() >= 960)
            ROS_WARN("SIGMA%2d : %4.2f, weight : %4.4f", i, expected_measurements.row(i).x(), ext_weights[i]);
          */
          expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
        }
        assert(expected_measurement_mean[0] >= 0 && expected_measurement_mean[0] <= 960);
        if (flag)
          assert(expected_measurement_mean[2] >= 0 && expected_measurement_mean[2] <= 960);
        //ROS_INFO("EXPECTED MEAN : %2.4f, %2.4f, %2.4f, %2.4f", expected_measurement_mean[0], expected_measurement_mean[1], expected_measurement_mean[2], expected_measurement_mean[3]);
        
        
        MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
          ROS_INFO("exps(%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",i, expected_measurements.row(i).transpose()[0], expected_measurements.row(i).transpose()[1], expected_measurements.row(i).transpose()[2], expected_measurements.row(i).transpose()[3], expected_measurements.row(i).transpose()[4]);
          ROS_INFO("expm(%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",i, expected_measurement_mean[0], expected_measurement_mean[1], expected_measurement_mean[2], expected_measurement_mean[3], expected_measurement_mean[4]);
          ROS_INFO("diff(%d) : %2.4f, %2.4f, %2.4f, %2.4f, %2.4f",i, diff[0], diff[1], diff[2], diff[3], diff[4]);
          expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
        }

        // calculated transformed covariance
        MatrixXt sigma = MatrixXt::Zero(N + K, K); // 9x4 9x2
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
          auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
          sigma += ext_weights[i] * (diffA * diffB.transpose());
        }

        MatrixXt kalman_gain = sigma * expected_measurement_cov.inverse();
        
        VectorXt innovation = (measurement - expected_measurement_mean); // Measurement residual (innovation)
        
        if (innovation[0] < -480 || innovation[0] > 480)
        {
          if (innovation[0] < 0)
          {
            innovation[0] += 960;
          }
          else
          {
            innovation[0] -= 960;
          }
          if (flag)
            innovation[2] = innovation[0];
        }

        VectorXt ext_mean = ext_mean_pred + kalman_gain * innovation; //(measurement - expected_measurement_mean);
        MatrixXt ext_cov = ext_cov_pred - kalman_gain * expected_measurement_cov * kalman_gain.transpose();
        mean = ext_mean.topLeftCorner(N, 1);
        cov = ext_cov.topLeftCorner(N, N);
      }

      template <typename Measurement>
      std::pair<VectorXt, MatrixXt> expected_measurement_distribution() const
      {
        //ROS_INFO("expected_measurement_distribution");
        const int N = mean.rows();                    // 5
        const int K = Measurement::RowsAtCompileTime; // 4
        //ROS_INFO("N : %d, K : %d", N, K);

        VectorXt ext_weights = calcWeights(N + K); // 5 + 4

        // create extended state space which includes error variances
        VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);                                          // 9, 1
        MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);                                       // 9, 9
        ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);                                         // first 5 equal to mean
        ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);                                           // first 5x5 equal to cov
        ext_cov_pred.bottomRightCorner(K, K) = system->template measurementNoiseCov<Measurement>(); // last 4x4 equal to constant noise

        ensurePositiveFinite(ext_cov_pred);
        MatrixXt ext_sigma_points = computeSigmaPoints(ext_mean_pred, ext_cov_pred); // 19x9

        // unscented transform
        double mean_x_neck = 0.0;
        double mean_x_ankle = 0.0;
        MatrixXt expected_measurements(2 * (N + K) + 1, K); // 19x4
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          expected_measurements.row(i) = system->template h<Measurement>(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));

          expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));

          mean_x_neck += expected_measurements.row(i).x();
          mean_x_ankle += expected_measurements.row(i)[2];
        }
        mean_x_neck /= ext_sigma_points.rows();
        mean_x_ankle /= ext_sigma_points.rows();
        //ROS_WARN("Mean_ankle SIGMA : %4.3f | Mean_neck SIGMA : %4.3f", mean_x_ankle, mean_x_neck);

        VectorXt expected_measurement_mean = VectorXt::Zero(K); // 4
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          if (abs(expected_measurements.row(i).x() - mean_x_neck) > 480 || abs(expected_measurements.row(i)[2] - mean_x_ankle) > 480)
          {
            if (mean_x_neck < 480 || mean_x_ankle < 480)
            {
              expected_measurements.row(i).x() -= 960;
              expected_measurements.row(i)[2] -= 960;
            }
            else
            {
              expected_measurements.row(i).x() += 960;
              expected_measurements.row(i)[2] += 960;
            }
          }
          expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
        }
        MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K); // 4x4
        for (int i = 0; i < ext_sigma_points.rows(); i++)
        {
          VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean; // 4
          expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
        }

        //ROS_WARN("EXPECTED MEASURE : %4.2f, %4.2f", expected_measurement_mean.x(), expected_measurement_mean.y());

        assert(expected_measurement_mean[0] >= 0 && expected_measurement_mean[0] <= 960);

        return std::make_pair(expected_measurement_mean, expected_measurement_cov); // 4, 4x4
      }

      /*			getter			*/
      const VectorXt &getMean() const { return mean; }
      const MatrixXt &getCov() const { return cov; }

      System &getSystem() { return *system; }
      const System &getSystem() const { return *system; }

      /*			setter			*/
      UnscentedKalmanFilterX &setMean(const VectorXt &m)
      {
        mean = m;
        return *this;
      }
      UnscentedKalmanFilterX &setCov(const MatrixXt &s)
      {
        cov = s;
        return *this;
      }

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      /**
   * @brief calculate weights for sigma points
   * @param N   dimension of state
   * @return weights (2 * N + 1)
   */
      VectorXt calcWeights(int N) const
      {
        VectorXt weights(2 * N + 1, 1); // 2 * 9 + 1 = 19
        weights[0] = lambda / (N + lambda);
        for (int i = 1; i < 2 * N + 1; i++)
        {
          weights[i] = 1 / (2 * (N + lambda));
        }
        return weights;
      }

      /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @return calculated sigma points
   */
      MatrixXt computeSigmaPoints(const VectorXt &mean, const MatrixXt &cov) const
      {
        const int n = mean.size();                  // 9
        assert(cov.rows() == n && cov.cols() == n); // 9x9

        MatrixXt sigma_points(2 * n + 1, n); // 19x9

        Eigen::LLT<MatrixXt> llt;
        llt.compute((n + lambda) * cov); // Cholesky deconposition
        MatrixXt l = llt.matrixL();

        sigma_points.row(0) = mean; // The first sigma point is the mean
        //ROS_INFO("SIGMA#00 : %4.2f, %4.2f", sigma_points.row(0).x(), sigma_points.row(0).y());
        for (int i = 0; i < n; i++)
        {
          sigma_points.row(1 + i * 2) = mean + l.col(i);
          //ROS_INFO("SIGMA#%2d : %4.2f, %4.2f", (1 + i * 2), sigma_points.row(1 + i * 2).x(), sigma_points.row(1 + i * 2).y());
          sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
          //ROS_INFO("SIGMA#%2d : %4.2f, %4.2f", (1 + i * 2 + 1), sigma_points.row(1 + i * 2 + 1).x(), sigma_points.row(1 + i * 2 + 1).y());
        }

        return sigma_points;
      }

      /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
      void ensurePositiveFinite(MatrixXt &cov) const
      {
        return;
        const double eps = 1e-9;

        Eigen::EigenSolver<MatrixXt> solver(cov);
        MatrixXt D = solver.pseudoEigenvalueMatrix();
        MatrixXt V = solver.pseudoEigenvectors();
        for (int i = 0; i < D.rows(); i++)
        {
          if (D(i, i) < eps)
          {
            D(i, i) = eps;
          }
        }

        cov = V * D * V.inverse();
      }

    public:
      const T lambda;
      std::shared_ptr<System> system;

      VectorXt mean;
      MatrixXt cov;
    };

  }
}

#endif
