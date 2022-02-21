#include <mutex>
#include <iostream>
#include <chrono>
#include <unordered_map>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>

#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <tfpose_ros/Persons.h>
#include <monocular_person_following/Target.h>
#include <monocular_person_following/Imprint.h>
#include <monocular_person_following/FaceDetectionArray.h>
#include <monocular_people_tracking/TrackArray.h>

#include <monocular_person_following/context.hpp>
#include <monocular_person_following/tracklet.hpp>
#include <monocular_person_following/state/state.hpp>
#include <monocular_person_following/state/initial_state.hpp>
#include <monocular_person_following/state/initial_training_state.hpp>

using namespace std::chrono;

namespace monocular_person_following
{

    class MonocularPersonFollowingNode
    {
    public:
        MonocularPersonFollowingNode()
            : nh(),
              private_nh("~"),
              target_pub(nh.advertise<Target>("/monocular_person_following/target", 1)),
              image_trans(nh),
              features_pub(image_trans.advertise("/monocular_person_following/features", 1)),
              image_sub(nh, "image", 10),
              tracks_sub(nh, "/monocular_people_tracking/tracks_corrected", 10),
              //tracks_sub(nh.subscribe("/monocular_people_tracking/tracks", 10, &MonocularPersonFollowingNode::tracks_callback, this)),
              faces_sub(nh, "/face_detector/faces", 10),
              sync(image_sub, tracks_sub, 30),
              sync_w_face(image_sub, tracks_sub, faces_sub, 30),
              reset_sub(private_nh.subscribe<std_msgs::Empty>("reset", 10, &MonocularPersonFollowingNode::reset_callback, this)),
              reset_service_server(private_nh.advertiseService("reset", &MonocularPersonFollowingNode::reset_service, this)),
              imprint_service_server(private_nh.advertiseService("imprint", &MonocularPersonFollowingNode::imprint_service, this))
        {
            state.reset(new InitialState());
            context.reset(new Context(private_nh));
            
            if (private_nh.param<bool>("use_face", true))
            {
                sync_w_face.registerCallback(boost::bind(&MonocularPersonFollowingNode::callback, this, _1, _2, _3));
            }
            else
            {
                sync.registerCallback(boost::bind(&MonocularPersonFollowingNode::callback, this, _1, _2, nullptr));
            }
            
            last_call = high_resolution_clock::now();
        }

        void tracks_callback(const monocular_people_tracking::TrackArrayConstPtr &tracks_msg){
            callback(nullptr, tracks_msg, nullptr);
        }

        void callback(const sensor_msgs::ImageConstPtr &image_msg, const monocular_people_tracking::TrackArrayConstPtr &tracks_msg, const monocular_person_following::FaceDetectionArrayConstPtr &faces_msg)
        {

            auto start = high_resolution_clock::now();

            auto stop1 = high_resolution_clock::now();
            duration<double, std::milli> elapsed = stop1 - last_call;
            last_call = stop1;

            ROS_INFO("ELAPSED SINCE LAST TARGET time : %4.4f ms", elapsed.count());

            auto cv_image = cv_bridge::toCvCopy(image_msg, "bgr8");
            //auto cv_image = cv_bridge::toCvCopy(tracks_msg->image, "bgr8");

            std::unordered_map<long, Tracklet::Ptr> tracks;

            std::unordered_map<long, FaceDetection const *> face_msgs;
            if (faces_msg != nullptr)
            {
                for (const auto &face : faces_msg->faces)
                {
                    face_msgs[face.track_id] = &face;
                }
            }

            for (const auto &track : tracks_msg->tracks)
            {
                tracks[track.id].reset(new Tracklet(tf_listener, tracks_msg->header, track)); // Create a tracklet for each track in msg

                if (track.associated_neck_ankle.empty())
                { // No more processing if no associated observation
                    continue;
                }

                cv::Rect person_region = calc_person_region(track, cv_image->image.size());
                tracks[track.id]->person_region = person_region;

                auto face = face_msgs.find(track.id);
                if (face != face_msgs.end() && !face->second->face_image.empty())
                {
                    auto face_image = cv_bridge::toCvCopy(face->second->face_image[0], "bgr8");
                    tracks[track.id]->face_image = face_image->image;
                }
            }

            std::lock_guard<std::mutex> lock(context_mutex);
            context->extract_features(cv_image->image, tracks);

            State *next_state = state->update(private_nh, *context, tracks);
            if (next_state != state.get())
            {
                state.reset(next_state);
            }

            if (target_pub.getNumSubscribers())
            {
                Target target;
                target.header = image_msg->header;
                //target.header = tracks_msg->image.header;
                target.state.data = state->state_name();
                target.target_id = state->target();

                if (tracks.find(target.target_id) == tracks.end())
                {
                    ROS_INFO("Target not available");
                    target.center_of_mass.x = -1.0;
                    target.center_of_mass.y = -1.0;
                    target.center_of_mass.z = -1.0;
                }
                else
                {
                    //ROS_INFO("ASS : %d" ,tracks[target.target_id]->track_msg->associated_neck_ankle.size());
                    if (tracks[target.target_id]->track_msg->associated_neck_ankle.size() == 2)
                    {
                        Eigen::Vector2f neck(tracks[target.target_id]->track_msg->associated_neck_ankle[0].x, tracks[target.target_id]->track_msg->associated_neck_ankle[0].y);
                        Eigen::Vector2f ankle(tracks[target.target_id]->track_msg->associated_neck_ankle[1].x, tracks[target.target_id]->track_msg->associated_neck_ankle[1].y);
                        Eigen::Vector2f center = (neck + ankle) / 2.0f;
                        target.center_of_mass.x = center[0];
                        target.center_of_mass.y = center[1];
                        target.center_of_mass.z = 2.0;
                    }
                    else if (tracks[target.target_id]->track_msg->associated_neck_ankle.size() == 1)
                    {
                        Eigen::Vector2f center(tracks[target.target_id]->track_msg->associated_neck_ankle[0].x, tracks[target.target_id]->track_msg->associated_neck_ankle[0].y);
                        target.center_of_mass.x = center[0];
                        target.center_of_mass.y = center[1];
                        target.center_of_mass.z = 1.0;
                    }
                    else
                    {
                        target.center_of_mass.x = 0.0;
                        target.center_of_mass.y = 0.0;
                        target.center_of_mass.z = 0.0;
                    }

                    geometry_msgs::Point pos = tracks[target.target_id]->track_msg->pos;
                    target.distance = sqrt(pos.x * pos.x + pos.y * pos.y);

                    ROS_INFO("TARGET_CALLBACK cm : %f, %f, %f | distance : %f", target.center_of_mass.x,target.center_of_mass.y,target.center_of_mass.z,target.distance);
                }

                target.track_ids.reserve(tracks_msg->tracks.size());
                target.confidences.reserve(tracks_msg->tracks.size());
                target.classifier_confidences.reserve(tracks_msg->tracks.size() * 2);

                std::vector<std::string> classifier_names = context->classifier_names();
                for (const auto &name : classifier_names)
                {
                    std_msgs::String classifier_name;
                    classifier_name.data = name;
                    target.classifier_names.push_back(classifier_name);
                }

                for (const auto &track : tracks)
                {
                    if (track.second->confidence)
                    {
                        target.track_ids.push_back(track.first);
                        target.confidences.push_back(*track.second->confidence);

                        if (track.second->classifier_confidences.size() != target.classifier_names.size())
                        {
                            ROS_ERROR_STREAM("num_classifiers did not match!!");
                            ROS_ERROR_STREAM(track.second->classifier_confidences.size() << " : " << target.classifier_names.size());
                        }
                        std::copy(track.second->classifier_confidences.begin(), track.second->classifier_confidences.end(), std::back_inserter(target.classifier_confidences));
                    }
                }

                target_pub.publish(target);
            }

            if (features_pub.getNumSubscribers())
            {
                cv::Mat features = context->visualize_body_features();
                if (features.data)
                {
                    cv_bridge::CvImage cv_image(image_msg->header, "bgr8", features);
                    features_pub.publish(cv_image.toImageMsg());
                }
            }
            auto stop = high_resolution_clock::now();
            duration<double, std::milli> ms_double = stop - start;

            ROS_INFO("TARGET_CALLBACK time : %4.4f ms", ms_double.count());
        }

        bool imprint_service(ImprintRequest &req, ImprintResponse &res)
        {
            reset(req.target_id);
            res.success = true;

            return true;
        }

        bool reset_service(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
        {
            reset();

            return true;
        }

        void reset_callback(const std_msgs::EmptyConstPtr &empty_msg)
        {
            reset();
        }

    private:
        cv::Rect2f calc_person_region(const monocular_people_tracking::Track &track, const cv::Size &image_size)
        {

            int track_image_w = 960;//1067;
            int track_image_h = 480;
            Eigen::Vector2f neck(track.associated_neck_ankle[0].x*(1.0*image_size.width/track_image_w), track.associated_neck_ankle[0].y*(1.0*image_size.height/track_image_h));
            Eigen::Vector2f ankle(track.associated_neck_ankle[1].x*(1.0*image_size.width/track_image_w), track.associated_neck_ankle[1].y*(1.0*image_size.height/track_image_h));

            Eigen::Vector2f center = (neck + ankle) / 2.0f; // Mean
            float height = (ankle.y() - neck.y()) * 1.25f;  // Estimate height
            float width = height * 0.25f;                   // Estimate width

            cv::Rect rect(center.x() - width / 2.0f, center.y() - height / 2.0f, width, height); // Create rect centered in center

            cv::Point tl = rect.tl(); // Top left point
            cv::Point br = rect.br(); // Botton right point

            // Assure the rect fits in the image
            tl.x = std::min(image_size.width-(int)round(width), std::max(0, tl.x));
            tl.y = std::min(image_size.height, std::max(0, tl.y));
            br.x = std::min(image_size.width, std::max((int)round(width), br.x));
            br.y = std::min(image_size.height, std::max(0, br.y));

            return cv::Rect(tl, br);
        }

        void reset(long target_id = -1)
        {
            ROS_INFO_STREAM("reset identification!!");
            std::lock_guard<std::mutex> lock(context_mutex);

            if (target_id < 0)
            {
                state.reset(new InitialState());
            }
            else
            {
                state.reset(new InitialTrainingState(target_id));
            }
            context.reset(new Context(private_nh));
        }

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        tf::TransformListener tf_listener;

        ros::Publisher target_pub;

        image_transport::ImageTransport image_trans;
        image_transport::Publisher features_pub;

        message_filters::Subscriber<sensor_msgs::Image> image_sub;
        message_filters::Subscriber<monocular_people_tracking::TrackArray> tracks_sub;
        //ros::Subscriber tracks_sub;
        message_filters::Subscriber<monocular_person_following::FaceDetectionArray> faces_sub;
        message_filters::TimeSynchronizer<sensor_msgs::Image, monocular_people_tracking::TrackArray> sync;
        message_filters::TimeSynchronizer<sensor_msgs::Image, monocular_people_tracking::TrackArray, monocular_person_following::FaceDetectionArray> sync_w_face;

        // reset service callbacks
        ros::Subscriber reset_sub;
        ros::ServiceServer reset_service_server;
        ros::ServiceServer imprint_service_server;

        std::mutex context_mutex;
        std::shared_ptr<State> state;
        std::unique_ptr<Context> context;

        high_resolution_clock::time_point last_call;
    };

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "monocular_person_following");
    std::unique_ptr<monocular_person_following::MonocularPersonFollowingNode> node(new monocular_person_following::MonocularPersonFollowingNode());
    ros::spin();

    return 0;
}
