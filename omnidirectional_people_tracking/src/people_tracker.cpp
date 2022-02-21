#include <monocular_people_tracking/people_tracker.hpp>

#include <kkl/alg/data_association.hpp>
#include <kkl/alg/nearest_neighbor_association.hpp>

#include <monocular_people_tracking/track_system.hpp>


namespace monocular_people_tracking {

//PeopleTracker::PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
PeopleTracker::PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const int image_w, const int image_h) {
    id_gen = 0;
    remove_trace_thresh = private_nh.param<double>("tracking_remove_trace_thresh", 5.0);
    dist_to_exists_thresh = private_nh.param<double>("tracking_newtrack_dist2exists_thersh", 100.0);

    data_association.reset(new kkl::alg::NearestNeighborAssociation<PersonTracker::Ptr, Observation::Ptr, AssociationDistance>(AssociationDistance(private_nh)));
    track_system.reset(new TrackSystem(private_nh, tf_listener, camera_frame_id, image_w, image_h));
    ROS_INFO("PeopleTracker initialized");
}

PeopleTracker::~PeopleTracker() {

}

void PeopleTracker::predict(ros::NodeHandle& nh, const ros::Time& stamp) {
    
    track_system->update_matrices(stamp);
    for(const auto& person : people) {
        person->predict(stamp);
    }
}

void PeopleTracker::correct(ros::NodeHandle& nh, const ros::Time& stamp, const std::vector<Observation::Ptr>& observations) {
    //ROS_INFO("People tracker correct...");
    if(!observations.empty()) {

        std::vector<bool> associated(observations.size(), false); // vector of boolean which tells if an observations is associated to a person
        auto associations = data_association->associate(people, observations); // Associate people to observations
        //ROS_INFO("ASSOCIATED : %d",associations.size());
        for(const auto& assoc : associations) {
            
            associated[assoc.observation] = true;
            people[assoc.tracker]->correct(stamp, observations[assoc.observation]);
        }

        for(int i=0; i<observations.size(); i++) {
            //ROS_INFO("observation %d", i);
            if(!associated[i] && observations[i]->ankle) {
                if(observations[i]->min_distance && *observations[i]->min_distance < dist_to_exists_thresh) {
                    ROS_WARN("discarded - dist_to_exist");
                    continue;
                }

                if(observations[i]->close2border) {
                    ROS_WARN("discarded - close2border");
                    continue;
                }

                //ROS_INFO("Create new tracker");

                PersonTracker::Ptr tracker(new PersonTracker(nh, track_system, stamp, id_gen++, observations[i]->neck_ankle_vector()));
                Eigen::Vector3f pos = tracker->pos();
                //ROS_INFO("INIT : x, y = %4.2f, %4.2f", pos.x(), pos.y());
                tracker->correct(stamp, observations[i]);
                Eigen::Vector3f pos2 = tracker->pos();
                //ROS_INFO("CORR : x, y = %4.2f, %4.2f", pos2.x(), pos2.y());
                people.push_back(tracker);
            }
        }
    }

    auto remove_loc = std::partition(people.begin(), people.end(), [&](const PersonTracker::Ptr& tracker) {
        return tracker->trace() < remove_trace_thresh;
    });
    removed_people.clear();
    std::copy(remove_loc, people.end(), std::back_inserter(removed_people));
    people.erase(remove_loc, people.end());
}

}
