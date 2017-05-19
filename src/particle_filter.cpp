#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "particle_filter.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;

    for(int i=0; i<num_particles; i++) {

        // Create a Gaussian distribution for x, y and yaw
        default_random_engine gen;
        normal_distribution<double> gps_x_position(x, std[0]);
        normal_distribution<double> gps_y_position(y, std[1]);
        normal_distribution<double> initial_theta(theta, std[2]);

        Particle p;
        p.id = i;
        p.x = gps_x_position(gen);
        p.y = gps_y_position(gen);
        p.theta = initial_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    for(int i=0; i<num_particles; i++) {
        // If yaw_rate is zero
        if(fabs(yaw_rate) < 1e-6) {
            particles[i].x = particles[i].x + (velocity * delta_t) * cos(particles[i].theta);
            particles[i].y = particles[i].y + (velocity * delta_t) * sin(particles[i].theta);
        } else {
            particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
        }

		// Create a Gaussian distribution for x, y and yaw
        random_device rd;
	    default_random_engine gen(rd());
	    normal_distribution<double> gps_x_position(particles[i].x, std_pos[0]);
	    normal_distribution<double> gps_y_position(particles[i].y, std_pos[1]);
	    normal_distribution<double> pred_theta(particles[i].theta, std_pos[2]);

	    particles[i].x = gps_x_position(gen);
	    particles[i].y = gps_y_position(gen);
	    particles[i].theta = pred_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for(int obs=0; obs<observations.size(); obs++) {
        double obs_x = observations[obs].x;
        double obs_y = observations[obs].y;
        double min_distance = 99999999.0;

        for(int l=0; l<predicted.size(); l++) {
            double delta_x = obs_x - predicted[l].x;
            double delta_y = obs_y - predicted[l].y;

			double distance = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));

            if (distance < min_distance) {
                min_distance = distance;
                observations[obs].id = l;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    for(int i=0; i<num_particles; i++) {
        double curr_x = particles[i].x;
        double curr_y = particles[i].y;
        double curr_theta = particles[i].theta;

        vector<LandmarkObs> pred_landmarks;

        // Iterate through the map landmarks
        for(int j=0; j<map_landmarks.landmark_list.size(); j++) {
            int j_id = map_landmarks.landmark_list[j].id_i;
            double j_x = map_landmarks.landmark_list[j].x_f;
            double j_y = map_landmarks.landmark_list[j].y_f;

            double delta_x = j_x - curr_x;
            double delta_y = j_y - curr_y;

            double distance = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));

            // If landmark is in sensor range
            if(distance<=sensor_range) {
				// Transform from vehicle coordinate to map coordinate
                j_x = delta_x * cos(curr_theta) + delta_y * sin(curr_theta);
                j_y = delta_y * cos(curr_theta) - delta_x * sin(curr_theta);

                // Create an array for landmarks in range and push the new findings to it
                LandmarkObs inrange_landmarks;
                inrange_landmarks.id = j_id;
                inrange_landmarks.x = j_x;
                inrange_landmarks.y = j_y;
                pred_landmarks.push_back(inrange_landmarks);
            }
        }

        // Create data associattions between landmarks and observations
        dataAssociation(pred_landmarks, observations);

        for(int m=0; m<observations.size(); m++) {
            int j_id = observations[m].id;
            double obs_x = observations[m].x;
            double obs_y = observations[m].y;

            double delta_x = obs_x - pred_landmarks[j_id].x;
            double delta_y = obs_y - pred_landmarks[j_id].y;

			double x_diff = pow(delta_x,2.0) / (2 * std_landmark[0] * std_landmark[0]);
			double y_diff = pow(delta_y,2.0) / (2 * std_landmark[1] * std_landmark[1]);

            particles[i].weight *= 1 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]) * exp(-(x_diff + y_diff));
        }

        weights[i] = particles[i].weight;
        particles[i].weight = 1.0;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
	vector<Particle> resampled_particles;
    
	default_random_engine gen;

    for(int i=0; i<particles.size(); i++) {
        discrete_distribution<int> index(weights.begin(), weights.end());
        resampled_particles.push_back(particles[index(gen)]);
    }

    particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
