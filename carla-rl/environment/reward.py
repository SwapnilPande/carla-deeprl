import numpy as np
import carla

def compute_reward(prev, current, config, verbose=False):
    # Convenience variable
    reward_config = config.reward_config

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    dist_to_goal_reward = prev_dist - cur_dist

    

    # Steer Reward
    steer = np.abs(current['control_steer'])
    steer_reward = -reward_config.steer_penalty_coeff * steer
    current["steer_reward"] = steer_reward

    # Speed Reward
    speed_reward = reward_config.speed_coeff * current["speed"]
    current["speed_reward"] = speed_reward

    # Acceleration Reward
    acceleration_reward = reward_config.acceleration_coeff * (current["speed"] - prev["speed"])
    current["acceleration_reward"] = acceleration_reward

    # Dist_to_trajectory reward
    if verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    dist_to_trajectory_reward = -reward_config.dist_to_trajectory_coeff * np.abs(current['dist_to_trajectory'])
    current["dist_to_trajectory_reward"] = dist_to_trajectory_reward


    # Light Reward

    light_reward = 0
    current["runover_light"] = False
    # Only compute reward if traffic light enabled
    if (not config.scenario_config.disable_traffic_light):
        if (_check_if_signal_crossed(prev, current) # Signal Crossed
            and (prev['nearest_traffic_actor_state'] == str(carla.TrafficLightState.Red)) # Light is red
            and (current["speed"] > config.scenario_config.zero_speed_threshold) # Vehicle is moving forward
            and (prev['initial_dist_to_red_light'] > config.obs_config.min_dist_from_red_light)): # We are within threshold distance of red light

            # Add reward if these conditions are true
            current["runover_light"] = True
            light_reward = -1 * (reward_config.const_light_penalty + reward_config.light_penalty_speed_coeff * current["speed"])
    current["light_reward"] = light_reward

    # Collision Reward
    is_collision = False
    lane_change = False
    obs_collision = (current["num_collisions"] - prev["num_collisions"]) > 0
    is_collision = obs_collision

    current['obs_collision'] = obs_collision
    current["is_collision"] = is_collision

    if(is_collision):
        # Using prev_speed in collision reward computation
        # due to non-determinism in speed at the time of collision
        collision_reward = -1 * (reward_config.const_collision_penalty + reward_config.collision_penalty_speed_coeff * prev["speed"])
        speed_reward = reward_config.speed_coeff * prev["speed"]

    else:
        collision_reward = 0
    current["collision_reward"] = collision_reward

    # Out of lane reward
    is_out_of_lane = False
    lane_change = False

    if not config.obs_config.disable_lane_invasion_sensor:
        print('lane invasion sensor is not disabled')
        is_out_of_lane = current["out_of_road"]

        # count any lane change also as a collision
        # if config.scenario_config.disable_lane_invasion_collision:
        lane_change = current['num_laneintersections'] > 0
        is_out_of_lane = is_out_of_lane or lane_change

    current['lane_change'] = lane_change
    current["is_out_of_lane"] = is_out_of_lane

    if(is_out_of_lane):
        out_of_lane_reward = -1 * (reward_config.const_out_of_lane_penalty + reward_config.out_of_lane_penalty_speed_coeff * prev["speed"])
    else:
        out_of_lane_reward = 0
    current["out_of_lane_reward"] = out_of_lane_reward


    # Success Reward
    success_reward = 0
    success = current["distance_to_goal"] < config.scenario_config.dist_for_success
    if success:
        success_reward += reward_config.success_reward
    current["success_reward"] = success_reward

    reward = dist_to_trajectory_reward + \
             speed_reward + steer_reward + \
             collision_reward + \
             out_of_lane_reward + \
             light_reward + \
             acceleration_reward + \
             success_reward +  \
             reward_config.constant_positive_reward +  \
             dist_to_goal_reward

    #print('dist_to_goal_reward',dist_to_goal_reward)
    #print('dist to trajectory',dist_to_trajectory_reward)

    # normalize reward
    reward = reward / reward_config.reward_normalize_factor


    current["step_reward"] = reward

    if verbose:
        print("dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, light_reward, steer_reward, success_reward, reward")
        print(dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, light_reward, steer_reward, success_reward, reward)


    if current["speed"] <= config.scenario_config.zero_speed_threshold:
        current["static_steps"] += 1
    return reward

def _check_if_signal_crossed(prev, current):

    # cross_from_one_light_to_no_light
    cross_to_no_light = current['dist_to_light'] == -1 and prev['dist_to_light'] > 0

    cross_to_next_light = (current['nearest_traffic_actor_id'] != -1 and prev['nearest_traffic_actor_id'] != -1 \
        and current['nearest_traffic_actor_id'] != prev['nearest_traffic_actor_id'])

    return cross_to_no_light or cross_to_next_light

def _compute_reward_corlT(prev, current, verbose=False):
    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    # Distance travelled toward the goal in m
    #distance_reward = np.clip(prev_dist - cur_dist, -10.0, 10.0)
    distance_reward = 1/(cur_dist)**0.5
    current["distance_reward"] = distance_reward

    # Change in speed (km/h)
    speed_reward = 0.05 * (current["speed"] - prev["speed"])
    current["speed_reward"] = speed_reward

    # Collision damage
    collision_reward = -.00002 * (current["num_collisions"] - prev["num_collisions"])
    current["collision_reward"] = collision_reward

    # New sidewalk intersection
    lane_intersection_reward = -2 * (current["num_laneintersections"] - prev["num_laneintersections"])
    current["lane_intersection_reward"] = lane_intersection_reward

    reward = distance_reward + speed_reward + collision_reward + lane_intersection_reward

    # Update state variables
    if np.absolute(lane_intersection_reward) > 0:
        current["offlane_steps"] += 1
    if current["speed"] == 0:
        current["static_steps"] += 1
    return reward