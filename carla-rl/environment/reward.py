import numpy as np
import carla

def compute_reward(name, prev_measurement, cur_measurement, config=None, verbose=False):
    if name == 'corl':
        reward = _compute_reward_corl(prev_measurement, cur_measurement, verbose=verbose)
    elif name == 'cirl':
        reward = _compute_reward_cirl(prev_measurement, cur_measurement, verbose=verbose)
    elif name == 'corl2':
        reward = _compute_reward_corl2(prev_measurement, cur_measurement, verbose=verbose)
    elif name == 'corlT':
        reward = _compute_reward_corlT(prev_measurement, cur_measurement, verbose=verbose)
    elif name == "simple":
        reward = _compute_reward_simple(prev_measurement, cur_measurement, verbose=verbose)
    elif name == "simple2":
        reward = _compute_reward_simple2(prev_measurement, cur_measurement, config, verbose=verbose)
    return reward

def _compute_reward_cirl(prev, current, verbose=False):
    # 1) Abnormal steer penalty
    """
    if (control.steer > 0) and (directions == 3):
        # Turn right when should go left
        steer_reward = -15
    elif (control.steer < 0) and (directions == 4):
        # Turn left when should go right
        steer_reward = -15
    elif (abs(control.steer) > 0.2) and (directions in [0, 2, 5]):
        # Turn when should go straight
        # TODO: directions 0, 2 could mean follow lane that is turning
        steer_reward = -20
    else:
        steer_reward = 0
    """
    steer_reward = 0
    current["steer_reward"] = steer_reward

    # 2) Collision penalty
    no_collisions = (current["num_collisions"] - prev["num_collisions"])
    collision = no_collisions > 0
    collision_reward = -30 if collision else 0
    current["collision_reward"] = collision_reward

    # 3) Sidewalk and opposite lane overlap penalty
    no_lane_intersections = (current["num_laneintersections"] - prev["num_laneintersections"])
    lane_change = no_lane_intersections > 0
    lane_intersection_reward = -30 if lane_change else 0
    current["lane_intersection_reward"] = lane_intersection_reward

    # 4) Speed reward (in km/h)
    #TODO: Incorporate directions once planner is ready. Default assumed to go straight.
    # converted to km/h
    speed = current["speed"] * 3.6
    speed_reward = speed if (speed < 30) else (60 - speed)
    # if directions in [0, 2]:
    #     # If following lane or going straight, limit speed to 30km/h
    #     speed_reward = speed if (speed < 30) else (60 - speed)
    # else:
    #     # If approaching intersection, limit speed to 20km/h
    #     speed_reward = speed if (speed < 20) else (40 - speed)
    current["speed_reward"] = speed_reward

    # Total reward (approximately scaled to [0, 1] range)
    reward = steer_reward + collision_reward + lane_intersection_reward + speed_reward
    reward /= 30

    if np.absolute(lane_intersection_reward) > 0:
        current["offlane_steps"] += 1
    if current["speed"] == 0:
        current["static_steps"] += 1

    return reward

def _compute_reward_corl(prev, current, verbose=False):
    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    distance_reward = 10000 * (prev_dist - cur_dist)
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

def _compute_reward_corl2(prev, current, verbose=False):
    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    goal_distance_reward = 1/(cur_dist)**0.5
    current["goal_distance_reward"] = goal_distance_reward

    # Distance travelled toward the goal in m
    distance_reward = 0.01 * (prev_dist - cur_dist)
    current["distance_reward"] = distance_reward

    # Change in speed (km/h)
    speed_reward = 0.05 * (current["speed"] - prev["speed"])
    current["speed_reward"] = speed_reward

    # Collision damage
    if((current["num_collisions"] - prev["num_collisions"]) > 0):
        collision_reward = -1
    else:
        collision_reward = 0
    current["collision_reward"] = collision_reward

    # New sidewalk intersection
    if((current["num_laneintersections"] - prev["num_laneintersections"]) > 0):
        lane_intersection_reward = -1
    else:
        lane_intersection_reward = 0
    current["lane_intersection_reward"] = lane_intersection_reward

    # # Collision damage
    # collision_reward = -.00002 * (current["num_collisions"] - prev["num_collisions"])
    # current["collision_reward"] = collision_reward

    # # New sidewalk intersection
    # lane_intersection_reward = -2 * (current["num_laneintersections"] - prev["num_laneintersections"])
    # current["lane_intersection_reward"] = lane_intersection_reward


    reward = goal_distance_reward + speed_reward + distance_reward + collision_reward + lane_intersection_reward

    print("goal_distance_reward, speed_reward, distance_reward, collision_reward, lane_intersection_reward, reward")
    print(goal_distance_reward, speed_reward, distance_reward, collision_reward, lane_intersection_reward, reward)
    # Update state variables
    if np.absolute(lane_intersection_reward) > 0:
        current["offlane_steps"] += 1
    if current["speed"] == 0:
        current["static_steps"] += 1
    return reward

def _compute_reward_simple(prev, current, verbose=False):
    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    dist_to_trajectory_reward = -1 * np.abs(current['dist_to_trajectory'])
    
    speed_reward = current["speed"]
    acceleration_reward = (current["speed"] - prev["speed"])
    
    # Collision damage
    if((current["num_collisions"] - prev["num_collisions"]) > 0):
        collision_reward = -1
    else:
        collision_reward = 0
    current["collision_reward"] = collision_reward

    # New sidewalk intersection
    if((current["num_laneintersections"] - prev["num_laneintersections"]) > 0):
        lane_intersection_reward = -1
    else:
        lane_intersection_reward = 0
    current["lane_intersection_reward"] = lane_intersection_reward

    reward = dist_to_trajectory_reward + speed_reward
    
    if verbose:
        print("dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, lane_intersection_reward, reward")
        print(dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, lane_intersection_reward, reward)
    
    # Update state variables
    if np.absolute(lane_intersection_reward) > 0:
        current["offlane_steps"] += 1
    if current["speed"] == 0:
        current["static_steps"] += 1
    return reward

def _compute_reward_simple2(prev, current, config=None, verbose=False):
    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    steer = np.abs(current['control_steer'])
    steer_reward = - config["steer_penalty_coeff"] * steer

    current["steer_reward"] = steer_reward

    if verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    dist_to_trajectory_reward = -1 * np.abs(current['dist_to_trajectory'])
    current["dist_to_trajectory_reward"] = dist_to_trajectory_reward
    speed_reward = current["speed"]
    acceleration_reward = (current["speed"] - prev["speed"])
    
    current["speed_reward"] = speed_reward

    light_reward = 0
    current["runover_light"] = False
    if (not config['disable_traffic_light']):
        if (_check_if_signal_crossed(prev, current)
            and (prev['nearest_traffic_actor_state'] == carla.TrafficLightState.Red)
            and (current["speed"] > config["zero_speed_threshold"])
            and (prev['initial_dist_to_red_light'] > config['min_dist_from_red_light'])):
            current["runover_light"] = True
            light_reward = -1 * (config["const_light_penalty"] + config["light_penalty_speed_coeff"] * current["speed"])
        else:
            current["runover_light"] = False
            light_reward = 0
    current["light_reward"] = light_reward

    is_collision = False
    lane_change = False
    obs_collision = (current["num_collisions"] - prev["num_collisions"]) > 0
    is_collision = obs_collision

    # count out_of_road also as a collision
    if config["enable_lane_invasion_sensor"]:
        is_collision = obs_collision or current["out_of_road"]
        
        # count any lane change also as a collision
        if config["enable_lane_invasion_collision"]:
            lane_change = current['num_laneintersections'] > 0
            is_collision = is_collision or lane_change

    current['obs_collision'] = obs_collision
    current['lane_change'] = lane_change
    current["is_collision"] = is_collision
    
    # Collision damage
    if(is_collision):
        # Using prev_speed in collision reward computation
        # due to non-determinism in speed at the time of collision
        collision_reward = -1 * (config["const_collision_penalty"] + config["collision_penalty_speed_coeff"] * prev["speed"])
        speed_reward = prev["speed"]
        # old collision reward
        # collision_reward = -1 * (config["const_collision_penalty"] + config["collision_penalty_speed_coeff"] * current["speed"])
        
    else:
        collision_reward = 0
    current["collision_reward"] = collision_reward

    # # New sidewalk intersection
    # if((current["num_laneintersections"] - prev["num_laneintersections"]) > 0):
    #     lane_intersection_reward = -1
    # else:
    #     lane_intersection_reward = 0
    # current["lane_intersection_reward"] = lane_intersection_reward

    reward = dist_to_trajectory_reward + speed_reward + steer_reward + collision_reward + light_reward

    # Adding constant positive reward to make dist_to_trajectory_reward positive
    reward += config["constant_positive_reward"]

    # clipping reward
    if config["clip_reward"]:
        if reward > 0:
            clipped_reward = 1
        elif reward < 0:
            clipped_reward = -1
    else:
        clipped_reward = reward

    # normalize reward
    clipped_reward = clipped_reward / config["reward_normalize_factor"]

    # success reward
    success = current["distance_to_goal"] < config["dist_for_success"]
    if success:
        clipped_reward += config["success_reward"]

    current["step_reward"] = clipped_reward

    if verbose:
        print("dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, light_reward, steer_reward, reward, clipped_reward")
        print(dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, light_reward, steer_reward, reward, clipped_reward)
    # Update state variables
    # if np.absolute(lane_intersection_reward) > 0:
    #     current["offlane_steps"] += 1
    if current["speed"] <= config["zero_speed_threshold"]:
        current["static_steps"] += 1
    return clipped_reward

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