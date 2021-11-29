import numpy as np


def fetch_actor_features(actor):
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

    bounding_box_loc = actor.bounding_box.get_world_vertices(transform)
    bounding_box = [(loc.x, loc.y) for loc in bounding_box_loc]

    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'theta': transform.rotation.yaw,
        'speed': speed,
        'bounding_box': bounding_box
    }


def normalize_actor_features(actor_features, ref, theta):
    """
    Normalize actor feature dictionary to reference point
    ref is a tuple (x, y, theta)
    """
    for i, (x,y) in enumerate(actor_features['bounding_box']):
        x,y = transform_to_pov((x,y), ref, theta)
        actor_features['bounding_box'][i] = (x,y)

    x,y = transform_to_pov((actor_features['x'], actor_features['y']), ref, theta)
    actor_features['x'], actor_features['y'] = x,y
    actor_features['theta'] = normalize_angle(actor_features['theta'] - theta)


def fetch_symbolic_dict(ego_actor, other_actors, episode_measurements):
    # get ego kinematics
    # ego_actor = self.carla_interface.get_ego_vehicle()._vehicle
    ego_features = fetch_actor_features(ego_actor)

    ref = ego_features['x'], ego_features['y']
    theta = ego_features['theta']

    normalize_actor_features(ego_features, ref, theta)

    # get other entities
    # other_actors = carla_interface.world.get_actors().filter('*vehicle*')
    vehicle_features = {actor.id: fetch_actor_features(actor) for actor in other_actors
        if actor.get_transform().location.distance(ego_actor.get_transform().location) < 20
        and actor.id != ego_actor.id
    }

    for vehicle_id in vehicle_features:
        features = vehicle_features[vehicle_id]
        normalize_actor_features(features, ref, theta)

    # normalize waypoints
    waypoints = episode_measurements['waypoints']
    for i, (x,y,_) in enumerate(waypoints):
        x,y = transform_to_pov((x,y), ref, theta)
        waypoints[i] = (x,y)

    features = {
        'ego_features': ego_features,
        'vehicle_features': vehicle_features,

        'light': episode_measurements['traffic_light']['red_light_dist'],

        'next_waypoints': waypoints,
        'next_orientation': episode_measurements['next_orientation'],
        'dist_to_trajectory': episode_measurements['dist_to_trajectory'],

        'obstacle_dist': episode_measurements['obstacles']['obstacle_dist'],
        'obstacle_speed': episode_measurements['obstacles']['obstacle_speed'],

        'x': ref[0],
        'y': ref[1],
        'theta': theta
    }
    return features


def transform_to_pov(src, ref, theta):
    """
    Transforms src to ref frame
    src and ref are tuples (x, y)
    """
    sx, sy = src
    rx, ry = ref

    x = sx - rx
    y = sy - ry

    theta = normalize_angle(theta)
    theta = -theta # because we want to transform to 0 rotation offset
    theta = np.radians(theta) # because np expects radians

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = np.array([[x],[y]])
    xy = rot_matrix.dot(xy).flatten()

    return xy[0], xy[1]


def normalize_angle(theta):
    theta = (theta + 360) % 360
    theta = theta if theta <= 180 else theta-360
    return theta
