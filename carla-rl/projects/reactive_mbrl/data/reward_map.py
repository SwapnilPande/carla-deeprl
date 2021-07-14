import numpy as np
import carla
import math

from shapely.geometry import Point, LineString, Polygon
import projects.reactive_mbrl.geometry.transformation as transform

MAP_SIZE = 16

def initialize_empty_map(n):
    return np.zeros((n, n))


def calculate_world_grid_points_old(env):
    pixel_x, pixel_y = np.meshgrid(np.arange(-MAP_SIZE/2, MAP_SIZE/2), np.arange(-MAP_SIZE/2, MAP_SIZE/2))
    pixel_xy = np.stack(
        [pixel_x.flatten(),
        pixel_y.flatten(),
        np.zeros(MAP_SIZE * MAP_SIZE),
        np.ones(MAP_SIZE * MAP_SIZE)], axis=-1)
    ego_transform = env.carla_interface.get_ego_vehicle()._vehicle.get_transform()
    return transform.transform_points(ego_transform, pixel_xy)

def calculate_world_grid_points(env):
    calibration = np.array([[8, 0, 8],
                            [0, 8, 8],
                            [0,  0,  1]])

    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/map'].sensor
    base_transform = ego_actor.get_transform()

    pixel_x, pixel_y = np.meshgrid(np.arange(16), np.arange(16))
    pixel_xy = np.stack([pixel_x.flatten(), pixel_y.flatten(), np.ones(16*16)], axis=-1)
    world_pts = np.linalg.inv(calibration).dot(pixel_xy.T).T[:,:2]

    # yaw = np.radians(((base_transform.rotation.yaw + 180) % 360) - 180)
    yaw = -(((np.radians(base_transform.rotation.yaw) + np.pi) % (2*np.pi)) - np.pi)
    rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    world_pts = world_pts.dot(rot_matrix)

    world_pts *= camera_actor.get_transform().location.z
    world_pts[:,0] += camera_actor.get_transform().location.x
    world_pts[:,1] += camera_actor.get_transform().location.y
    return world_pts, pixel_xy


def calculate_path_following_reward(env, world_pts):
    waypoints = env.carla_interface.next_waypoints
    path = LineString([waypoint_to_numpy(wpt) for wpt in waypoints])
    def distance_to_path_reward(pt):
        return max(20 - (Point([pt[0], pt[1]]).distance(path)), 0)

    return np.array([distance_to_path_reward(pt) for pt in world_pts])


def waypoint_to_numpy(waypoint):
    return [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z]


def calculate_reward_map(env):
    world_pts, pixel_xy = calculate_world_grid_points(env)

    positions, labels = calculate_lane_violations_labels(env, world_pts)
    labels = calculate_vehicle_collisions(env, positions, labels)

    reward_map = np.zeros((MAP_SIZE, MAP_SIZE))
    reward_map[pixel_xy[:,0].astype(int), pixel_xy[:,1].astype(int)] = labels
    reward_map = reward_map[::-1]

    return reward_map, world_pts

def calculate_vehicle_collisions(env, positions, labels):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    base_transform = ego_actor.get_transform()
    actors = [actor for actor in env.carla_interface.actor_fleet.actor_list
        if 'vehicle' in actor.type_id
        and actor.get_transform().location.distance(base_transform.location) < 15
        and actor != ego_actor]

    if len(actors) <= 0:
        return labels

    bounding_boxes = np.array([create_bbox(actor.bounding_box.extent) for actor in actors])
    vehicles = np.array([extract_loc(actor) for actor in actors])
    num_vehicles = len(vehicles)

    for i in range(len(actors)):
        yaw = actors[i].get_transform().rotation.yaw
        bounding_boxes[i] = rotate_points(bounding_boxes[i], yaw)

    vehicles = bounding_boxes + vehicles[:, None, :]
    points = [Point(positions[i,0], positions[i,1]) for i in range(len(positions))]
    mask = np.zeros(len(labels))

    for i in range(len(actors)):
        poly = Polygon([(vehicles[i,j,0], vehicles[i,j,1]) for j in range(4)])
        in_poly = np.array([point.within(poly) for point in points])
        mask = np.logical_or(mask, in_poly)

    labels[mask] = 3
    return labels


def calculate_lane_violations_labels(env, world_pts):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    base_transform = ego_actor.get_transform()
    base_waypoint = env.carla_interface.map.get_waypoint(base_transform.location, project_to_road=True)
    positions = []
    labels = []
    for i, pt in enumerate(world_pts):
        x_loc, y_loc = pt[0], pt[1]
        positions.append((x_loc, y_loc))
        location = carla.Location(x=x_loc, y=y_loc, z=base_transform.location.z)
        waypoint = env.carla_interface.map.get_waypoint(location, project_to_road=False)

        # check if off-road
        if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
            labels.append(1)
            continue

        # check if lane violation
        if not waypoint.is_junction:
            base_yaw = base_waypoint.transform.rotation.yaw
            yaw = waypoint.transform.rotation.yaw
            waypoint_angle = (((base_yaw - yaw) + 180) % 360) - 180

            if np.abs(waypoint_angle) > 150:
                labels.append(2)
                continue

        labels.append(0)

    return np.array(positions), np.array(labels)

def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])


def extract_loc(actor):
    return (actor.get_transform().location.x, actor.get_transform().location.y)

def create_bbox(extent):
    return [(extent.x, extent.y),
     (extent.x, -extent.y),
     (-extent.x, -extent.y),
     (-extent.x, extent.y)]
