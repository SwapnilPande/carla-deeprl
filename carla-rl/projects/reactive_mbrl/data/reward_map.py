import numpy as np
import carla
import math

from shapely.geometry import Point, LineString, Polygon
import projects.reactive_mbrl.geometry.transformation as transform

MAP_SIZE = 64
MIN_REWARD = -1
TARGET_SPEED = 5


def initialize_empty_map(n):
    return np.zeros((n, n))


def calculate_world_grid_points_old(env):
    pixel_x, pixel_y = np.meshgrid(
        np.arange(-MAP_SIZE / 2, MAP_SIZE / 2), np.arange(-MAP_SIZE / 2, MAP_SIZE / 2)
    )
    pixel_xy = np.stack(
        [
            pixel_x.flatten(),
            pixel_y.flatten(),
            np.zeros(MAP_SIZE * MAP_SIZE),
            np.ones(MAP_SIZE * MAP_SIZE),
        ],
        axis=-1,
    )
    ego_transform = env.carla_interface.get_ego_vehicle()._vehicle.get_transform()
    return transform.transform_points(ego_transform, pixel_xy)


def calculate_world_grid_points(env):
    calibration = np.array(
        [[MAP_SIZE / 2, 0, MAP_SIZE / 2], [0, MAP_SIZE / 2, MAP_SIZE / 2], [0, 0, 1]]
    )

    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors[
        "sensor.camera.rgb/map"
    ].sensor
    base_transform = ego_actor.get_transform()

    pixel_x, pixel_y = np.meshgrid(np.arange(MAP_SIZE), np.arange(MAP_SIZE))
    pixel_xy = np.stack(
        [pixel_x.flatten(), pixel_y.flatten(), np.ones(MAP_SIZE * MAP_SIZE)], axis=-1
    )
    world_pts = np.linalg.inv(calibration).dot(pixel_xy.T).T[:, :2]
    world_pts *= 5

    world_pts = transform.transform_points(
        base_transform, transform.points_to_homogeneous(world_pts)
    )[:, :2]
    # yaw = -(((np.radians(base_transform.rotation.yaw) + np.pi) % (2 * np.pi)) - np.pi)
    # rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    # world_pts = world_pts.dot(rot_matrix)

    # world_pts *= camera_actor.get_transform().location.z
    # world_pts *= 10
    # world_pts[:, 0] += camera_actor.get_transform().location.x
    # world_pts[:, 1] += camera_actor.get_transform().location.y - 5
    return world_pts, pixel_xy


def calculate_path_following_reward(world_pts, route):
    # waypoints = env.carla_interface.next_waypoints
    path = LineString(route.tolist())

    def distance_to_path_reward(pt):
        d = Point([pt[0], pt[1]]).distance(path)
        return -d

    return np.array([distance_to_path_reward(pt) for pt in world_pts])


def waypoint_to_numpy(waypoint):
    return [
        waypoint.transform.location.x,
        waypoint.transform.location.y,
        waypoint.transform.location.z,
    ]


def calculate_reward_map(env, route):
    world_pts, pixel_xy = calculate_world_grid_points(env)

    positions, labels = calculate_lane_violations_labels(env, world_pts)
    labels = calculate_vehicle_collisions(env, positions, labels).astype(float)
    # labels += calculate_path_following_reward(world_pts, route)
    labels = np.clip(labels, MIN_REWARD, 0.0)

    reward_map = np.zeros((MAP_SIZE, MAP_SIZE))
    reward_map[pixel_xy[:, 0].astype(int), pixel_xy[:, 1].astype(int)] = labels
    # reward_map = reward_map[::-1]

    return reward_map, world_pts, pixel_xy


def calculate_action_value_map(locs, yaws, speeds, ref_wpt, target_speed=TARGET_SPEED):

    rewards = []
    loc_losses = []
    yaw_losses = []
    speed_losses = []
    for (loc, yaw, speed) in zip(locs, yaws, speeds):
        loc_loss, yaw_loss, speed_loss = calculate_action_value(loc, yaw[0], speed[0], ref_wpt, target_speed)
        rewards.append(loc_loss + yaw_loss + speed_loss)
        loc_losses.append(loc_loss)
        yaw_losses.append(yaw_loss)
        speed_losses.append(speed_loss)
    rewards = np.array(rewards)
    loc_losses = np.array(loc_losses)
    yaw_losses = np.array(yaw_losses)
    speed_losses = np.array(speed_losses)

    return loc_losses, yaw_losses, speed_losses, rewards

def calculate_action_value(loc, yaw, speed, ref_wpt, target_speed=TARGET_SPEED):
    wpt_loc, wpt_yaw = ref_wpt
    loc_loss = np.linalg.norm(wpt_loc - loc)
    yaw_loss = np.abs(wpt_yaw - yaw)
    speed_loss = np.abs(target_speed - speed)
    return loc_loss, yaw_loss, speed_loss


def get_closest_waypoint(loc, route):
    min_dist = 10000
    min_wpt = None
    for (wpt_x, wpt_y, wpt_yaw) in route:
        d = np.linalg.norm(loc - np.array([wpt_x, wpt_y]))
        if d < min_dist:
            min_dist = d
            min_wpt = (np.array([wpt_x, wpt_y]), wpt_yaw)
    return min_wpt



def calculate_4d_reward_map(env, route, speed):
    world_pts, pixel_xy = calculate_world_grid_points(env)
    num_pts, _ = world_pts.shape
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    base_transform = ego_actor.get_transform()
    ego_pos = np.array([base_transform.location.x, base_transform.location.y])
    import pdb

    pdb.set_trace()


def calculate_vehicle_collisions(env, positions, labels):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    base_transform = ego_actor.get_transform()
    actors = [
        actor
        for actor in env.carla_interface.actor_fleet.actor_list
        if "vehicle" in actor.type_id
        and actor.get_transform().location.distance(base_transform.location) < 15
        and actor != ego_actor
    ]

    if len(actors) <= 0:
        return labels

    bounding_boxes = np.array(
        [create_bbox(actor.bounding_box.extent) for actor in actors]
    )
    vehicles = np.array([extract_loc(actor) for actor in actors])
    num_vehicles = len(vehicles)

    for i in range(len(actors)):
        yaw = actors[i].get_transform().rotation.yaw
        bounding_boxes[i] = rotate_points(bounding_boxes[i], yaw)

    vehicles = bounding_boxes + vehicles[:, None, :]
    points = [Point(positions[i, 0], positions[i, 1]) for i in range(len(positions))]
    mask = np.zeros(len(labels))

    for i in range(len(actors)):
        poly = Polygon([(vehicles[i, j, 0], vehicles[i, j, 1]) for j in range(4)])
        in_poly = np.array([point.within(poly) for point in points])
        mask = np.logical_or(mask, in_poly)

    labels[mask] = MIN_REWARD
    return labels


def calculate_lane_violations_labels(env, world_pts):
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    base_transform = ego_actor.get_transform()
    base_waypoint = env.carla_interface.map.get_waypoint(
        base_transform.location, project_to_road=True
    )
    positions = []
    labels = []
    for i, pt in enumerate(world_pts):
        x_loc, y_loc = pt[0], pt[1]
        positions.append((x_loc, y_loc))
        location = carla.Location(x=x_loc, y=y_loc, z=base_transform.location.z)
        waypoint = env.carla_interface.map.get_waypoint(location, project_to_road=False)

        # check if off-road
        if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
            labels.append(MIN_REWARD)
            continue

        # check if lane violation
        if not waypoint.is_junction:
            base_yaw = base_waypoint.transform.rotation.yaw
            yaw = waypoint.transform.rotation.yaw
            waypoint_angle = (((base_yaw - yaw) + 180) % 360) - 180

            if np.abs(waypoint_angle) > 150:
                labels.append(MIN_REWARD)
                continue

        labels.append(0)

    return np.array(positions), np.array(labels)


def rotate_points(points, angle):
    radian = angle * math.pi / 180
    return points @ np.array(
        [[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]]
    )


def extract_loc(actor):
    return (actor.get_transform().location.x, actor.get_transform().location.y)


def create_bbox(extent):
    return [
        (extent.x, extent.y),
        (extent.x, -extent.y),
        (-extent.x, -extent.y),
        (-extent.x, extent.y),
    ]
