import environment.carla_interfaces.controller as controller
import numpy as np
import os
from shapely.geometry import Polygon, Point
import projects.reactive_mbrl.geometry.transformation as transform
import math
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import carla


class PIDAgent:

    def __init__(self, npc_predictor, output_path=None):
        self.npc_predictor = npc_predictor
        # self.output_path = output_path
        self.index = 0

    def reset(self, waypoints):
        self.index = 0
        # self.scenario_path = os.path.join(self.output_path, scenario_name)
        self.previous_action = None

        # if not os.path.isdir(self.scenario_path):
        #     os.makedirs(self.scenario_path)

    def predict(self, env, info, current_speed, target_speed):
        self.index += 1
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        waypoint = env.carla_interface.next_waypoints[0]
        throt_controller = controller.PIDLongitudinalController()
        steer_controller = controller.PIDLateralController(ego_actor)

        steer = steer_controller.pid_control(waypoint)
        steer = np.clip(steer, -1, 1)

        ego_polygon, predicted_obs, (light, light_dist) = self.predict_obstacles(env)
        d = self.calculate_distance_to_obstacle(env, ego_polygon, predicted_obs)
        desired_target_speed = calculate_desired_target_speed(target_speed, d)
        throt = throt_controller.pid_control(desired_target_speed, current_speed, enable_brake=True)

        should_stop_for_light = light is not None and light.state == carla.TrafficLightState.Red and light_dist <= 5.0
        
        if (d <= 3.0 or should_stop_for_light) and current_speed > 0:
            throt = -1.0

        action = np.array([steer, throt])
        # self.save_data(env, info, ego_polygon, predicted_obs)

        # print(f"action is {action}")
        return action

    def save_data(self, env, info, ego_poly, vehicles):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        ego_transform = ego_actor.get_transform()

        ax = axes[0]
        ego_coords = np.array(list(ego_poly.exterior.coords))
        ax.plot(ego_coords[:, 0], ego_coords[:, 1], color='red')
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        ego_loc = ego_actor.get_transform().location
        for vehicle in vehicles:
            ax.plot(vehicle[:, 0], vehicle[:, 1], color='blue')
        actors = [actor for actor in env.carla_interface.actor_fleet.actor_list
                if "vehicle" in actor.type_id 
                and actor != ego_actor
                and actor.get_transform().location.distance(ego_transform.location) < 20
        ]
        for actor in actors:
            plot_actor(ax, actor)
        ax.set_xlim(ego_loc.x - 20, ego_loc.x + 20)
        ax.set_ylim(ego_loc.y - 20, ego_loc.y + 20)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.annotate(f"num_vehicles = {len(vehicles)}", xy=(0, 0), fontsize=20, xycoords='figure points')
        if len(vehicles) > 0:
            ax.annotate(f"{vehicles[0]}", xy=(0, 1), fontsize=20, xycoords='figure points')
        ax.annotate(f"{ego_coords}", xy=(0, 50), fontsize=20, xycoords='figure points')
        # ax.axis('equal')

        ax = axes[1]
        topdown = info["sensor.camera.rgb/top"]
        ax.imshow(topdown)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        plt.savefig(os.path.join(self.scenario_path, f"reward_{self.index}.png"))
        plt.clf()
        plt.close()

    def predict_obstacles(self, env):
        lights = env.carla_interface.world.get_actors().filter('*traffic_light*')
        light, light_dist, _ = env.carla_interface.actor_fleet.ego_vehicle.find_nearest_traffic_light(lights)

        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        base_transform = ego_actor.get_transform()
        ego_yaw = base_transform.rotation.yaw
        ego_bb = create_bbox(ego_actor.bounding_box.extent)
        ego_bb = rotate_points(ego_bb, ego_yaw)
        ego_bb += extract_loc(ego_actor)
        ego_polygon = Polygon(ego_bb)
        actors = [actor for actor in env.carla_interface.actor_fleet.actor_list
                if "vehicle" in actor.type_id 
                and actor != ego_actor
                and actor.get_transform().location.distance(base_transform.location) < 20
        ]

        if len(actors) <= 0:
            return ego_polygon, [], (light, light_dist)

        vehicles = self.npc_predictor.predict(actors, ego_actor)

        bounding_boxes = np.array(
            [create_bbox(actor.bounding_box.extent) for actor in actors]
        )
        current_vehicles = np.array([extract_loc(actor) for actor in actors])
        
        for i in range(len(actors)):
            yaw = actors[i].get_transform().rotation.yaw
            bounding_boxes[i] = rotate_points(bounding_boxes[i], yaw)
        
        current_vehicles = bounding_boxes + current_vehicles[:, None, :]
        vehicles.extend(current_vehicles)

        return ego_polygon, vehicles, (light, light_dist)


    def calculate_distance_to_obstacle(self, env, ego_polygon, vehicles):
        if len(vehicles) <= 0:
            return 100
        
        polygons = [Polygon(vehicle) for vehicle in vehicles]
        
        waypoints = env.carla_interface.next_waypoints[:20]
        
        for waypoint in waypoints:
            location = Point([waypoint.transform.location.x, waypoint.transform.location.y])
            for poly in polygons:
                if location.within(poly):
                    return ego_polygon.distance(poly)
        
        return 100


def rotate_points(points, angle):
    radian = angle * math.pi / 180
    return points @ np.array(
        [[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]]
    )

def create_bbox(extent):
    return [
        (extent.x, extent.y),
        (extent.x, -extent.y),
        (-extent.x, -extent.y),
        (-extent.x, extent.y),
    ]

def extract_loc(actor):
    return (actor.get_transform().location.x, actor.get_transform().location.y)

def calculate_desired_target_speed(current_target_speed, distance):
    target_speed = update_based_on_distance(current_target_speed, distance)
    return target_speed

def update_based_on_distance(current_target_speed, distance):

    if distance <= 3.0:
        return 0.0

    if distance >= 15.0:
        return current_target_speed

    return current_target_speed * (distance - 3.0) / (15.0 - 3.0)

def plot_actors(ax, env, include_ego=False, ego_id=0):
    for actor in env.carla_interface.actor_fleet.actor_list:
        if "vehicle" not in actor.type_id:
            continue

        plot_actor(ax, actor, color="black")


def plot_actor(ax, actor, color="black"):
    yaw = actor.get_transform().rotation.yaw
    extent = actor.bounding_box.extent
    bounding_box = get_local_points(extent)
    actor_global = transform.transform_points(actor.get_transform(), bounding_box)
    lower_left = actor_global[-1]
    rec = patches.Rectangle(lower_left, 2 * extent.x, 2 * extent.y, angle=yaw, fill=True)
    ax.add_patch(rec)
    ax.plot(actor_global[:, 0], actor_global[:, 1], color=color)

def get_local_points(extent):
    return np.array(
        [
            [-extent.x, extent.y, 0, 1],
            [extent.x, extent.y, 0, 1],
            [extent.x, -extent.y, 0, 1],
            [-extent.x, -extent.y, 0, 1],
        ]
    )

