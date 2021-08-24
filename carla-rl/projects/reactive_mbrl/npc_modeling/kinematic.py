import numpy as np
import torch
from shapely.geometry import LineString, Point, Polygon
import projects.reactive_mbrl.geometry.transformation as transform
import math
import carla


class Kinematic:

    def __init__(self, model, route):
        self.model = model 
        self.route = route[:, :2]
        
    def predict(self, actors, ego):
        bounding_boxes = []
        for actor in actors:
            actor_distance = self.distance_from_route(actor)
            if actor_distance > 5 or self.non_interacting(actor, ego):
                bounding_boxes.append(self.predict_accelerate(actor))
                continue
            if self.is_yielding(actor, ego):
                # print("Modifying actor prediction")
                bounding_boxes.append(self.predict_decelerate(actor))
            else:
                bounding_boxes.append(self.predict_accelerate(actor))
        return bounding_boxes

    def predict_accelerate(self, actor):
        return self.predict_with_acceleration(actor, 1.0)

    def predict_decelerate(self, actor):
        return self.predict_with_acceleration(actor, -1.0)

    def predict_with_acceleration(self, actor, acceleration):
        base_transform = actor.get_transform()
        action = np.array([0.0, acceleration], dtype=np.float32)
        yaw = np.radians(base_transform.rotation.yaw)
        loc = np.array([base_transform.location.x, base_transform.location.y])
        speed = np.linalg.norm(np.array([actor.get_velocity().x, actor.get_velocity().y]))
        pred_loc, pred_yaw, pred_speed =  self.forward(loc, speed, yaw, action)
        pred_loc = pred_loc.detach().numpy()
        pred_yaw = pred_yaw.item()
        pred_transform = carla.Transform(
            location=carla.Location(x=pred_loc[0][0], y=pred_loc[0][1], z=0.0),
            rotation=carla.Rotation(pitch=0.0, yaw=np.degrees(pred_yaw), roll=0.0)
        )
        extent = actor.bounding_box.extent
        bounding_box = get_local_points(extent)
        prediction = transform.transform_points(pred_transform, bounding_box)
        return prediction[:, :2]

    
    def non_interacting(self, actor, ego):
        actor_yaw = actor.get_transform().rotation.yaw
        ego_yaw = ego.get_transform().rotation.yaw
        # print(f"actor_yaw {actor_yaw}, ego_yaw {ego_yaw}")
        diff = abs(actor_yaw - ego_yaw)
        return diff < 10 or abs(diff - math.pi) > 170


    def is_yielding(self, actor, ego):
        actor_prediction = self.predict_actor(actor)
        ego_prediction = self.predict_actor(ego)
        conflict = compute_conflict_region(actor_prediction, ego_prediction)
        if not conflict:
            return False
        actor_decel_required_to_stop = compute_decel_required_to_stop(actor, conflict)
        ego_decel_required_to_stop = compute_decel_required_to_stop(ego, conflict)
        # print(f"Actor decel required to stop {actor_decel_required_to_stop}")
        # print(f"Ego decel required to stop {ego_decel_required_to_stop}")
        return actor_decel_required_to_stop > ego_decel_required_to_stop

    
    def distance_from_route(self, actor):
        actor_loc = Point([actor.get_transform().location.x, actor.get_transform().location.y])
        path = LineString(self.route.tolist())
        return actor_loc.distance(path)
   

    def predict_actor(self, actor, steps=30):
        predictions = []
        
        base_transform = actor.get_transform()
        yaw = np.radians(base_transform.rotation.yaw)
        loc = np.array([base_transform.location.x, base_transform.location.y])
        speed = np.linalg.norm(np.array([actor.get_velocity().x, actor.get_velocity().y]))
        action = actor.get_control()
        action = np.array([action.steer, action.throttle], dtype=np.float32)

        for _ in range(steps):
            loc, yaw, _ = self.forward(loc, speed, yaw, action)
            loc = loc.detach().numpy()
            yaw = yaw.detach().item()
            predictions.append(loc[0])
        
        return np.array(predictions)

    def forward(self, loc, speed, yaw, actions):
        num_acts = 1
        locs = np.tile(loc, (num_acts, 1))
        yaws = np.array([yaw], dtype=np.float32)
        yaws = np.tile(yaws, (num_acts, 1))
        speeds = np.array([speed], dtype=np.float32)
        speeds = np.tile(speeds, (num_acts, 1))
        
        locs = torch.tensor(locs)
        yaws = torch.tensor(yaws)
        speeds = torch.tensor(speeds)
        actions = torch.tensor(actions)
        
        pred_locs, pred_yaws, pred_speeds = self.model.forward(
            locs, yaws, speeds, actions
        )

        return pred_locs, pred_yaws, pred_speeds

def compute_decel_required_to_stop(actor, conflict):
    yaw = actor.get_transform().rotation.yaw
    extent = actor.bounding_box.extent
    bounding_box = get_local_points(extent)
    actor_global = transform.transform_points(actor.get_transform(), bounding_box)
    actor_global = actor_global[:, :2]
    actor_poly = Polygon(actor_global)
    return actor_poly.distance(conflict)

def compute_conflict_region(actor_prediction, ego_prediction):
    actor_ls = LineString(actor_prediction.tolist())
    ego_ls = LineString(ego_prediction.tolist())
    intersection = actor_ls.intersection(ego_ls)
    return intersection
 
def get_local_points(extent):
    return np.array(
        [
            [-extent.x, extent.y, 0, 1],
            [extent.x, extent.y, 0, 1],
            [extent.x, -extent.y, 0, 1],
            [-extent.x, -extent.y, 0, 1],
        ]
    )

def normalize(angle):
    while angle > 180:
        angle -= 2 * 180
    while angle < -180:
        angle += 2 * 180
    return angle 

