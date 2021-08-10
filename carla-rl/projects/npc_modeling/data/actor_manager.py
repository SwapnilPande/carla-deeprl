import numpy as np
import carla


class ActorTrack:
    def __init__(self, id):
        self.transforms = []
        self.velocities = []
        self.accelerations = []
        self.controls = []
        self.is_alive = True
        self.id = id

    def add_transform(self, transform):
        self.transforms.append(transform)

    def add_velocity(self, vel):
        self.velocities.append([vel.x, vel.y, vel.z])

    def add_acceleration(self, accel):
        self.accelerations.append([accel.x, accel.y, accel.z])

    def add_control(self, control):
        self.controls.append([control.steer, control.throttle, control.brake])


class ActorManager:
    def __init__(self):
        self.actors = {}
        self.timestamps = []

    def step(self, actor_list, camera, info):
        valid_actors = set()
        vehicles = [actor for actor in actor_list if "vehicle" in actor.type_id]
        bounding_boxes = [get_bounding_box(vehicle, camera) for vehicle in vehicles]

        for (actor, bb) in zip(vehicles, bounding_boxes):
            if all(bb[:, 2] > 0):
                self.add_actor_transform(actor.id, actor.get_transform().get_matrix())
                self.add_actor_velocity(actor.id, actor.get_velocity())
                self.add_actor_acceleration(actor.id, actor.get_acceleration())
                self.add_actor_control(actor.id, actor.get_control())
                import pdb

                pdb.set_trace()
                self.add_actor_topdown(actor.id, info["sensor.camera.rgb/top"], bb)
                valid_actors.add(actor.id)

        dead_actors = [self.actors[id] for id in self.actors if id not in valid_actors]
        assert len(dead_actors) == 0

    def is_valid(self, actor):
        return "vehicle" in actor.type_id

    def add_actor_transform(self, id, transform):
        if id not in self.actors:
            self.actors[id] = ActorTrack(id)
        self.actors[id].add_transform(transform)

    def add_actor_velocity(self, id, vel):
        if id not in self.actors:
            self.actors[id] = ActorTrack(id)
        self.actors[id].add_velocity(vel)

    def add_actor_acceleration(self, id, accel):
        if id not in self.actors:
            self.actors[id] = ActorTrack(id)
        self.actors[id].add_acceleration(accel)

    def add_actor_control(self, id, control):
        if id not in self.actors:
            self.actors[id] = ActorTrack(id)
        self.actors[id].add_control(control)

    def add_actor_topdown(self, id, topdown, bb):
        if id not in self.actors:
            self.actors[id] = ActorTrack(id)
        self.actors[id].add_topdown(topdown)


def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def vehicle_to_world(cords, vehicle):

    bb_transform = carla.Transform(vehicle.bounding_box.location)
    # bb_vehicle_matrix = bb_transform.get_matrix()
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords


def world_to_sensor(cords, sensor):
    # sensor_world_matrix = sensor.transform.get_matrix()
    sensor_world_matrix = get_matrix(sensor.transform)
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords


def vehicle_to_sensor(cords, vehicle, sensor):
    world_cord = vehicle_to_world(cords, vehicle)
    sensor_cord = world_to_sensor(world_cord, sensor)
    return sensor_cord


def create_bb_points(vehicle):
    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords


def get_bounding_box(vehicle, camera):
    bb_cords = create_bb_points(vehicle)
    cords_x_y_z = vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
    cords_y_minus_z_x = np.concatenate(
        [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
    )
    bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
    camera_bbox = np.concatenate(
        [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1
    )
    return camera_bbox
