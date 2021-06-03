import traceback
import numpy as np
import cv2
from client_bounding_boxes import ClientSideBoundingBoxes
from environment import CarlaEnv

np.random.seed(1)

env = CarlaEnv()

try:
    env.reset()
    calibration = np.array([[64, 0, 64],
                            [0, 64, 64],
                            [0,  0,  1]])
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/top'].sensor
    camera_actor.calibration = calibration

    for i in range(2000):
        action = env.get_autopilot_action()
        # action = np.array([0,-1])
        _, _, done, _ = env.step(action)
        image = env.render()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for vehicle_actor in env.carla_interface.actor_fleet.actor_list:
            if 'vehicle' not in vehicle_actor.type_id:
                continue

            box = ClientSideBoundingBoxes.get_bounding_box(vehicle_actor, camera_actor)
            box = box.astype(int)
            for i in range(8):
                cv2.circle(image, (box[i,0], box[i,1]), radius=1, color=(0,255,0), thickness=-1)

        cv2.imshow('pov', image)
        cv2.waitKey(1)

        if done:
            env.reset()
            ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
            camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/top'].sensor
            camera_actor.calibration = calibration
except:
    traceback.print_exc()
    # import ipdb; ipdb.set_trace()
finally:
    env.close()
    cv2.destroyAllWindows()