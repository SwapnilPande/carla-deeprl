''' Planner '''
import os
import glob
import sys
import math
import numpy as np
from environment.carla_interfaces.agents.navigation.global_route_planner import GlobalRoutePlanner
from environment.carla_interfaces.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from environment.carla_interfaces.agents.tools.misc import distance_vehicle
from collections import deque

# CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
# if CARLA_9_4_PATH == None:
#     raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

# try:
#     sys.path.append(glob.glob(CARLA_9_4_PATH+'/**/*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

try:
    import carla
except Exception as e:
    print("Failed to import Carla")
    raise e

class GlobalPlanner():

    def __init__(self):
        self._grp = None
        self._hop_resolution = 2.0
        self.MIN_DISTANCE_PERCENTAGE = 0.9
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._waypoints_queue_old = deque(maxlen=20000)
        self.dist_to_trajectory = 0
        self.second_last_waypoint = None
        self.last_waypoint = None
        self._min_distance = self._hop_resolution * self.MIN_DISTANCE_PERCENTAGE

    def trace_route(self, map, start_transform, destination_transform):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        start_waypoint, end_waypoint = map.get_waypoint(start_transform.location), map.get_waypoint(destination_transform.location)

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(map, self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def compute_distances_between_waypoints(self, current_plan):
        modified_plan = []
        last_waypoint = current_plan[-1][0]
        dist = 0
        for elem in reversed(current_plan):
            waypoint, unk = elem
            dist += last_waypoint.transform.location.distance(waypoint.transform.location)
            modified_plan.append((waypoint, unk, dist))
            last_waypoint = waypoint
        modified_plan.reverse()
        return modified_plan

    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        prev_wp = None
        modified_plan = self.compute_distances_between_waypoints(current_plan)
        for elem in modified_plan:
            # self.printwaypoint(elem[0])
            if not self.sameWaypoint(elem[0], prev_wp):
                # print("Added wp")
                self._waypoints_queue.append(elem)
                self._waypoints_queue_old.append(elem)
            prev_wp = elem[0]
        self.previous_waypoint = None

    def get_next_orientation(self, vehicle_transform):

        next_waypoints_angles = []
        next_waypoint_found = False
        num_next_waypoints = 5
        max_index = 0
        for i, (waypoint, _) in enumerate(self._waypoints_queue_old):

            dot, angle, _ = self.get_dot_product_and_angle(vehicle_transform, waypoint)

            # next_waypoint_found implies the first waypoint with
            # positive dot product is found
            if not next_waypoint_found:
                if dot >= 0:
                    max_index = i
                    next_waypoint_found = True
                    next_waypoints_angles = [angle]
            else:
                if len(next_waypoints_angles) < num_next_waypoints:
                    next_waypoints_angles.append(angle)
                else:
                    break
        if max_index > 0:
            for i in range(max_index):
                self._waypoints_queue_old.popleft()

        if next_waypoint_found and len(next_waypoints_angles) > 0:
            angle = np.mean(np.array(next_waypoints_angles))
        else:
            print("No next waypoint found!")
            angle = 0

        return angle, 0

    # def get_next_orientation_new(self, vehicle_transform):

    #     next_waypoints_angles = []
    #     next_waypoints = []
    #     next_waypoint_found = False
    #     num_next_waypoints = 5
    #     max_index = 0
    #     min_dist = np.inf
    #     # print("length of waypoints queue {0}".format(len(self._waypoints_queue)))
    #     for i, (waypoint, _) in enumerate(self._waypoints_queue):

    #         dist_to_waypoint = distance_vehicle(waypoint, vehicle_transform)
    #         dot, angle = self.get_dot_product_and_angle(vehicle_transform, waypoint)

    #         # next_waypoint_found implies the first waypoint with
    #         # positive dot product is found
    #         if not next_waypoint_found:
    #             if dist_to_waypoint < min_dist:
    #                 min_dist = dist_to_waypoint
    #                 max_index = i
    #                 next_waypoints_angles = [angle]
    #                 next_waypoints = [waypoint]
    #             else:
    #                 next_waypoint_found = True
    #         else:
    #             if len(next_waypoints_angles) < num_next_waypoints:
    #                 next_waypoints_angles.append(angle)
    #                 next_waypoints.append(waypoint)
    #             else:
    #                 break
    #     if max_index > 0:
    #         q_len = len(self._waypoints_queue)

    #         # Remove all waypoints except the closest one (max_index)
    #         for i in range(max_index):
    #             waypoint, _ = self._waypoints_queue.popleft()

    #             # Store second-last waypoint for corner case
    #             if i == q_len - 2:
    #                 self.second_last_waypoint = waypoint

    #     next_waypoints_angles_array = np.array(next_waypoints_angles)
    #     print("full angle {0}, selected 2nd angle {1}, selected 3rd angle {2}".format(np.mean(next_waypoints_angles_array), np.mean(next_waypoints_angles_array[1:]), np.mean(next_waypoints_angles_array[2:])))
    #     print(next_waypoints_angles)
    #     if len(next_waypoints_angles) > 2:
    #         angle = np.mean(next_waypoints_angles_array[2:])
    #     elif len(next_waypoints_angles) > 1:
    #         angle = np.mean(next_waypoints_angles_array[1:])
    #     elif len(next_waypoints_angles) > 0:
    #         angle = np.mean(next_waypoints_angles_array)
    #     else:
    #         print("No next waypoint found!")
    #         angle = 0

    #     if len(next_waypoints) > 1:
    #         self.dist_to_trajectory = self.getPointToLineDistance(
    #                                 vehicle_transform,
    #                                 next_waypoints[0],
    #                                 next_waypoints[1])
    #     else:
    #         # Reached near last waypoint
    #         # use second_last_waypoint

    #         print("Needed to use second_last_waypoint")
    #         last_wp, _ = self._waypoints_queue[0]
    #         self.dist_to_trajectory = self.getPointToLineDistance(
    #                                 vehicle_transform,
    #                                 self.second_last_waypoint,
    #                                 last_wp)

    #     return angle, self.dist_to_trajectory

    def waypoints_to_list(self):
        wp_list = []
        for waypoint in self._waypoints_queue:
            wp_list.append([waypoint[0].transform.location.x, waypoint[0].transform.location.y, waypoint[0].transform.rotation.yaw])

        return wp_list

    def check_if_waypoint_crossed(self, vehicle_transform, waypoint1, waypoint2):
        point = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        point1_on_line = np.array([waypoint1.transform.location.x, waypoint1.transform.location.y])
        point2_on_line = np.array([waypoint2.transform.location.x, waypoint2.transform.location.y])

        wp_vector = point2_on_line - point1_on_line
        vehicle_vector = point - point1_on_line

        # Check if dot product is positive
        return np.dot(wp_vector, vehicle_vector) > 0

    def get_next_orientation_new(self, vehicle_transform):

        next_waypoints_angles = []
        next_waypoints_vectors = []
        next_waypoints = []
        next_waypoint_found = False
        num_next_waypoints = 5
        num_extended_lookahead_waypoints = 5
        max_index = -1
        min_dist = np.inf
        for i, (waypoint, _, dist) in enumerate(self._waypoints_queue):
            dist_i = distance_vehicle(
                    waypoint, vehicle_transform)
            # print("i:{0}, dist : {1}".format(i, dist_i))
            if(i > 20):
                break

            if dist_i < self._min_distance:
                passed = False
                if len(self._waypoints_queue) - i > 1:
                    # get dist from vehicle to a line formed by the next two wps
                    passed = self.check_if_waypoint_crossed(
                                            vehicle_transform,
                                            waypoint,
                                            self._waypoints_queue[i+1][0])
                if passed:
                    max_index = i

        q_len = len(self._waypoints_queue)
        if max_index >= 0:
            for i in range(max_index + 1):
                waypoint, _, dist= self._waypoints_queue.popleft()

                if i == q_len - 1:
                    self.last_waypoint = waypoint
                elif i == q_len - 2:
                    self.second_last_waypoint = waypoint

                if(i == max_index):
                    self.previous_waypoint = waypoint

        for i, (waypoint, _, dist) in enumerate(self._waypoints_queue):
            if i > num_next_waypoints + num_extended_lookahead_waypoints - 1:
                break
            dist_to_waypoint = distance_vehicle(waypoint, vehicle_transform)
            dot, angle, w_vec = self.get_dot_product_and_angle(vehicle_transform, waypoint)

            if len(next_waypoints_angles) == 0:
                next_waypoints_angles = [angle]
                next_waypoints = [waypoint]
                dist_to_goal = dist
                next_waypoints_vectors = [w_vec]
            else:
                next_waypoints_angles.append(angle)
                next_waypoints.append(waypoint)
                next_waypoints_vectors.append(w_vec)

        # for i, (waypoint, _) in enumerate(self._waypoints_queue):

        #     dist_to_waypoint = distance_vehicle(waypoint, vehicle_transform)
        #     dot, angle = self.get_dot_product_and_angle(vehicle_transform, waypoint)

        #     # next_waypoint_found implies the first waypoint with
        #     # positive dot product is found
        #     if not next_waypoint_found:
        #         if dist_to_waypoint < min_dist:
        #             min_dist = dist_to_waypoint
        #             max_index = i
        #             next_waypoints_angles = [angle]
        #             next_waypoints = [waypoint]
        #         else:
        #             next_waypoint_found = True
        #     else:
        #         if len(next_waypoints_angles) < num_next_waypoints:
        #             next_waypoints_angles.append(angle)
        #             next_waypoints.append(waypoint)
        #         else:
        #             break
        # if max_index > 0:
        #     q_len = len(self._waypoints_queue)

        #     # Remove all waypoints except the closest one (max_index)
        #     for i in range(max_index):
        #         waypoint, _ = self._waypoints_queue.popleft()

        #         # Store second-last waypoint for corner case
        #         if i == q_len - 2:
        #             self.second_last_waypoint = waypoint

        next_waypoints_angles_array = np.array(next_waypoints_angles)
        # if len(next_waypoints_angles) > 2:
        #     angle = np.mean(next_waypoints_angles_array[2:])
        # elif len(next_waypoints_angles) > 1:
        #     angle = np.mean(next_waypoints_angles_array[1:])
        if len(next_waypoints_angles) > 0:
            angle = np.mean(next_waypoints_angles_array[:num_next_waypoints])
        else:
            print("No next waypoint found!")
            dist_to_goal = 0
            angle = 0

        if(len(next_waypoints_angles) > num_next_waypoints):
            angle_extended_lookahead = np.mean(next_waypoints_angles_array[num_next_waypoints:])
        else:
            angle_extended_lookahead = 0


        if len(next_waypoints) > 1:
            if(self.previous_waypoint is not None):
                self.dist_to_trajectory = self.getPointToLineDistance(
                                    vehicle_transform,
                                    self.previous_waypoint,
                                    next_waypoints[0])
            else:
                self.dist_to_trajectory = self.getPointToLineDistance(
                                        vehicle_transform,
                                        next_waypoints[0],
                                        next_waypoints[1])
        elif len(next_waypoints) > 0:
            self.dist_to_trajectory = self.getPointToLineDistance(
                                    vehicle_transform,
                                    self.second_last_waypoint,
                                    next_waypoints[0])

        else:
            # Reached near last waypoint
            # use second_last_waypoint

            print("Needed to use second_last_waypoint")
            if self.second_last_waypoint is not None and self.last_waypoint is not None:
                self.dist_to_trajectory = self.getPointToLineDistance(
                                        vehicle_transform,
                                        self.second_last_waypoint,
                                        self.last_waypoint)
            else:
                self.dist_to_trajectory = 0

        # Below is an approximation of dist_to_goal which was used earlier.
        dist_to_goal_approx = len(self._waypoints_queue) *self._hop_resolution

        return angle, angle_extended_lookahead, self.dist_to_trajectory, dist_to_goal, next_waypoints, next_waypoints_angles, next_waypoints_vectors, self.waypoints_to_list()

    def get_dot_product_and_angle(self, vehicle_transform, waypoint):

        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x -
                          v_begin.x, waypoint.transform.location.y -
                          v_begin.y, 0.0])
        dot = np.dot(w_vec, v_vec)
        angle = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            angle *= -1.0

        # returning dot product, angle, and vector (x,y)
        return dot, angle, w_vec[:2]

    def getPointToLineDistance(self, vehicle_transform, waypoint1, waypoint2):
        point = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        point1_on_line = np.array([waypoint1.transform.location.x, waypoint1.transform.location.y])
        point2_on_line = np.array([waypoint2.transform.location.x, waypoint2.transform.location.y])
        return self.getPointToLineDistanceHelper(point, point1_on_line, point2_on_line)

    def getPointToLineDistanceHelper(self, point, point1_on_line, point2_on_line):
        a_vec = point2_on_line - point1_on_line
        b_vec = point - point1_on_line
        # returning signed distance
        return np.cross(a_vec, b_vec) / np.linalg.norm(a_vec)

    def printwaypoint(self, waypoint):
        print("x:{}, y:{}".format(waypoint.transform.location.x, waypoint.transform.location.y))

    def sameWaypoint(self, waypoint1, waypoint2):

        if waypoint1 is None or waypoint2 is None:
            return True
        x1 = waypoint1.transform.location.x
        y1 = waypoint1.transform.location.y
        x2 = waypoint2.transform.location.x
        y2 = waypoint2.transform.location.y

        return (x1 == x2) and (y1 == y2)