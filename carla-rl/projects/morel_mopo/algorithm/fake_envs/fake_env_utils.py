import torch
import numpy as np

class NormalizedTensor:
    def __init__(self, mean, std, device):
        self.device = device

        if(not isinstance(mean, torch.Tensor)):
            if(not isinstance(mean, (np.ndarray, list))):
                raise Exception("Mean must be torch tensor, list, or numpy array")

            mean = torch.FloatTensor(mean)

        if(not isinstance(std, torch.Tensor)):
            if(not isinstance(std, (np.ndarray, list))):
                raise Exception("Std must be torch tensor, list, or numpy array")

            std = torch.FloatTensor(std)

        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

        self.dim = self.mean.shape[-1]
        if(self.dim != self.std.shape[-1]):
            raise Exception("Mean and Std dimensions are different")

        self._unnormalized_array = None

    @property
    def unnormalized(self):
        return self._unnormalized_array

    @unnormalized.setter
    def unnormalized(self, unnormalized_arr):
        if(not isinstance(unnormalized_arr, torch.Tensor)):
            if(not isinstance(unnormalized_arr, (np.ndarray, list))):
                raise Exception("Input must be torch tensor, list, or numpy array")

            unnormalized_arr = torch.FloatTensor(unnormalized_arr)

        if(self.dim != unnormalized_arr.shape[-1]):
            raise Exception("Dimension of input tensor ({}) does not match dimension of mean ({})".format(unnormalized_arr.shape[-1],
                                                                                                            self.dim))

        unnormalized_arr = unnormalized_arr.to(self.device)

        self._unnormalized_array = unnormalized_arr

    @property
    def normalized(self):
        if(self._unnormalized_array is None):
            return None

        return self.normalize_array(self._unnormalized_array)

    @normalized.setter
    def normalized(self, normalized_arr):
        if(not isinstance(normalized_arr, torch.Tensor)):
            if(not isinstance(normalized_arr, (np.ndarray, list))):
                raise Exception("Input must be torch tensor, list, or numpy array")

            normalized_arr = torch.FloatTensor(normalized_arr)

        if(self.dim != normalized_arr.shape[-1]):
            raise Exception("Dimension of input tensor ({}) does not match dimension of mean ({})".format(
                normalized_arr.shape[-1],
                self.dim))

        normalized_arr = normalized_arr.to(self.device)

        self._unnormalized_array = self.unnormalize_array(normalized_arr)

    def normalize_array(self, array):
        return (array - self.mean)/self.std

    def unnormalize_array(self, array):
        return self.std * array + self.mean


    def __str__(self):
        return str(self._unnormalized_array)


def distance_vehicle(waypoint, vehicle_pose, device):
    '''
    Calculates L2 distance between waypoint, vehicle
    @param waypoint:     torch.Tensor([x,y])
            vehicle_pose: torch.Tensor([x, y, yaw])
    '''
    if not torch.is_tensor(waypoint):
        waypoint = torch.FloatTensor(waypoint).to(device)
    vehicle_loc = vehicle_pose[:2].to(device) # get x, y
    dist = torch.dist(waypoint, vehicle_loc).item()
    return dist



def get_dot_product_and_angle(vehicle_pose, waypoint, device):
    '''
    Calculates dot product, angle between vehicle, waypoint
    @param waypoint:     [x, y]
        vehicle_pose: torch.Tensor([x, y, yaw])
    '''

    waypoint = torch.FloatTensor(waypoint).to(device)

    v_begin         = vehicle_pose[:2]
    vehicle_yaw     = vehicle_pose[2]
    v_end = v_begin + torch.Tensor([torch.cos(torch.deg2rad(vehicle_yaw)), \
                                    torch.sin(torch.deg2rad(vehicle_yaw))]).to(device)

    # vehicle vector: direction vehicle is pointing in global coordinates
    v_vec = torch.sub(v_end, v_begin)
    # vector from vehicle's position to next waypoint
    w_vec = torch.sub(waypoint, v_begin)
    # steering error: angle between vehicle vector and vector pointing from vehicle loc to
    # waypoint
    dot   = torch.dot(w_vec, v_vec)
    angle = torch.acos(torch.clip(dot /
                                (torch.linalg.norm(w_vec) * torch.linalg.norm(v_vec)), -1.0, 1.0))

    try:
        assert(torch.isclose(torch.cos(angle), torch.clip(torch.dot(w_vec, v_vec) / \
            (torch.linalg.norm(w_vec) * torch.linalg.norm(v_vec)), -1.0, 1.0), atol=1e-3))
    except:
        import ipdb; ipdb.set_trace()

    # make vectors 3D for cross product
    v_vec_3d = torch.hstack((v_vec, torch.Tensor([0]).to(device)))
    w_vec_3d = torch.hstack((w_vec, torch.Tensor([0]).to(device)))

    _cross = torch.cross(v_vec_3d, w_vec_3d)

    # if negative steer, turn left
    if _cross[2] < 0:
        angle *= -1.0

    # assert cross product a x b = |a||b|sin(angle)
    # assert(torch.isclose(_cross[2], torch.norm(v_vec_3d) * torch.norm(w_vec_3d) * torch.sin(angle), atol=1e-2))

    return dot, angle, w_vec



def vehicle_to_line_distance(vehicle_pose, waypoint1, waypoint2, device):
    '''
    Gets distance of vehicle to a line formed by two waypoints
    @param waypoint1:     [x,y]
        waypoint2:     [x,y]
        vehicle_pose:  torch.Tensor([x,y, yaw])
    '''

    waypoint1 = torch.FloatTensor(waypoint1).to(device)
    waypoint2 = torch.FloatTensor(waypoint2).to(device)

    if torch.allclose(waypoint1, waypoint2):
        # distance = distance_vehicle(waypoint1, vehicle_pose, device)
        # return abs(distance)
        raise Exception(f"FAKE_ENV: Waypoints are too close together: {waypoint1}, {waypoint2}")


    vehicle_loc   = vehicle_pose[:2] # x, y coords

    a_vec = torch.sub(waypoint2, waypoint1)   # forms line between two waypoints
    b_vec = torch.sub(vehicle_loc, waypoint1) # line from vehicle to first waypoint

    # make 3d for cross product
    a_vec_3d = torch.hstack((a_vec, torch.Tensor([0]).to(device)))
    b_vec_3d = torch.hstack((b_vec, torch.Tensor([0]).to(device)))

    dist_vec = torch.cross(a_vec_3d, b_vec_3d) / torch.linalg.norm(a_vec_3d)
    distance =  dist_vec[2] # dist

    return distance

def compute_trajectory_velocity(next_waypoints, vehicle_pose, vehicle_speed, device, second_last_waypoint = None, last_waypoint = None, previous_waypoint = None):
    ''' Calculates the velocity of the vehicle in the trajectory axis

    '''
    trajectory_velocity = vehicle_speed

    waypoint1 = None
    waypoint2 = None
    if len(next_waypoints) > 1:
        if(previous_waypoint is not None):
            # get dist from vehicle to a line formed by the next two wps
            waypoint1 = previous_waypoint
            waypoint2 = next_waypoints[0]

        else:
            waypoint1 = next_waypoints[0]
            waypoint2 = next_waypoints[1]

    # if only one next waypoint, use it and second to last
    elif len(next_waypoints) == 1:
        if second_last_waypoint:
            waypoint1 = second_last_waypoint
            waypoint2 = next_waypoints[0]

        else:
            print("CODE BROKE HERE UH OH _----------------------")

    else: # Run out of wps
        if second_last_waypoint and last_waypoint:
            waypoint1 = second_last_waypoint
            waypoint2 = last_waypoint

        else:
            print("CODE BROKE HERE UH OH _----------------------")


    if(waypoint1 is not None and waypoint2 is not None):
        waypoint1 = torch.FloatTensor(waypoint1).to(device)
        waypoint2 = torch.FloatTensor(waypoint2).to(device)


        if torch.allclose(waypoint1, waypoint2):
            # distance = distance_vehicle(waypoint1, vehicle_pose, device)
            # return abs(distance)
            print("WARNING: Waypoints are too close together: {waypoint1}, {waypoint2}")

            return vehicle_speed

        vehicle_yaw   = vehicle_pose[2] # x, y coords
        # Get unit vecotr of trajectory
        a_vec = torch.sub(waypoint2, waypoint1)/ torch.norm(torch.sub(waypoint2, waypoint1))

        # Vehicle velocity vector (unit)
        v_vec = trajectory_velocity * torch.Tensor([torch.cos(torch.deg2rad(vehicle_yaw)), \
                                        torch.sin(torch.deg2rad(vehicle_yaw))]).to(device)

        # Project Speed (in direction of vehicle heading) onto trajectory
        trajectory_velocity = torch.dot(v_vec, a_vec)
    return trajectory_velocity

def filter_waypoints(waypoints):
    """This function will remove duplicate waypoints from the waypoint list

    Sometimes, the waypoints in the dataset contain duplicate waypoints. We need to remove these duplicates to prevent errors in processing waypoints.
    """

    waypoints = waypoints.tolist()
    popped = False
    i = 1

    while(i < len(waypoints)):
        if(torch.allclose(torch.FloatTensor(waypoints[i-1]), torch.FloatTensor(waypoints[i]))):
            waypoints.pop(i)
            popped = True
        else:
            i += 1
    # if(popped):
    #     print("FAKE_ENV: POPPED DUPLICATE WAYPOINTS")

    return torch.FloatTensor(waypoints)


def check_if_waypoint_crossed(vehicle_pose, waypoint1, waypoint2, device):
    '''
    Checks if vehicle crossed a waypoint
    @param waypoint:     [x,y]
        vehicle_pose: torch.Tensor([x, y, yaw])
    '''
    wp_vector = torch.tensor(waypoint2[0:2]).to(device) - torch.tensor(waypoint1[0:2]).to(device)

    vehicle_vector = vehicle_pose[:2] - torch.tensor(waypoint1[0:2]).to(device)

    # Check if dot product is positive
    return torch.dot(wp_vector, vehicle_vector) > 0




def process_waypoints(waypoints, vehicle_pose, device, second_last_waypoint = None, last_waypoint = None, previous_waypoint = None):
    '''
    Calculate dist to trajectory, angle
    @param waypoints:    [torch.Tensor([wp1_x,wp1_y]), torch.Tensor([wp2_x,wp2_y)......]
        vehicle pose: torch.Tensor([x,y,yaw])
    @returns dist_to_trajectory, angle, ...
    '''

    vehicle_pose.to(device)

    waypoints = waypoints[:, :2]
    waypoints = waypoints.tolist()

    next_waypoints_angles = []
    next_waypoints_vectors = []
    next_waypoints = []
    num_next_waypoints = 5
    last_waypoint, second_last_waypoint = None, None

    # closest wp to car
    min_dist_index = -1

    # Minimum distance to waypoint before we delete it
    # This number is taken from GlobalPlanner
    MIN_REMOVE_DISTANCE = 1.8

    for i, waypoint in enumerate(waypoints[:20]):
        # find wp that yields min dist between itself and car
        dist_i = distance_vehicle(waypoint, vehicle_pose, device)
        # print(f'wp {i},  {waypoint}, dist: {dist_i}')
        if dist_i <= MIN_REMOVE_DISTANCE:
            passed = False
            if len(waypoints) - i > 1:
                # get dist from vehicle to a line formed by the next two wps
                passed = check_if_waypoint_crossed(
                                        vehicle_pose,
                                        waypoint,
                                        waypoints[i+1],
                                        device)
            if passed:
                min_dist_index = i

    wp_len = len(waypoints)
    if(wp_len < 2 and second_last_waypoint is None):
        print("JUST AS I HAD SUSPECTED")
    if min_dist_index >= 0:
        # pop waypoints up until the one with min distance to vehicle
        for i in range(min_dist_index + 1):
            waypoint = waypoints.pop(0)
            # set last, second-to-last waypoints
            if i == wp_len - 1:
                last_waypoint = waypoint
            elif i == wp_len - 2:
                second_last_waypoint = waypoint

            if(i == min_dist_index):
                previous_waypoint = waypoint

    remaining_waypoints = waypoints
    # only keep next N waypoints
    for i, waypoint in enumerate(waypoints[:num_next_waypoints]):
        # dist to waypoint
        dot, angle, w_vec = get_dot_product_and_angle(vehicle_pose, waypoint, device)

        # add back waypoints
        next_waypoints_angles.append(angle)
        next_waypoints.append(waypoint)
        next_waypoints_vectors.append(w_vec)

    # get mean of all angles to figure out which direction to turn
    if len(next_waypoints_angles) > 0:
        angle = torch.mean(torch.FloatTensor(next_waypoints_angles))
    else:
        # print("No next waypoint found!")
        angle = 0


    if len(next_waypoints) > 1:
        if(previous_waypoint is not None):
            # get dist from vehicle to a line formed by the next two wps
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    previous_waypoint,
                                    next_waypoints[0],
                                    device)
        else:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    next_waypoints[0],
                                    next_waypoints[1],
                                    device)

    # if only one next waypoint, use it and second to last
    elif len(next_waypoints) == 1:
        if second_last_waypoint:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    second_last_waypoint,
                                    next_waypoints[0],
                                    device)
        else:
            print("CODE BROKE HERE UH OH _----------------------")
            dist_to_trajectory = 0.0

    else: # Run out of wps
        if second_last_waypoint and last_waypoint:
            dist_to_trajectory = vehicle_to_line_distance(
                                    vehicle_pose,
                                    second_last_waypoint,
                                    last_waypoint,
                                    device)
        else:
            print("CODE BROKE HERE UH OH _----------------------")
            dist_to_trajectory = 0.0

    return angle, dist_to_trajectory, next_waypoints, next_waypoints_angles, next_waypoints_vectors, remaining_waypoints, second_last_waypoint, last_waypoint, previous_waypoint



def rot(theta):
    # New implementation that can take a list of thetas
    # Returns a list of rot matrices
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    row1 = torch.stack([cos_theta, -sin_theta], dim=-1)
    row2 = torch.stack([sin_theta,  cos_theta], dim=-1)

    R = torch.stack([row1, row2], dim=-2)

    return R


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.
    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    # Get vector to obstacle
    target_vector = target_transform[0:2] - current_transform[0:2]
    # Get magnitude of vector
    norm_target = torch.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True, norm_target

    # Regardless of direction, obstacle is too far
    if norm_target > max_distance:
        return False, norm_target

    forward_vector = torch.tensor([torch.cos(torch.deg2rad(current_transform[2])), torch.sin(torch.deg2rad(current_transform[2]))]).to(current_transform.device)
    dot = torch.dot(forward_vector, target_vector) / norm_target

    return (dot > 0, norm_target)


def check_if_vehicle_in_same_lane(target_vehicle, next_waypoints, dist_to_trajec_threshold, device):
    # Get the dist to trajectory for the target vehicle
    _, \
    dist_to_trajectory, \
    _,\
    _, _, \
    _, \
    _, \
    _, _ = process_waypoints(next_waypoints,
                                                    target_vehicle,
                                                    device)

    if(not isinstance(dist_to_trajectory, torch.Tensor)):
        dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(device)

    # if(torch.abs(dist_to_trajectory) < dist_to_trajec_threshold):
    #     print(f"DIST TO TRAJECTORY {dist_to_trajectory}")


    # If the target vehicle is within dist_to_trajectory_threshold of the trajectory, we can assume that the target vehicle is in the same lane
    return torch.abs(dist_to_trajectory) < dist_to_trajec_threshold



