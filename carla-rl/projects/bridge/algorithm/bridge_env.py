import gym
from gym.spaces import Box
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvas

class LavaBridge(gym.Env):

    def __init__(self):
        # Observation space is [x, y, x_dot, y_dot]
        self.observation_space = Box(low = np.array([-20, -20, -1, 1]),
                                    high = np.array([ 20,  20,  1, 1]))

        # Action space is [F_x, F_y]
        self.action_space = Box(low = np.array([-1, -1]),
                                high = np.array([ 1,  1]))

        # Simulation time step - we assume force is constant during time step
        self.dt = 0.1

        # Mass of object in kg
        self.mass = 0.9

        # Goal position
        self.goal = np.array([0, 10])

        self.lava_penalty = -500

        # For each lava pit, we have
        self.lava_pits = [
            Box(low = np.array([-10, -8]),
                high = np.array([-0.5, 8])),

            Box(low = np.array([0.5, -8]),
                high = np.array([10, 8]))
        ]

        self.enable_lava_walls = False

        # Region where our agent can move, agent cannot leave this area
        self.playable_area = Box(low = np.array([-20, -20]),
                                high = np.array([ 20,  20]))

        # Minimum distance in meters from goal to be considered at_goal
        self.goal_delta = 0.5

        self.timeout_steps = 500

    def construct_obs(self):
        return np.concatenate([self.x, self.x_dot])

    def in_lava(self):
        #TODO add intersection checking to make sure we don't jump over lava

        lava = False
        for lava_pit in self.lava_pits:
            lava = lava_pit.contains(self.x) | lava

        return lava


    def did_intersect(self, start, end, edge):
        return start <= edge < end


    def check_lava_wall_collision(self, new_x, new_x_dot):
        for lava_pit in self.lava_pits:

            # Check if we crossed bottom edge in this step
            if(self.did_intersect(self.x[0], new_x[0], lava_pit.low[0])
                and lava_pit.low[1] <= new_x[1] <= lava_pit.high[1]):
                new_x[0] = lava_pit.low[0]
                new_x_dot[0] = 0

                return new_x, new_x_dot

            # Check if we crossed top edge point in this step
            if(self.did_intersect(self.x[0], new_x[0], lava_pit.high[0])
                and lava_pit.low[1] <= new_x[1] <= lava_pit.high[1]):
                new_x[0] = lava_pit.high[0]
                new_x_dot[0] = 0

                return new_x, new_x_dot

            # Check if we crossed left edge in this step
            if(self.did_intersect(self.x[1], new_x[1], lava_pit.low[1])
                and lava_pit.low[0] <= new_x[0] <= lava_pit.high[0]):
                new_x[1] = lava_pit.low[1]
                new_x_dot[1] = 0

                return new_x, new_x_dot

            # Check if we crossed right edge point in this step
            if(self.did_intersect(self.x[1], new_x[1], lava_pit.high[1])
                and lava_pit.low[0] <= new_x[0] <= lava_pit.high[0]):
                new_x[1] = lava_pit.high[1]
                new_x_dot[1] = 0

                return new_x, new_x_dot

        return new_x, new_x_dot



    def at_goal(self):
        return np.sum((self.x - self.goal)**2) < self.goal_delta

    def get_matplotlib_image(self):
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.subplots()

        ax.scatter(self.x[0], self.x[1], color = "blue")
        ax.scatter(self.goal[0], self.goal[1], color = "green")

        # Draw left rectangle
        for lava_pit in self.lava_pits:
            delta = lava_pit.high - lava_pit.low
            patch = patches.Rectangle(lava_pit.low, delta[0], delta[1], fill = True, color = "red")

            ax.add_patch(patch)

        ax.set_xlim(self.observation_space.low[0], self.observation_space.high[0])
        ax.set_ylim(self.observation_space.low[1], self.observation_space.high[1])

        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.close(fig)

        return img

    def reset(self):
        # Start at the bottom of map, centered w.r.t bridge
        self.x = np.array([0, -10])

        # Start with zero velocity
        self.x_dot = np.array([0, 0])

        self.steps = 0


        return self.construct_obs()


    def step(self, action):
        # Action is [F_x, F_y]

        # Get acceleration
        a = action / self.mass

        new_x_dot = a * self.dt + self.x_dot
        new_x = 0.5 * a * self.dt**2 + self.x_dot * self.dt + self.x

        # Bound checking
        # If we're west of the playable area
        if(new_x[0] < self.playable_area.low[0]):
            new_x[0] = self.playable_area.low[0]
            new_x_dot[0] = 0
        # If we're east of the playable area
        elif(new_x[0] > self.playable_area.high[0]):
            new_x[0] = self.playable_area.high[0]
            new_x_dot[0] = 0

        # If we're south of the playable area
        if(new_x[1] < self.playable_area.low[1]):
            new_x[1] = self.playable_area.low[1]
            new_x_dot[1] = 0
        # If we're right of the playable area
        elif(new_x[1] > self.playable_area.high[1]):
            new_x[1] = self.playable_area.high[1]
            new_x_dot[1] = 0

        # Lava wall bound checking
        if(self.enable_lava_walls):
            new_x, new_x_dot = self.check_lava_wall_collision(new_x, new_x_dot)

        # Update state aftewards
        self.x = new_x
        self.x_dot = new_x_dot

        # Compute done conditions
        lava = self.in_lava() and not self.enable_lava_walls
        goal = self.at_goal()

        self.steps += 1

        done = lava or goal or (self.steps > self.timeout_steps)

        # Reward at each step is the negative squared distance between the current position and the goal
        # Add negative penalty
        reward = -np.sum((self.x - self.goal)**2)/800 + self.lava_penalty * int(lava)

        info = {}

        return self.construct_obs(), reward, done, info

    def render(self):
        return self.get_matplotlib_image()










