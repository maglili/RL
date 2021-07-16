import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


# def mapping_0_1(z):
#     if z == 0.0:
#         return 1.0
#     else:
#         return 0.0


# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# test = cv2.imread("pngegg.png") / 255.0
# test = np.float32(test)
# start = time.time()
# test = np.vectorize(mapping_0_1)(test)  # ok, but slow
# print(time.time() - start)
# plt.imshow(test)
# plt.show()
# quit()


class ChopperScape(Env):
    def __init__(self):
        super().__init__()

        # Define a 2-D observation space
        self.observation_shape = (600, 800, 3)  # h,w,c
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape, dtype=np.float16),
            high=np.ones(self.observation_shape, dtype=np.float16),
            dtype=np.float16,
        )

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(
            6,
        )

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Maximum fuel chopper can take at once
        self.max_fuel = 1000

        # Permissible area of helicper to be
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the heliopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x : x + elem_shape[0]] = elem.icon

        text = "Fuel Left: {} | Return(total rewards): {}".format(
            self.fuel_left, self.ep_return
        )
        text2 = " | Birds: {} Aliens: {} Fuels: {}".format(
            self.bird_count, self.aliens_count, self.fuel_count
        )
        text = text + text2

        # Put the info on canvas
        self.canvas = cv2.putText(
            self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        )

    def reset(self):
        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return = 0

        # Number of birds
        self.bird_count = 0
        self.fuel_count = 0
        self.aliens_count = 0

        # Determine a place to intialise the chopper in
        x = random.randrange(
            int(self.observation_shape[0] * 0.05),
            int(self.observation_shape[0] * 0.10),
        )
        y = random.randrange(
            int(self.observation_shape[1] * 0.25),
            int(self.observation_shape[1] * 0.45),
        )

        # Intialise the chopper
        self.chopper = Chopper(
            "chopper", self.x_max, self.x_min, self.y_max, self.y_min
        )
        self.chopper.set_position(x, y)

        # Intialise the elements
        self.elements = [self.chopper]

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return self.canvas

    def render(self, mode="human"):
        assert mode in [
            "human",
            "rgb_array",
        ], 'Invalid mode, must be either "human" or "rgb_array"'
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}

    def has_collided(self, elem1, elem2):
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step.
        reward = 1

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(0, 5)
        elif action == 1:
            self.chopper.move(0, -5)
        elif action == 2:
            self.chopper.move(5, 0)
        elif action == 3:
            self.chopper.move(-5, 0)
        elif action == 4:
            self.chopper.move(0, 0)

        # Spawn a bird at the right edge with prob 0.01
        if (random.random() < 0.04) and (self.bird_count <= 8):

            # Spawn a bird
            spawned_bird = Bird(
                "bird_{}".format(self.bird_count),
                self.x_max,
                self.x_min,
                self.y_max,
                self.y_min,
            )
            self.bird_count += 1

            # Compute the x,y co-ordinates of the position from where the bird has to be spawned
            # Horizontally, the position is on the right edge and vertically, the height is randomly
            # sampled from the set of permissible values
            bird_x = self.x_max
            bird_y = random.randrange(self.y_min, self.y_max)
            spawned_bird.set_position(self.x_max, bird_y)

            # Append the spawned bird to the elements currently present in Env.
            self.elements.append(spawned_bird)

        # Spawn a fuel at the bottom edge with prob 0.01
        if (random.random() < 0.005) and (self.fuel_count <= 2):
            # Spawn a fuel tank
            spawned_fuel = Fuel(
                "fuel_{}".format(self.bird_count),
                self.x_max,
                self.x_min,
                self.y_max,
                self.y_min,
            )
            self.fuel_count += 1

            # Compute the x,y co-ordinates of the position from where the fuel tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            fuel_x = random.randrange(self.x_min, self.x_max)
            fuel_y = self.y_max
            spawned_fuel.set_position(fuel_x, fuel_y)

            # Append the spawned fuel tank to the elemetns currently present in the Env.
            self.elements.append(spawned_fuel)

        # ======================================================================
        # Spawn a alien at the bottom edge with prob 0.005
        if (random.random() < 0.02) and (self.aliens_count <= 8):
            # Spawn a fuel tank
            spawned_alien = Aliens(
                "aliens_{}".format(self.aliens_count),
                self.x_max,
                self.x_min,
                self.y_max,
                self.y_min,
            )
            self.aliens_count += 1

            # Compute the x,y co-ordinates of the position from where the fuel tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            alien_x = random.randrange(self.x_min, self.x_max)
            alien_y = self.y_min
            spawned_alien.set_position(alien_x, alien_y)

            # Append the spawned fuel tank to the elemetns currently present in the Env.
            self.elements.append(spawned_alien)
        # ======================================================================

        # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Bird):
                # If the bird has reached the left edge, remove it from the Env
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                    self.bird_count -= 1
                else:
                    # Move the bird left by 5 pts.
                    elem.move(-5, 0)

                # If the bird has collided.
                if self.has_collided(self.chopper, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    reward = -10
                    self.elements.remove(self.chopper)

            if isinstance(elem, Fuel):
                # If the fuel tank has reached the top, remove it from the Env
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                    self.fuel_count -= 1
                else:
                    # Move the Tank up by 5 pts.
                    elem.move(0, -5)

                # If the fuel tank has collided with the chopper.
                if self.has_collided(self.chopper, elem):
                    # Remove the fuel tank from the env.
                    self.elements.remove(elem)
                    self.fuel_count -= 1

                    # Fill the fuel tank of the chopper to full.
                    self.fuel_left = self.max_fuel

            # ==================================================================
            if isinstance(elem, Aliens):
                # If the fuel tank has reached the top, remove it from the Env
                if elem.get_position()[1] + elem.icon_h >= self.y_max:
                    self.elements.remove(elem)
                    self.aliens_count -= 1
                else:
                    # Move the Tank up by 5 pts.
                    elem.move(0, 1)

                # If the fuel tank has collided with the chopper.
                if self.has_collided(self.chopper, elem):
                    done = True
                    reward = -10
                    self.elements.remove(self.chopper)
            # ==================================================================

        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # If out of fuel, end the episode.
        if self.fuel_left == 0:
            done = True

        return self.canvas, reward, done, []


class Point:
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Chopper(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("pics\helicopter.png") / 255.0
        # test = cv2.imread("./pics/pngegg.png") / 255.0
        # test = np.float32(test)
        # self.icon = np.vectorize(mapping_0_1)(test)  # ok, but slow
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Bird(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("./pics/grouse-bird-flying-silhouette.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Fuel(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("./pics/SeekPng.com_key-png-image_3862137.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Aliens(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("./pics/android-logo.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


if __name__ == "__main__":

    env = ChopperScape()
    obs = env.reset()

    episode_return = 0

    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_return += reward

        # Render the game
        env.render()
        time.sleep(0.01)

        if done == True:
            print("======Episode Finish=====")
            print("Episode Return:", episode_return)
            print("=" * 25)
            break

    env.close()
# plt.imshow(obs)
# plt.show()
