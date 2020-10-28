
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Lab06_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo/gym_gazebo/envs/ros_ws/src/enph353_lab06/launch/lab06_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        max_row = 20
        slice_bgr = cv_image[-max_row:-1, :]
        slice_gray = cv2.cvtColor(slice_bgr, cv2.COLOR_BGR2GRAY)
        width = slice_gray.shape[1]
        height = slice_gray.shape[0]
        num_intervals = 10
        # Convert to Binary image
        threshold = 150
        _, img_bin = cv2.threshold(slice_gray, threshold, 255, cv2.THRESH_BINARY)
        # average rows
        averaged_rows = np.zeros(width)
        for j in range(width):
            for i in range(height):
                averaged_rows[j] = averaged_rows[j] + img_bin[i, j]
            averaged_rows[j] = averaged_rows[j] / height

        # print("slice gray: {}".format(slice_gray[0]))
        # cv2.imshow("Image window", img_bin)
        # cv2.waitKey(0)

        # Find index of center of road by setting two thresholds
        off_road = 180
        on_road = 70

        left_edge = None
        right_edge = None

        # left edge of frame is already black because left edge of raod
        # not included in frame
        if averaged_rows[0] < on_road:
            left_edge = 0
        if averaged_rows[-1] < on_road:
            right_edge = width

        for j in range(1, len(averaged_rows)-1):
            if averaged_rows[j] < on_road and averaged_rows[j-1] > on_road:
                left_edge = j
            if averaged_rows[j] > off_road and averaged_rows[j-1] < off_road:
                right_edge = j

        if left_edge is not None and right_edge is not None:
            center_idx = int((right_edge + left_edge) / 2)
            region = int(round(center_idx/width*(num_intervals-1)))
            state[region] = 1
            self.timeout = 0
        else:
            # line not detected
            self.timeout += 1

        if self.timeout > 30:
            done = True
        # print("left edge: {}".format(left_edge))
        # print("right edge: {}".format(right_edge))
        # print("self.timeout{}".format(self.timeout))

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        # actually take an action
        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.05   # added a small forward lin velocity
            vel_cmd.angular.z = 0.2
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.05 # added a small forward lin velocity
            vel_cmd.angular.z = -0.2

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action (the action you just took!)
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
