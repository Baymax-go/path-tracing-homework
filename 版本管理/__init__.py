import sys
import os
import numpy as np
from math import atan2, sin, cos, pi, tan
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/Control/")

from DR20API.sim import *

from Control.LQR_Kinematic_Model import *

class Controller:
    def __init__(self, port = 19997):
        """
        Initialize the controller of DR20 robot, and connect and start the simulation in CoppeliaSim.

        Arguments:
        port -- The port used to connect to coppeliaSim, default 19997.
        """
        self.current_map = np.zeros((120,120),dtype="uint8")
        self.port = port
        self.client = self.connect_simulation(self.port)
        # Get handles        #  得到机器人仿真环境中 所有处理对象
        _ , self.robot = sim.simxGetObjectHandle(self.client,"dr20",sim.simx_opmode_blocking)   # 阻塞模式
        _ , self.sensor = sim.simxGetObjectHandle(self.client, "Hokuyo_URG_04LX_UG01", sim.simx_opmode_blocking)    # 得到雷达
        self.handle_left_wheel = sim.simxGetObjectHandle(self.client, "dr20_leftWheelJoint_", sim.simx_opmode_blocking)  # 得到左轮
        self.handle_right_wheel = sim.simxGetObjectHandle(self.client, "dr20_rightWheelJoint_", sim.simx_opmode_blocking)   # 得到右轮

        # Get data from Lidar         sim.simx_opmode_oneshot  非阻塞模式
        sim.simxAddStatusbarMessage(self.client, "python_remote_connected\n", sim.simx_opmode_oneshot)    # 	Adds a message to the status bar. 将消息 添加到 状态栏
        _ , data = sim.simxGetStringSignal(self.client, 'UG01_distance', sim.simx_opmode_streaming)     # 雷达数据
        _ , left_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_left_wheel[1], -1, sim.simx_opmode_streaming)  # 指示相对于我们想要的位置的参考框架。指定 -1 以检索绝对位置，
        _ , right_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_right_wheel[1], -1, sim.simx_opmode_streaming)
        _, pos = sim.simxGetObjectPosition(self.client, self.robot, -1, sim.simx_opmode_streaming)
        _, orientation = sim.simxGetObjectOrientation(self.client, self.robot, -1, sim.simx_opmode_streaming)   # 对象的方向  欧拉角  eulerAngles : 欧拉角（alpha、beta 和 gamma）
        _, sensor_pos = sim.simxGetObjectPosition(self.client, self.sensor, -1, sim.simx_opmode_streaming)
        _, sensor_orientation = sim.simxGetObjectOrientation(self.client, self.sensor, -1, sim.simx_opmode_streaming)
        _, data = sim.simxGetStringSignal(self.client, 'UG01_distance', sim.simx_opmode_streaming)
        sim.simxSynchronousTrigger(self.client)   # 向服务器发送同步触发信号。
        # In CoppeliaSim, you should use simx_opmode_streaming mode to get data first time,
        # and then use simx_opmode_blocking mode
        _, pos = sim.simxGetObjectPosition(self.client, self.robot, -1, sim.simx_opmode_blocking)
        _, orientation = sim.simxGetObjectOrientation(self.client, self.robot, -1, sim.simx_opmode_buffer)
        _, left_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_left_wheel[1], -1,  sim.simx_opmode_blocking)
        _, right_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_right_wheel[1], -1,   sim.simx_opmode_blocking)
        self.vehl = np.linalg.norm(np.array(left_wheel_pos)-np.array(right_wheel_pos)) #  np.linalg.norm  求向量、矩阵 范数   默认求取 2 范数
        _, sensor_pos = sim.simxGetObjectPosition(self.client, self.sensor, -1, sim.simx_opmode_blocking)
        _, sensor_orientation = sim.simxGetObjectOrientation(self.client, self.sensor, -1, sim.simx_opmode_buffer)
        _, data = sim.simxGetStringSignal(self.client, 'UG01_distance', sim.simx_opmode_buffer)
        sim.simxSynchronousTrigger(self.client)
        data = sim.simxUnpackFloats(data)       # 将字符串解压到浮点数组中。   解算 雷达数据
        self.robot_pos = pos[0:-1]

    def connect_simulation(self, port):
        """
        Connect and start simulation.

        Arguments:
        port -- The port used to connect to CoppeliaSim, default 19997.

        Return:
        clientID -- Client ID to communicate with CoppeliaSim.
        """
        clientID = sim.simxStart("127.0.0.1", port, True, True, 5000, 5)
        sim.simxSynchronous(clientID,True)     # 启动远程服务器服务 同步操作模式
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)      # 阻塞模式启动

        if clientID < 0:      # simStart  返回 -1  表示连接失败
            print("Connection failed.")
            exit()

        else:
            print("Connection success.")

        return clientID

    def stop_simulation(self):
        """
        Stop the simulation.
        """
        sim.simxStopSimulation(self.client, sim.simx_opmode_blocking)
        time.sleep(0.5)
        print("Stop the simulation.")
#  lidar  数据处理  需要我研究
    def update_map(self):
        """
        Update the map based on the current information of laser scanner. The obstacles are inflated to avoid collision.
                                                                                                障碍物充气以避免碰撞。
        Return:
        current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
        """
        _, pos = sim.simxGetObjectPosition(self.client, self.sensor, -1, sim.simx_opmode_blocking)
        _, orientation = sim.simxGetObjectOrientation(self.client, self.sensor, -1, sim.simx_opmode_buffer)
        _, data = sim.simxGetStringSignal(self.client, 'UG01_distance', sim.simx_opmode_buffer)
        sim.simxSynchronousTrigger(self.client)
        data = sim.simxUnpackFloats(data)

        scale = 10.
        AtoR = 1.0 / 180.0 * pi
        pixel_x, pixel_y = 0, 0
        for i in range(1,685):
            absolute_angle = AtoR * ((i - 1) * 240 / (684 - 1) + (-120)) + orientation[2]
            lidar_pose_x = pos[0]
            lidar_pose_y = pos[1]
            # print(data)
            # print(i)
            if abs(data[i*3 -2]) > 1:
                obstacle_x = lidar_pose_x + data[i*3 - 2] * cos(absolute_angle - 0 * AtoR)
                obstacle_y = lidar_pose_y + data[i*3 - 2] * cos(absolute_angle - 90 * AtoR)
                pixel_x=round(obstacle_x*scale)
                pixel_y=round(obstacle_y*scale)

            if pixel_x > 3 and pixel_x <= 117 and pixel_y > 3 and pixel_y <= 116:
                for i in range(-2,3):
                    for j in range(-2,3):
                        self.current_map[pixel_x + i][pixel_y + j] = 1

        current_map = self.current_map
        return current_map


    def move_robot(self, path):
        """
        Given planned path of the robot,        给定机器人的计划路径，
        control the robot track a part of path, with a maximum of 3 meters from the start position.
                                                    控制机器人跟踪路径的一部分，距离起始位置最多3米。
        Arguments:
        path -- A N*2 array indicating the planned path.
        """
        k1 = 1.5
        k2 = 0
        v = 6

        pre_error = 0
        path = np.array(path)/10
        for i in range(1,len(path)):
            if np.linalg.norm(path[i] - path[0]) >= 3 and np.linalg.norm(path[i-1] - path[0]) <= 3:
                path = path[0:i]
                break

        final_target = np.array(path[-1])

        _, pos = sim.simxGetObjectPosition(self.client, self.robot, -1, sim.simx_opmode_blocking)
        pos = pos[0:-1]

        _, orientation = sim.simxGetObjectOrientation(self.client, self.robot, -1, sim.simx_opmode_buffer)

        for i in range(1,len(path)):
            target = path[i]

            while np.linalg.norm(np.array(target) - np.array(pos)) > 0.1:
                move = target - np.array(pos)
                theta = orientation[2]
                theta_goal = atan2(move[1], move[0])
                theta_error = theta - theta_goal

                if theta_error < -pi:
                    theta_error += 2 * pi
                elif theta_error > pi:
                    theta_error -= 2 * pi

                u = -(k1 * theta_error + k2 * (pre_error - theta_error))
                pre_error = theta_error

                if abs(theta_error) < 0.1:
                    v_r = v + u
                    v_l = v - u
                elif abs(theta_error) > 0.1:
                    v_r = u
                    v_l = -u

                sim.simxSetJointTargetVelocity(self.client, self.handle_left_wheel[1], v_l,
                                               sim.simx_opmode_streaming)
                sim.simxSetJointTargetVelocity(self.client, self.handle_right_wheel[1], v_r,
                                               sim.simx_opmode_streaming)
                sim.simxSynchronousTrigger(self.client)

                _, pos = sim.simxGetObjectPosition(self.client, self.robot, -1, sim.simx_opmode_blocking)
                pos = pos[0:-1]
                self.robot_pos = pos
                _, orientation = sim.simxGetObjectOrientation(self.client, self.robot, -1, sim.simx_opmode_buffer)

    def get_robot_pos(self):
        """
        Get current position of the robot.

        Return:
        robot_pos -- A 2D vector indicating the coordinate of robot's current position in the grid map.
        """
        _, pos = sim.simxGetObjectPosition(self.client, self.robot, -1, sim.simx_opmode_blocking)
        self.robot_pos = pos[0:-1]
        robot_pos = np.array(self.robot_pos)
        robot_pos = (robot_pos * 10).astype(np.int16)      # 转换成 int16
        return robot_pos

    def get_robot_ori(self):
        """
        Get current orientation of the robot.

        Return:
        robot_ori -- A float number indicating current orientation of the robot in radian.
        """
        _, orientation = sim.simxGetObjectOrientation(self.client, self.robot, -1, sim.simx_opmode_buffer)
        self.robot_ori = orientation[2]
        robot_ori = self.robot_ori
        return robot_ori