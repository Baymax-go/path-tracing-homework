import sys
import os
import numpy as np
from math import atan2, sin, cos, pi, tan
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/Control/")

from DR20API.sim import *
from Control.reeds_shepp import *
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
        _ , self.robot = sim.simxGetObjectHandle(self.client,"Car",sim.simx_opmode_blocking)   # 阻塞模式
        _ , self.sensor = sim.simxGetObjectHandle(self.client, "Hokuyo_URG_04LX_UG01", sim.simx_opmode_blocking)    # 得到雷达

        # print("self.sensor:",self.sensor)      # self.sensor: 110

        self.handle_steer_left_wheel = sim.simxGetObjectHandle(self.client, "front_left", sim.simx_opmode_blocking)  # 得到左轮
        self.handle_steer_right_wheel = sim.simxGetObjectHandle(self.client, "front_right", sim.simx_opmode_blocking)  # 得到左轮

        self.handle_rear_left_wheel = sim.simxGetObjectHandle(self.client, "rear_left_Joint_", sim.simx_opmode_blocking)  # 得到左轮
        self.handle_rear_right_wheel = sim.simxGetObjectHandle(self.client, "rear_right_Joint_", sim.simx_opmode_blocking)   # 得到右轮

        # Get data from Lidar         sim.simx_opmode_oneshot  非阻塞模式
        sim.simxAddStatusbarMessage(self.client, "python_remote_connected\n", sim.simx_opmode_oneshot)    # 	Adds a message to the status bar. 将消息 添加到 状态栏
        _ , data = sim.simxGetStringSignal(self.client, 'UG01_distance', sim.simx_opmode_streaming)     # 雷达数据
        _ , left_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_rear_left_wheel[1], -1, sim.simx_opmode_streaming)  # 指示相对于我们想要的位置的参考框架。指定 -1 以检索绝对位置，
        _ , right_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_rear_right_wheel[1], -1, sim.simx_opmode_streaming)
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
        _, left_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_rear_left_wheel[1], -1,  sim.simx_opmode_blocking)
        _, right_wheel_pos = sim.simxGetObjectPosition(self.client, self.handle_rear_right_wheel[1], -1,   sim.simx_opmode_blocking)

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

        # print("sensor_pos:",pos)   #sensor_pos: [1.5755136013031006, 1.6047954559326172, 0.20199944078922272]

        _, orientation = sim.simxGetObjectOrientation(self.client, self.sensor, -1, sim.simx_opmode_buffer)

        # print("sensor_orientation:", orientation)   # sensor_orientation: [-0.0006501877796836197, -6.684593972750008e-05, 1.658248782157898]

        _, data = sim.simxGetStringSignal(self.client, 'UG01_distance', sim.simx_opmode_buffer)
        sim.simxSynchronousTrigger(self.client)
        data = sim.simxUnpackFloats(data)

        # print("sensor_data:", data)

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

                sim.simxSetJointTargetVelocity(self.client, self.handle_rear_left_wheel[1], v_l,
                                               sim.simx_opmode_streaming)
                sim.simxSetJointTargetVelocity(self.client, self.handle_rear_right_wheel[1], v_r,
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

                                                #     path = Path(x, y, yaw, direc, cost)

                                              #   感觉 需要 return  车的 状态
    def LQR_move_robot(self, vehicle_state_, path):
        path_ = list()
        for i in range(1,len(path.x)):
            path_.append([path.x[i],path.y[i]])

        # print(path_)
        path_ = np.array(path_)
        for i in range(1,len(path_)):
            if np.linalg.norm((path_[i] - path_[0])) >= 30 and np.linalg.norm((path_[i-1] - path_[0])) <= 30:
                path_ = path_[0:i]
                break

        lat_controller = LatController()  # 基于LQR的横向控制器
        lon_controller = LonController()  # 纵向控制器采用PID
                                                                             # 定义了一个 车辆状态  对象            初始速度 不能为 0
        vehicle_state = vehicle_state_

            # VehicleState(x=current_pos[0], y=current_pos[1], yaw=current_ori, v=0.1, gear=path.direction[0])          # ????????????????????????????????????

        #  这里需要 计算一下 参考轨迹 的曲率 数组
        k, _ = calc_curvature(path.x, path.y, path.yaw, path.direction)
        # print("曲率:",k)



        ref_trajectory = TrajectoryAnalyzer(path.x[0:(len(path_)-1)], path.y[0:(len(path_)-1)], path.yaw[0:(len(path_)-1)], k[0:(len(path_)-1)])  # return theta_e, e_cg, yaw_ref, k_ref:曲率

        target = path_[-1]
        # for i in range(1, len(path_)):
        while True:
            dist = math.hypot(vehicle_state.x - target[0], vehicle_state.y - target[1])    # 车辆 距离 终点 的距离
            print("dist:   ", dist)
            if dist <= 1.5:
                break
            print(vehicle_state.gear)


            if vehicle_state.gear == Gear.GEAR_DRIVE:          # ？？？？？？？？？？？？？？
                target_speed = 10.0
            else:
                target_speed = 10.0

            delta_opt, theta_e, e_cg, dir_index = \
                lat_controller.ComputeControlCommand(vehicle_state, ref_trajectory)  # return: steering angle (optimal u), theta_e, e_cg
            print("feedback:",delta_opt, theta_e, e_cg)

            a_opt = lon_controller.ComputeControlCommand(target_speed, vehicle_state, dist)    # return: control command (acceleration) [m / s^2]

            vehicle_state.UpdateVehicleState(delta_opt, a_opt, e_cg, theta_e, vehicle_state.gear)  # 更新车辆 状态

            # print("vehicle_state.v: ",vehicle_state.v)
            # print("a_opt:",a_opt)
            # print("vehicle_state.gear:",vehicle_state.gear)
            # print("vehicle_state.steer:",vehicle_state.steer)

            # if vehicle_state.gear == Gear.GEAR_REVERSE:
            #     vehicle_state.steer = - vehicle_state.steer

            sim.simxSetJointTargetPosition(self.client, self.handle_steer_left_wheel[1], (vehicle_state.steer), sim.simx_opmode_streaming)
            sim.simxSetJointTargetPosition(self.client, self.handle_steer_right_wheel[1], (vehicle_state.steer), sim.simx_opmode_streaming)

            sim.simxSetJointTargetVelocity(self.client, self.handle_rear_left_wheel[1], np.deg2rad(vehicle_state.v * 144), sim.simx_opmode_streaming)
            sim.simxSetJointTargetVelocity(self.client, self.handle_rear_right_wheel[1], np.deg2rad(vehicle_state.v * 144), sim.simx_opmode_streaming)

            sim.simxSynchronousTrigger(self.client)

                                                        #  updata_ vehicle_state
            # print("path.direction[dir_index]:",path.direction[dir_index])
            # print("path.dir: ",path.direction)
            # print("dir_index:",dir_index)
            if path.direction[dir_index] == 1:
                vehicle_state.gear = Gear.GEAR_DRIVE
            else:
                vehicle_state.gear = Gear.GEAR_REVERSE

            xy = Controller.get_robot_pos(self)
            vehicle_state.x = xy[0]
            vehicle_state.y = xy[1]
            vehicle_state.yaw = Controller.get_robot_ori(self) + np.pi  # 补偿

        # print("finish")

        return vehicle_state



    def move_init(self):

        sim.simxSetJointTargetVelocity(self.client, self.handle_steer_left_wheel[1], 0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.client, self.handle_steer_right_wheel[1], 0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(self.client, self.handle_steer_left_wheel[1], 0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(self.client, self.handle_steer_right_wheel[1], 0, sim.simx_opmode_streaming)

        sim.simxSetJointTargetVelocity(self.client, self.handle_rear_left_wheel[1], 0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.client, self.handle_rear_right_wheel[1], 0, sim.simx_opmode_streaming)
        sim.simxSynchronousTrigger(self.client)




            #  方向盘 位置 靠 PID 控制   只能输入 误差值
    def test_move(self):

        current_pos_ = Controller.get_robot_pos(self)

        print("current_pos_: ",current_pos_)




        for i in range(1,50):
            sim.simxSetJointTargetPosition(self.client, self.handle_steer_left_wheel[1], (np.deg2rad(0)),
                                           sim.simx_opmode_streaming)
            sim.simxSetJointTargetPosition(self.client, self.handle_steer_right_wheel[1], (np.deg2rad(0)),
                                           sim.simx_opmode_streaming)

            sim.simxSetJointTargetVelocity(self.client, self.handle_rear_left_wheel[1], np.deg2rad(720),
                                           sim.simx_opmode_streaming)
            sim.simxSetJointTargetVelocity(self.client, self.handle_rear_right_wheel[1], np.deg2rad(720),
                                           sim.simx_opmode_streaming)
            sim.simxSynchronousTrigger(self.client)


        current_pos_end = Controller.get_robot_pos(self)
        print("current_pos_end: ", current_pos_end)