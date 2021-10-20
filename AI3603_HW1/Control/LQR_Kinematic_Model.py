"""
LQR and PID Controller
author: huiming zhou
"""

import os
import sys
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../AI3603_HW1/")

# import Control.draw_lqr as draw
from Control.config_control import *
import Control.reeds_shepp as rs




# Controller Config
ts = 0.05  # [s]
l_f = 5.0  # [m]     #  wheel_base  =  l_f + l_r  = 0.5 (m)
l_r = 5.0  # [m]
max_iteration = 150      #Number of iterations
eps = 0.01           # Upper error limit

matrix_q = [1.5, 0.0, 70.0, 0.0]
matrix_r = [70.0]

state_size = 4

max_acceleration = 5.0  # [m / s^2]
max_steer_angle = np.deg2rad(40)  # [rad]
max_speed = 10.0  # [m / s]


class Gear(Enum):
    GEAR_DRIVE = 1            # forward
    GEAR_REVERSE = 2          # Backward


class VehicleState:          # 车辆状态
    def __init__(self, x=0.0, y=0.0, yaw=0.0,
                 v=0.0, gear=Gear.GEAR_DRIVE):  #  位置，速度，前进or倒退
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.e_cg = 0.0             # 参考轨迹的横向误差
        self.theta_e = 0.0          # 相对于参考轨迹的偏航误差

        self.gear = gear
        self.steer = 0.0            # 方向盘需要转角

    def UpdateVehicleState(self, delta, a, e_cg, theta_e,
                           gear=Gear.GEAR_DRIVE):
        """
        update states of vehicle
        :param theta_e: yaw error to ref trajectory   相对于参考轨迹的偏航误差
        :param e_cg: lateral error to ref trajectory    参考轨迹的横向误差
        :param delta: steering angle [rad]          转向角[rad]
        :param a: acceleration [m / s^2]            加速度
        :param gear: gear mode [GEAR_DRIVE / GEAR/REVERSE]
        """

        wheelbase_ = l_r + l_f
        delta, a = self.RegulateInput(delta, a)    #限制速度、转角大小

        self.gear = gear
        self.steer = delta          # 直接修正航向

        self.e_cg = e_cg
        self.theta_e = theta_e
        # print(gear == Gear.GEAR_DRIVE)
        # print("Gear.GEAR_DRIVE:",Gear.GEAR_DRIVE)


        if gear == Gear.GEAR_DRIVE:
            self.v += a * ts
        else:
            self.v += -1.0 * a * ts

        self.v = self.RegulateOutput(self.v)


    @staticmethod          # #40°   #5  m/s^2
    def RegulateInput(delta, a):
        """
        regulate delta to : - max_steer_angle ~ max_steer_angle
        regulate a to : - max_acceleration ~ max_acceleration
        :param delta: steering angle [rad]
        :param a: acceleration [m / s^2]
        :return: regulated delta and acceleration
        """

        if delta < -1.0 * max_steer_angle:    #45°
            delta = -1.0 * max_steer_angle

        if delta > 1.0 * max_steer_angle:
            delta = 1.0 * max_steer_angle

        if a < -1.0 * max_acceleration:    #5  m/s^2
            a = -1.0 * max_acceleration

        if a > 1.0 * max_acceleration:
            a = 1.0 * max_acceleration

        return delta, a

    @staticmethod           #  10 m/s
    def RegulateOutput(v):
        """
        regulate v to : -max_speed ~ max_speed
        :param v: calculated speed [m / s]
        :return: regulated speed
        """

        max_speed_ = max_speed

        if v < -1.0 * max_speed_:
            v = -1.0 * max_speed_

        if v > 1.0 * max_speed_:
            v = 1.0 * max_speed_

        return v

                    # 轨迹分析器   计算 误差
class TrajectoryAnalyzer:        # 轨迹分析器

    def __init__(self, x, y, yaw, k):
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        self.k_ = k    #  规划路径 曲率

        self.ind_old = 0   # index
        self.ind_end = len(x)    #


                            # :return: theta_e, e_cg, yaw_ref, k_ref
    def ToTrajectoryFrame(self, vehicle_state):
        """
        errors to trajectory frame    轨迹框误差
        theta_e = yaw_vehicle - yaw_ref_path
        e_cg = lateral distance of center of gravity (cg) in frenet frame
        yaw_ref =  yaw_ref_path
        :param vehicle_state: vehicle state (class VehicleState)
        :return: theta_e, e_cg, yaw_ref, k_ref
        """

        x_cg = vehicle_state.x
        y_cg = vehicle_state.y
        yaw = vehicle_state.yaw

        # calc nearest point in ref path
        dx = [x_cg - ix for ix in self.x_[self.ind_old: self.ind_end]]
        dy = [y_cg - iy for iy in self.y_[self.ind_old: self.ind_end]]

        ind_add = int(np.argmin(np.hypot(dx, dy)))
        dist = math.hypot(dx[ind_add], dy[ind_add])      # 计算出距离

        # calc lateral relative position of vehicle to ref path      计算车辆相对于参考路径的横向相对位置
        vec_axle_rot_90 = np.array([[math.cos(yaw + math.pi / 2.0)],
                                    [math.sin(yaw + math.pi / 2.0)]])

        vec_path_2_cg = np.array([[dx[ind_add]],
                                  [dy[ind_add]]])

        if np.dot(vec_axle_rot_90.T, vec_path_2_cg) > 0.0:    # 点乘
            e_cg = 1.0 * dist  # vehicle on the right of ref path 参考路径右侧的车辆
        else:
            e_cg = -1.0 * dist  # vehicle on the left of ref path  参考路径左侧的车辆

        # calc yaw error: theta_e = yaw_vehicle - yaw_ref
        self.ind_old += ind_add
        yaw_ref = self.yaw_[self.ind_old]
        theta_e = pi_2_pi(yaw - yaw_ref)

        # calc ref curvature   计算参考曲率
        k_ref = self.k_[self.ind_old]
        # print("ind_old: ",self.ind_old)
        return theta_e, e_cg, yaw_ref, k_ref, self.ind_old  # ??????????

               #     Lateral Controller using LQR     基于LQR的横向控制器


class LatController:
    """
    Lateral Controller using LQR     基于LQR的横向控制器
    """
                                                        #  return steer_angle, theta_e, e_cg
    def ComputeControlCommand(self, vehicle_state, ref_trajectory):
        """
        calc lateral control command.
        :param vehicle_state: vehicle state
        :param ref_trajectory: reference trajectory (analyzer)
        :return: steering angle (optimal u), theta_e, e_cg
        """

        ts_ = ts
        e_cg_old = vehicle_state.e_cg
        theta_e_old = vehicle_state.theta_e    #  存储上次值

        theta_e, e_cg, yaw_ref, k_ref, dir_index = \
            ref_trajectory.ToTrajectoryFrame(vehicle_state)      #根据计算存入  新值

        matrix_ad_, matrix_bd_ = self.UpdateMatrix(vehicle_state)   #    什么矩阵

        matrix_state_ = np.zeros((state_size, 1))

        matrix_r_ = np.diag(matrix_r)               #  array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵    #array是一个二维矩阵时，结果输出矩阵的对角线元素
        matrix_q_ = np.diag(matrix_q)          # 计算出 Q矩阵， R矩阵

        matrix_k_ = self.SolveLQRProblem(matrix_ad_, matrix_bd_, matrix_q_,   # 算法计算出 反馈矩阵 K
                                         matrix_r_, eps, max_iteration)

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cg_old) / ts_
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts_

        steer_angle_feedback = -(matrix_k_ @ matrix_state_)[0][0]
        # print(steer_angle_feedback)

        # steer_angle_feedforward = self.ComputeFeedForward(k_ref)    # 前馈控制项
        steer_angle_feedforward = 0
        steer_angle = steer_angle_feedback + steer_angle_feedforward     # 前馈控制 + 反馈控制

        return steer_angle, theta_e, e_cg, dir_index    #  ?????????

    @staticmethod       #  计算前馈控制项以减小稳态误差。
    def ComputeFeedForward(ref_curvature):
        """
        calc feedforward control term to decrease the steady error.    计算前馈控制项以减小稳态误差。
        :param ref_curvature: curvature of the target point in ref trajectory
        :return: feedforward term
        """

        wheelbase_ = l_f + l_r

        steer_angle_feedforward = wheelbase_ * ref_curvature

        return steer_angle_feedforward

    @staticmethod    #  求 matrix_k_
    def SolveLQRProblem(A, B, Q, R, tolerance, max_num_iteration):
        """
        iteratively calculating feedback matrix K  迭代计算反馈矩阵K
        :param A: matrix_a_
        :param B: matrix_b_
        :param Q: matrix_q_
        :param R: matrix_r_
        :param tolerance: lqr_eps
        :param max_num_iteration: max_iteration
        :return: feedback matrix K
        """

        assert np.size(A, 0) == np.size(A, 1) and \
               np.size(B, 0) == np.size(A, 0) and \
               np.size(Q, 0) == np.size(Q, 1) and \
               np.size(Q, 0) == np.size(A, 1) and \
               np.size(R, 0) == np.size(R, 1) and \
               np.size(R, 0) == np.size(B, 1), \
            "LQR solver: one or more matrices have incompatible dimensions."

        M = np.zeros((np.size(Q, 0), np.size(R, 1)))

        AT = A.T    # 转置矩阵
        BT = B.T
        MT = M.T

        P = Q
        num_iteration = 0
        diff = math.inf

        while num_iteration < max_num_iteration and diff > tolerance:
            num_iteration += 1
            P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                     np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

            # check the difference between P and P_next
            diff = (abs(P_next - P)).max()
            P = P_next

        if num_iteration >= max_num_iteration:
            print("LQR solver cannot converge to a solution",
                  "last consecutive result diff is: ", diff)

        K = np.linalg.inv(BT @ P @ B + R) @ (BT @ P @ A + MT)

        return K

    @staticmethod    #    matrix_ad_, matrix_bd_
    def UpdateMatrix(vehicle_state):
        """
        calc A and b matrices of linearized, discrete system.  线性化离散系统的计算A和b矩阵。
        :return: A, b
        """

        ts_ = ts
        wheelbase_ = l_f + l_r

        v = vehicle_state.v

        matrix_ad_ = np.zeros((state_size, state_size))  # time discrete A matrix

        matrix_ad_[0][0] = 1.0
        matrix_ad_[0][1] = ts_
        matrix_ad_[1][2] = v
        matrix_ad_[2][2] = 1.0
        matrix_ad_[2][3] = ts_

        # b = [0.0, 0.0, 0.0, v / L].T
        matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
        matrix_bd_[3][0] = v / wheelbase_

        return matrix_ad_, matrix_bd_

             #    Longitudinal Controller using PID.    纵向控制器采用PID。


class LonController:
    """
    Longitudinal Controller using PID.    纵向控制器采用PID。
    """

    @staticmethod
    def ComputeControlCommand(target_speed, vehicle_state, dist):
        """
        calc acceleration command using PID.
        :param target_speed: target speed [m / s]
        :param vehicle_state: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """

        if vehicle_state.gear == Gear.GEAR_DRIVE:
            direct = 1.0
        else:
            direct = -1.0

        a = 0.3 * (target_speed - direct * vehicle_state.v)

        if dist < 10.0:
            if vehicle_state.v > 2.0:
                a = -3.0
            elif vehicle_state.v < -2:
                a = -1.0

        return a


def pi_2_pi(angle):
    """
    regulate theta to -pi ~ pi.
    :param angle: input angle
    :return: regulated angle
    """

    M_PI = math.pi

    if angle > M_PI:
        return angle - 2.0 * M_PI

    if angle < -M_PI:
        return angle + 2.0 * M_PI

    return angle


def generate_path(s):
    """
    design path using reeds-shepp path generator.
    divide paths into sections, in each section the direction is the same.
    :param s: objective positions and directions.
    :return: paths
    """
    wheelbase_ = l_f + l_r   # 车宽

    max_c = math.tan(0.5 * max_steer_angle) / wheelbase_        #最大曲率
    path_x, path_y, yaw, direct, rc = [], [], [], [], []         # 路径 列表
    x_rec, y_rec, yaw_rec, direct_rec, rc_rec = [], [], [], [], []
    direct_flag = 1.0

    for i in range(len(s) - 1):
        s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        path_i = rs.calc_optimal_path(s_x, s_y, s_yaw,
                                      g_x, g_y, g_yaw, max_c)

        irc, rds = rs.calc_curvature(path_i.x, path_i.y, path_i.yaw, path_i.directions)   # 计算曲率

        ix = path_i.x
        iy = path_i.y
        iyaw = path_i.yaw
        idirect = path_i.directions

        for j in range(len(ix)):
            if idirect[j] == direct_flag:
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                direct_rec.append(idirect[j])
                rc_rec.append(irc[j])
            else:
                if len(x_rec) == 0 or direct_rec[0] != direct_flag:
                    direct_flag = idirect[j]
                    continue

                path_x.append(x_rec)
                path_y.append(y_rec)
                yaw.append(yaw_rec)
                direct.append(direct_rec)
                rc.append(rc_rec)
                x_rec, y_rec, yaw_rec, direct_rec, rc_rec = \
                    [x_rec[-1]], [y_rec[-1]], [yaw_rec[-1]], [-direct_rec[-1]], [rc_rec[-1]]

    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    direct.append(direct_rec)
    rc.append(rc_rec)

    x_all, y_all = [], []
    for ix, iy in zip(path_x, path_y):
        x_all += ix
        y_all += iy

    return path_x, path_y, yaw, direct, rc, x_all, y_all


def main():
    # generate path
    states = [(0, 0, 0), (20, 15, 0), (35, 20, 90), (40, 0, 180),
              (20, 0, 120), (5, -10, 180), (15, 5, 30)]
    #
    # states = [(-3, 3, 120), (10, -7, 30), (10, 13, 30), (20, 5, -25),
    #           (35, 10, 180), (30, -10, 160), (5, -12, 90)]


    x_ref, y_ref, yaw_ref, direct, curv, x_all, y_all = generate_path(states)
    # print(len(x_ref))
    # print(x_all)
    # print(yaw_ref)
    # print(x_ref,"\n", y_ref,'\n', yaw_ref,'\n', direct,'\n' ,curv,'\n', x_all,'\n', y_all)

    wheelbase_ = l_f + l_r

    maxTime = 100.0   # 模拟100  秒
    yaw_old = 0.0     # 初始方向角
    x0, y0, yaw0, direct0 = \
        x_ref[0][0], y_ref[0][0], yaw_ref[0][0], direct[0][0]  # 从第一段路径开始

    x_rec, y_rec, yaw_rec, direct_rec = [], [], [], []     #   根据下面 推断， 记录 车辆历史状态

    lat_controller = LatController()  #     基于LQR的横向控制器
    lon_controller = LonController()  #     纵向控制器采用PID

    # print(direct)
    for x, y, yaw, gear, k in zip(x_ref, y_ref, yaw_ref, direct, curv):    # 遍历 路径
        t = 0.0                     # 此处遍历 7 次，  每段路上 方向相同
        # print(gear)

        if gear[0] == 1.0:         # 正转
            direct = Gear.GEAR_DRIVE
        else:
            direct = Gear.GEAR_REVERSE
        # print(direct)
                                                # 定义了一个 轨迹分析器 ,对象
        ref_trajectory = TrajectoryAnalyzer(x, y, yaw, k)        # return theta_e, e_cg, yaw_ref, k_ref
       # print(ref_trajectory)
        vehicle_state = VehicleState(x=x0, y=y0, yaw=yaw0, v=0.1, gear=direct)
                                        # 定义了一个 车辆状态  对象
        # print(t)
'''
        while t < maxTime:   # 每段距离 最长 跑 100 s

            dist = math.hypot(vehicle_state.x - x[-1], vehicle_state.y - y[-1])     # 车辆 距离 终点 的距离

            if gear[0] > 0:
                target_speed = 25.0 / 3.6
            else:
                target_speed = 15.0 / 3.6
                                                          # 控制器 部分
            delta_opt, theta_e, e_cg = \
                lat_controller.ComputeControlCommand(vehicle_state, ref_trajectory)   # return: steering angle (optimal u), theta_e, e_cg

            a_opt = lon_controller.ComputeControlCommand(target_speed, vehicle_state, dist)   # return: control command (acceleration) [m / s^2]

            vehicle_state.UpdateVehicleState(delta_opt, a_opt, e_cg, theta_e, direct)        #  更新车辆 状态

            t += ts    #  ts = 0.1  # [s]

            if dist <= 0.5:   # ？？？
                break

            x_rec.append(vehicle_state.x)
            y_rec.append(vehicle_state.y)
            yaw_rec.append(vehicle_state.yaw)     #  记录车辆历史状态

            dy = (vehicle_state.yaw - yaw_old) / (vehicle_state.v * ts)        #  dy  是干嘛的？

            # steer = rs.pi_2_pi(-math.atan(wheelbase_ * dy))

            yaw_old = vehicle_state.yaw
            x0 = x_rec[-1]
            y0 = y_rec[-1]
            yaw0 = yaw_rec[-1]

            plt.cla()   # 清楚图中 当前 活动 轴
            plt.plot(x_all, y_all, color='gray', linewidth=2.0)     # 画   地图路径
            plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')   # 画 车辆 已走过的 路径
            # plt.plot(x[ind], y[ind], '.r')
            draw.draw_car(x0, y0, yaw0, -vehicle_state.steer)   # 画 当前 车辆 状态
            plt.axis("equal")
            plt.title("LQR (Kinematic): v=" + str(vehicle_state.v * 3.6)[:4] + "km/h")
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event:
                                         [exit(0) if event.key == 'escape' else None])
            plt.pause(0.001)

    plt.show()

'''
if __name__ == '__main__':
    main()
