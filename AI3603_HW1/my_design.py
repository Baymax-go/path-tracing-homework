import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/DR20API/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/Control/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/HybridAstarPlanner/")

import DR20API
from HybridAstarPlanner.astar import *
from HybridAstarPlanner.hybrid_astar import *
from Control.reeds_shepp import *
from Control.LQR_Kinematic_Model import *

def A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal,
    plan a path from current position to the goal using A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)
    obstacles_x.append(0)
    obstacles_x.append(119)
    obstacles_y.append(0)
    obstacles_y.append(119)


    plt.plot(obstacles_x,obstacles_y,"sk")
    plt.plot(current_pos[0], current_pos[1],"sk")
    plt.show()

    # print("obstacles_x: ",obstacles_x)
    # print("obstacles_y: ",obstacles_y)
    pathx, pathy = astar_planning(current_pos[0], current_pos[1], goal_pos[0], goal_pos[1],obstacles_x, obstacles_y, 1.0, 0.5 )
    path = [pathx, pathy]


    ###  END CODE HERE  ###
    return path


def Hybrid_A_star(current_map, current_pos, current_ori, goal_pos, goal_ori):
    ### START CODE HERE ###
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)
    obstacles_x.append(0)
    obstacles_x.append(119)
    obstacles_y.append(0)
    obstacles_y.append(119)

    plt.plot(obstacles_x, obstacles_y, "sk")
                                    #起点车信息， 终点车信息， 障碍物信息，  距离分辨率，航向角分辨率
    # print(current_pos[0],current_pos[1],current_ori)
    # print(goal_pos[0], goal_pos[1], goal_ori)
    # print(obstacles_x, obstacles_y)
    path = hybrid_astar_planning(current_pos[0], current_pos[1], current_ori, goal_pos[0], goal_pos[1], goal_ori, obstacles_x, obstacles_y, 1.0, np.deg2rad(1.0))
                       #     path = Path(rx, ry, ryaw, direc, cost)
    plt.plot(path.x, path.y, linewidth=1.5, color='r')
    # plt.plot(list(range(len(path.direction))),path.direction)
    plt.show()
    # print(path.direction)
    ###  END CODE HERE  ###
    return path



#    return is_reached
def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot,
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###
    if np.linalg.norm(np.array(goal_pos) - np.array(current_pos)) <= 1.5:
        is_reached = True
    else:
        is_reached = False
    print(is_reached)
    ###  END CODE HERE  ###
    return is_reached




if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    goal_ori = np.deg2rad(50)

    controller = DR20API.Controller()   # 实例化 一个 Robot 控制器对象

    controller.move_init()
    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_ori = controller.get_robot_ori() + np.pi  # 补偿
    current_map = controller.update_map()

    # print("current_pos:",current_pos) ##Current position of the robot is [13 12].
    # print("current_ori",current_ori)  # Current orientation of the robot is -1.561651349067688.

    # path = A_star(current_map, current_pos, goal_pos)
    # print(np.array(path))
    # plt.plot(path[0],path[1],)
    # plt.show()

    path = Hybrid_A_star(current_map, current_pos, current_ori, goal_pos, goal_ori)
    # k, _ = calc_curvature(path.x, path.y, path.yaw, path.direction)
    # plt.figure()
    # plt.plot(path.x,k)
    # plt.show()
    # print(k)

    # plt.plot(path.x, path.y, linewidth=1.5, color='r')
    # plt.title("Hybrid A*")
    # plt.show()

    if path.direction[0] == 1:
        vehicle_state =  VehicleState(x=current_pos[0], y=current_pos[1], yaw=current_ori, v=0.1, gear=Gear.GEAR_DRIVE)
    else:
        vehicle_state = VehicleState(x=current_pos[0], y=current_pos[1], yaw=current_ori, v=0.1, gear=Gear.GEAR_REVERSE)

    while not reach_goal(current_pos, goal_pos):

        # Plan a path based on current map from current position of the robot to the goal. 根据当前地图规划从机器人当前位置到目标的路径。
        path = Hybrid_A_star(current_map, current_pos, current_ori, goal_pos, goal_ori)
        # plt.plot(path.x, path.y)
        # plt.title("Hybrid A*")
        # plt.show()
        # Move the robot along the path to a certain distance.     # 这里设定最大 移动 3 m 距离
        if path.direction[0] == 1:
            vehicle_state = VehicleState(x=current_pos[0], y=current_pos[1], yaw=current_ori, v=0.1,
                                         gear=Gear.GEAR_DRIVE)
        else:
            vehicle_state = VehicleState(x=current_pos[0], y=current_pos[1], yaw=current_ori, v=0.1,
                                         gear=Gear.GEAR_REVERSE)

        vehicle_state = controller.LQR_move_robot(vehicle_state, path)

        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        current_ori = controller.get_robot_ori() + np.pi  # 补偿
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

        print("current_pos:",current_pos) ##Current position of the robot is [13 12].
        print("current_ori",current_ori)  # Current orientation of the robot is -1.561651349067688.
        print("here")

    # Stop the simulation.
    controller.stop_simulation()

'''
    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.  计划移动并重新规划循环，直到机器人达到目标。
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal. 根据当前地图规划从机器人当前位置到目标的路径。
        # path = A_star(current_map, current_pos, goal_pos)
        path = Hybrid_A_star(current_map, current_pos, current_ori, goal_pos, goal_ori)
                # path.x   path.y      path.yaw   path.direction   path.cost
        plt.plot(path.x,path.y)
        plt.title("Hybrid A*")
        plt.axis("equal")
        plt.show()
        # print(len(path.x))
        print(path.y)
        # print(path.yaw)

        # path_ = list(zip(path[0],path[1]))

        # print(path_)


        # Move the robot along the path to a certain distance.     # 这里设定最大 移动 3 m 距离
        # controller.LQR_move_robot(path)

        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        current_ori = controller.get_robot_ori()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

        # # print(path_)


    # Stop the simulation.
    controller.stop_simulation()

'''


