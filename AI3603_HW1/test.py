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

    pathx, pathy = astar_planning(current_pos[0], current_pos[1], goal_pos[0], goal_pos[1],obstacles_x, obstacles_y, 1.0, 2.0 )
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
                                    #起点车信息， 终点车信息， 障碍物信息，  距离分辨率，航向角分辨率
    path = hybrid_astar_planning(current_pos[0], current_pos[1], current_ori, goal_pos[0], goal_pos[1], goal_ori, obstacles_x, obstacles_y, 1, np.deg2rad(1.0))


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
    if np.linalg.norm(np.array(goal_pos) - np.array(current_pos)) < 1:
        is_reached = True
    else:
        is_reached = False

    ###  END CODE HERE  ###
    return is_reached





if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    goal_ori = np.deg2rad(90)

    controller = DR20API.Controller()   # 实例化 一个 Robot 控制器对象

    controller.move_init()
    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_ori = controller.get_robot_ori()
    current_map = controller.update_map()


    print(f'Current position of the robot is {current_pos}.')

    print(f'Current orientation of the robot is {current_ori}.')

    # Define a test path.
    path = [[16, 17], [17, 17], [18, 17], [19, 17], [20, 17], [21, 17], [22, 17], [23, 17], [24, 17], [25, 17], [26, 17], [27, 17]]
    # Visualize the map and path.   可视化地图和路径
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)


    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)   # 生成网格线
    plt.axis("equal")
    plt.show()

    print('move')
    # Move the robot along the test path.
    print('Moving the robot ...')
    print("cao")
    controller.test_move()


    # Stop the simulation.
    controller.stop_simulation()

