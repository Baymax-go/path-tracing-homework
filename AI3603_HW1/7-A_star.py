import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/DR20API/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/HybridAstarPlanner/")
import DR20API
import numpy as np

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.


from HybridAstarPlanner.astar import *


###  END CODE HERE  ###
#    return path
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
    if np.linalg.norm(np.array(goal_pos) - np.array(current_pos)) < 2:
        is_reached = True
    else:
        is_reached = False

    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()
    # print(len(current_map))
    # print(current_map[0][0])
    # print(current_map[119][119])
    # controller.stop_simulation()

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.  计划移动并重新规划循环，直到机器人达到目标。
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal. 根据当前地图规划从机器人当前位置到目标的路径。
        path = A_star(current_map, current_pos, goal_pos)

        # print(path)
        path_ = list(zip(path[0],path[1]))
        # print(path_)

        # Move the robot along the path to a certain distance.     # 这里设定最大 移动 3 m 距离
        controller.move_robot(path_)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

        # # print(path_)


    # Stop the simulation.
    controller.stop_simulation()
