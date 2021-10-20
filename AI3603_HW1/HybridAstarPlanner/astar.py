import heapq
import math
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node


class Para:
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw
        self.yw = yw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set


def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution      网格分辨率
    :param rr: robot radius   机器人半径
    :return: path
    """
                    # cost   p_index
    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)    # 起始位置
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)   #

    open_set, closed_set = dict(), dict()   # 创建字典
    open_set[calc_index(n_start, P)] = n_start       #  ？？？

    q_priority = []         # 空列表
    heapq.heappush(q_priority,       #  一次堆栈压入 两个  一个  f函数值，  一个 索引index
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))

    while True:
        if not open_set:      #  为遍历 空间 非空继续
            break

        _, ind = heapq.heappop(q_priority)       #  弹出 最小值
        n_curr = open_set[ind]   #  取出索引
        closed_set[ind] = n_curr    # 存入 遍历集合
        open_set.pop(ind)      # 集合中去除

        for i in range(len(P.motion)):   # 遍历 8  个方向
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)   #  计算 代价

            if not check_node(node, P, obsmap):   # 检测是否出界，或者是否存在障碍物
                continue  # 跳出当前 for 循环

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P)))

    # print("closed_set:" , closed_set)
    # print("n_start", n_start)
    # print("n_goal", n_goal)
    # print("P: ", P)

    pathx, pathy = extract_path(closed_set, n_start, n_goal, P)   #  得到规划路径

    return pathx, pathy


def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):   # 带障碍的 完整启发函数 cost 计算
    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, reso, rr)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))   #

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]   # 初始化 无穷矩阵
#    print(P.yw,P.xw)
    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap


def check_node(node, P, obsmap):
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False

    return True


def u_cost(u):
    return math.hypot(u[0], u[1])


def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)


def h(node, n_goal):
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)   # hypot 欧几里德范数


def calc_index(node, P):
  #  print(P.miny,P.xw,P.minx)
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


def calc_parameters(ox, oy, rr, reso):
    minx, miny = round(min(ox)), round(min(oy))    #  round() 方法返回浮点数x的四舍五入值。
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = maxx - minx, maxy - miny     # 得到长 宽

    motion = get_motion()    #  得到机器人8个运动 动作
    P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)   # 环境参数
    obsmap = calc_obsmap(ox, oy, rr, P)   #  障碍地图
    # print(obsmap)
    return P, obsmap


def calc_obsmap(ox, oy, rr, P):
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]   #  _ 是占位符， 表示不在意变量的值 只是用于循环遍历n次。
                                          # 创建一个空地图
    for x in range(P.xw):          # 遍历地图每一个点
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):  #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:  #hypot() 返回欧几里德范数 sqrt(x*x + y*y)。
                    obsmap[x][y] = True
                    break

    return obsmap


def extract_path(closed_set, n_start, n_goal, P):
    pathx, pathy = [n_goal.x], [n_goal.y]
    n_ind = calc_index(n_goal, P)

    while True:
        # print("n_ind_length: ", n_ind)
        # print("len(closed_set): ", len(closed_set))
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        n_ind = node.pind

        if node == n_start:
            break                 #  到此 反向遍历 路径

    pathx = [x * P.reso for x in reversed(pathx)]     # 将数组反向，计算真实路径
    pathy = [y * P.reso for y in reversed(pathy)]

    return pathx, pathy


def get_motion():    #   8 个 机器人运动方向
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion


def get_env():
    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):     # 地图边界
        ox.append(0.0)
        oy.append(i)
    for i in range(40):     # 障碍物
        ox.append(20.0)
        oy.append(i)
    for i in range(40):  # 障碍物
        ox.append(40.0)
        oy.append(60.0 - i)

    return ox, oy


def main():
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    robot_radius = 2.0      # 机器人半径为 2.0
    grid_resolution = 1.0    # 机器人  步长 1.0
    ox, oy = get_env()         # 生成地图

    pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)


    # aaa=zip(pathx, pathy)
    # for i in aaa:
    #     print(i)

    plt.plot(ox, oy, 'sk')
    plt.plot(pathx, pathy, '-r')
    plt.plot(sx, sy, 'sg')
    plt.plot(gx, gy, 'sb')
    plt.axis("equal")
    plt.show()



if __name__ == '__main__':
    main()
