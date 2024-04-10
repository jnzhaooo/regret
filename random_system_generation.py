from networkx.classes.digraph import DiGraph
from basic_operations import PK_WTS, DFA, product, shortest_path, PK_WTS_Grid, Dijskstra, WTS_of_PK_WTS
from Knowledge_Game_Arena import Knowledge_Game_Arena
from strategy_synthesis import regret_min_strategy_synthesis

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches


def Dijkstra(lapmat, start, end):
    n = lapmat.shape[0]
    cost = np.zeros([n])
    prev = np.zeros([n], dtype=int)
    visited = np.zeros([n], dtype=int)
    # init
    for i in range(n):
        for j in range(n):
            if lapmat[i, j] == -1:
                lapmat[i, j] = 10000
    for i in range(n):
        cost[i] = lapmat[start, i]
        prev[i] = start
        visited[i] = -1
    # dijk
    for i in range(n):
        unvisited_idx = []
        for j in range(n):
            if visited[j] == -1:
                unvisited_idx.append(j)
        unvisited_cost = cost[unvisited_idx]
        sorted_idx = np.argsort(unvisited_cost)
        next_idx = unvisited_idx[sorted_idx[0]]
        visited[next_idx] = 1
        for j in range(n):
            if cost[j] > cost[next_idx] + lapmat[next_idx, j]:
                cost[j] = cost[next_idx] + lapmat[next_idx, j]
                prev[j] = next_idx
    # back-track
    target_list = []
    current_node = end
    while True:
        target_list.append(int(current_node))
        if current_node == start:
            break
        current_node = prev[current_node]
    return target_list[::-1], cost[end]


def gen_random_system(width, height, num_f, num_e, num_uncertainty, num_obstacle, num_initial, connectivity):
    np.random.seed(int(time.time()))
    n = width * height
    o_ids = random.sample(range(n), num_obstacle)
    non_obstacles = list(range(n))
    for o_id in o_ids:
        non_obstacles.remove(o_id)
    f_ids = random.sample(non_obstacles, num_f)
    e_ids = random.sample(non_obstacles, num_e)
    i_ids = random.sample(non_obstacles, num_initial)

    temp_lapmat = -np.ones([n, n]) + np.eye(n, n)
    Node_list = []
    Edge_list = []
    Initial_list = []
    for i in range(width):
        for j in range(height):
            node_name = "w" + str(i) + "h" + str(j)
            attr = []
            self_id = np.ravel_multi_index([i, j], [width, height])
            is_obstacle = False
            for id_ in o_ids:
                if self_id == id_:
                    is_obstacle = True
                    break
            if is_obstacle:
                Node_list.append((node_name, {None}))
                continue
            for id_ in f_ids:
                if self_id == id_:
                    attr.append("f")
                    break
            for id_ in e_ids:
                if self_id == id_:
                    attr.append("e")
                    break
            for id_ in i_ids:
                if self_id == id_:
                    Initial_list.append(node_name)
                    break

            if len(attr) == 0:
                attr = {None}
            Node_list.append((node_name, attr))

            # connectivity
            if i > 0:
                left_id = np.ravel_multi_index([i - 1, j], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, left_id] = 1
                    temp_lapmat[left_id, self_id] = 1
            if i < width - 1:
                right_id = np.ravel_multi_index([i + 1, j], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, right_id] = 1
                    temp_lapmat[right_id, self_id] = 1
            if j > 0:
                up_id = np.ravel_multi_index([i, j - 1], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, up_id] = 1
                    temp_lapmat[up_id, self_id] = 1
            if j < height - 1:
                down_id = np.ravel_multi_index([i, j + 1], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, down_id] = 1
                    temp_lapmat[down_id, self_id] = 1
    for o_i in o_ids:
        temp_lapmat[:, o_i] = -1
        temp_lapmat[o_i, :] = -1
        temp_lapmat[o_i, o_i] = 0

    for i in range(n):
        for j in range(n):
            if temp_lapmat[i, j] == 1:
                Edge_list.append((Node_list[i][0], Node_list[j][0], 1, False))

    u_e_ids = random.sample(range(len(Edge_list)), num_uncertainty)

    for i in range(len(Edge_list)):
        for u_e_id in u_e_ids:
            if i == u_e_id:
                Edge_list[i] = (Edge_list[i][0], Edge_list[i][1], 1, True)
                for i_ in range(len(Edge_list)):
                    edge = Edge_list[i_]
                    if edge[0] == Edge_list[i][1] and edge[1] == Edge_list[i][0]:
                        Edge_list[i_] = (Edge_list[i_][0], Edge_list[i_][1], 1, True)
    print("system generated.")
    # check reachability
    is_reachable = False
    for i_id in i_ids:
        for e_id in e_ids:
            _, cost = Dijkstra(temp_lapmat, i_id, e_id)
            if cost > 9999.0:
                continue
            for f_id in f_ids:
                _, cost = Dijkstra(temp_lapmat, e_id, f_id)
                if cost < 9999.0:
                    is_reachable = True
                    break
            if is_reachable:
                break
        if is_reachable:
            break
    if is_reachable:
        print("system reachability checked.")
    else:
        return -1
    # plot
    will_plot = True
    if will_plot:
        plt.figure(0)
        plt.xlim([-1, width + 1])
        plt.ylim([-1, height + 1])
        plt.gca().add_patch(
            plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=None, edgecolor=[0, 0, 0], fill=False,
                          linewidth=5))

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        for i in range(width):
            for j in range(height):
                is_obstacle = False
                node_id = np.ravel_multi_index([i, j], [width, height])
                color = [1, 1, 1]
                if len(Node_list[node_id][1]) == 2:
                    color = [1, 1, 0]
                elif list(Node_list[node_id][1])[0] == 'e':
                    color = [0, 1, 0]
                elif list(Node_list[node_id][1])[0] == 'f':
                    color = [1, 0, 0]
                else:
                    for o_id in o_ids:
                        if node_id == o_id:
                            is_obstacle = True
                            color = [0, 0, 0]
                            break
                    for i_id in i_ids:
                        if node_id == i_id:
                            color = [0, 0, 1]
                plt.gca().add_patch(
                    plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                                  linewidth=0))
                if is_obstacle:
                    continue
                # connectivity
                if i > 0:
                    left_id = np.ravel_multi_index([i - 1, j], [width, height])
                    if temp_lapmat[left_id, node_id] == -1:
                        plt.plot([i, i], [j, j + 1], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[left_id][0]:
                                plt.plot([i, i], [j, j + 1], linewidth=3, color=[0, 0, 0], linestyle='dashed')
                if i < width - 1:
                    right_id = np.ravel_multi_index([i + 1, j], [width, height])
                    if temp_lapmat[right_id, node_id] == -1:
                        plt.plot([i + 1, i + 1], [j, j + 1], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[right_id][0]:
                                plt.plot([i + 1, i + 1], [j, j + 1], linewidth=3, color=[0, 0, 0], linestyle='dashed')
                if j > 0:
                    down_id = np.ravel_multi_index([i, j - 1], [width, height])
                    if temp_lapmat[down_id, node_id] == -1:
                        plt.plot([i, i + 1], [j, j], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[down_id][0]:
                                plt.plot([i, i + 1], [j, j], linewidth=3, color=[0, 0, 0], linestyle='dashed')
                if j < height - 1:
                    up_id = np.ravel_multi_index([i, j + 1], [width, height])
                    if temp_lapmat[up_id, node_id] == -1:
                        plt.plot([i, i + 1], [j + 1, j + 1], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[up_id][0]:
                                plt.plot([i, i + 1], [j + 1, j + 1], linewidth=3, color=[0, 0, 0], linestyle='dashed')
        plt.show()
    return Node_list, Edge_list, Initial_list


def gen_random_system_generalized(width, height, ap_names, ap_nums, num_uncertainty, num_obstacle, num_initial,
                                  connectivity, max_connected_obstacles):
    np.random.seed(int(time.time()))
    n = width * height
    # generate obstacles
    if max_connected_obstacles == 1:
        o_ids = random.sample(range(n), num_obstacle)
        non_obstacles = list(range(n))
        for o_id in o_ids:
            non_obstacles.remove(o_id)
    else:
        o_ids = []
        non_obstacles = list(range(n))
        connected_o_nums = []
        num_obstacle_temp = num_obstacle
        while True:
            sup = min(max_connected_obstacles, num_obstacle_temp) + 1
            temp = np.random.randint(1, sup)
            connected_o_nums.append(temp)
            num_obstacle_temp -= temp
            if num_obstacle_temp == 0:
                break
        for connected_o_num in connected_o_nums:
            current_region = []
            o_id = random.sample(non_obstacles, 1)[0]
            current_region.append(o_id)
            able_to_grow = True
            while True:
                if len(current_region) == connected_o_num or not able_to_grow:
                    for o_id in current_region:
                        non_obstacles.remove(o_id)
                        o_ids.append(o_id)
                    break
                neighbor_region = []
                for o_id in current_region:
                    i, j = np.unravel_index(o_id, [width, height])
                    if i > 0:
                        left_id = np.ravel_multi_index([i - 1, j], [width, height])
                        if left_id not in neighbor_region and left_id not in current_region and left_id not in o_ids:
                            neighbor_region.append(left_id)
                    if i < width - 1:
                        right_id = np.ravel_multi_index([i + 1, j], [width, height])
                        if right_id not in neighbor_region and right_id not in current_region and right_id not in o_ids:
                            neighbor_region.append(right_id)
                    if j > 0:
                        down_id = np.ravel_multi_index([i, j - 1], [width, height])
                        if down_id not in neighbor_region and down_id not in current_region and down_id not in o_ids:
                            neighbor_region.append(down_id)
                    if j < height - 1:
                        up_id = np.ravel_multi_index([i, j + 1], [width, height])
                        if up_id not in neighbor_region and up_id not in current_region and up_id not in o_ids:
                            neighbor_region.append(up_id)
                if len(neighbor_region) == 0:
                    able_to_grow = False
                else:
                    next_o_id = random.sample(neighbor_region, 1)[0]
                    current_region.append(next_o_id)

    Potential_obstacle_list = random.sample(non_obstacles, num_uncertainty)
    for p_o_id in Potential_obstacle_list:
        non_obstacles.remove(p_o_id)

    ap_ids = []
    for num in ap_nums:
        ap_ids.append(random.sample(non_obstacles, num))
    i_ids = random.sample(non_obstacles, num_initial)

    temp_lapmat = -np.ones([n, n]) + np.eye(n, n)
    Node_list = []
    Edge_list = []
    Initial_list = []
    for i in range(width):
        for j in range(height):
            node_name = "w" + str(i) + "h" + str(j)
            attr = []
            self_id = np.ravel_multi_index([i, j], [width, height])
            is_obstacle = False
            for id_ in o_ids:
                if self_id == id_:
                    is_obstacle = True
                    break
            if is_obstacle:
                Node_list.append((node_name, {None}))
                continue
            for ap_i in range(len(ap_ids)):
                ids_ = ap_ids[ap_i]
                for id_ in ids_:
                    if self_id == id_:
                        attr.append(ap_names[ap_i])
                        break
            for id_ in i_ids:
                if self_id == id_:
                    Initial_list.append(node_name)
                    break

            if len(attr) == 0:
                attr = {None}
            Node_list.append((node_name, attr))

            # connectivity
            if i > 0:
                left_id = np.ravel_multi_index([i - 1, j], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, left_id] = 1
                    temp_lapmat[left_id, self_id] = 1
            if i < width - 1:
                right_id = np.ravel_multi_index([i + 1, j], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, right_id] = 1
                    temp_lapmat[right_id, self_id] = 1
            if j > 0:
                down_id = np.ravel_multi_index([i, j - 1], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, down_id] = 1
                    temp_lapmat[down_id, self_id] = 1
            if j < height - 1:
                up_id = np.ravel_multi_index([i, j + 1], [width, height])
                if np.random.rand(1) < connectivity:
                    temp_lapmat[self_id, up_id] = 1
                    temp_lapmat[up_id, self_id] = 1

    for p_o_i in Potential_obstacle_list:
        i, j = np.unravel_index(p_o_i, [width, height])
        if i > 0:
            left_id = np.ravel_multi_index([i - 1, j], [width, height])
            temp_lapmat[left_id, p_o_i] = 1
            temp_lapmat[p_o_i, left_id] = 1
        if i < width - 1:
            right_id = np.ravel_multi_index([i + 1, j], [width, height])
            temp_lapmat[right_id, p_o_i] = 1
            temp_lapmat[p_o_i, right_id] = 1  
        if j > 0:
            down_id = np.ravel_multi_index([i, j - 1], [width, height])
            temp_lapmat[down_id, p_o_i] = 1
            temp_lapmat[p_o_i, down_id] = 1
        if j < height - 1:
            up_id = np.ravel_multi_index([i, j + 1], [width, height])
            temp_lapmat[up_id, p_o_i] = 1
            temp_lapmat[p_o_i, up_id] = 1

    for o_i in o_ids:
        temp_lapmat[:, o_i] = -1
        temp_lapmat[o_i, :] = -1
        temp_lapmat[o_i, o_i] = 0

    for i in range(n):
        for j in range(n):
            if temp_lapmat[i, j] == 1:
                is_uncertain = False
                for p_o_i in Potential_obstacle_list:
                    if p_o_i == i or p_o_i == j:
                        is_uncertain = True
                        break
                Edge_list.append((Node_list[i][0], Node_list[j][0], 1, is_uncertain))

    # print("system generated.")
    # check reachability

    # plot
    will_plot = False
    if will_plot:
        plt.figure(0)
        plt.xlim([-1, width + 1])
        plt.ylim([-1, height + 1])
        plt.gca().add_patch(
            plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=None, edgecolor=[0, 0, 0], fill=False,
                          linewidth=5))

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        random_colors = np.random.rand(len(ap_names), 3)

        for i in range(width):
            for j in range(height):
                is_obstacle = False
                node_id = np.ravel_multi_index([i, j], [width, height])
                ap_list = list(Node_list[node_id][1])
                if ap_list[0] is None:
                    color = np.array([1., 1., 1.])
                else:
                    color = np.array([0., 0., 0.])
                    for ap_name in ap_list:
                        ap_i = ap_names.index(ap_name)
                        color += random_colors[ap_i, :]
                    color /= len(ap_list)
                for o_id in o_ids:
                    if node_id == o_id:
                        is_obstacle = True
                        color = [0., 0., 0.]
                        break
                for p_o_id in Potential_obstacle_list:
                    if node_id == p_o_id:
                        # is_potential_obstacle = True
                        color = [0.5, 0.5, 0.5]
                        break
                for i_id in i_ids:
                    if node_id == i_id:
                        color = [0., 0., 1.]
                plt.gca().add_patch(
                    plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                                  linewidth=0))
                if is_obstacle:
                    continue
                # connectivity
                if i > 0:
                    left_id = np.ravel_multi_index([i - 1, j], [width, height])
                    if temp_lapmat[left_id, node_id] == -1:
                        plt.plot([i, i], [j, j + 1], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[left_id][0]:
                                plt.plot([i, i], [j, j + 1], linewidth=3, color=[0, 0, 0], linestyle='dashed')
                if i < width - 1:
                    right_id = np.ravel_multi_index([i + 1, j], [width, height])
                    if temp_lapmat[right_id, node_id] == -1:
                        plt.plot([i + 1, i + 1], [j, j + 1], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[right_id][0]:
                                plt.plot([i + 1, i + 1], [j, j + 1], linewidth=3, color=[0, 0, 0], linestyle='dashed')
                if j > 0:
                    down_id = np.ravel_multi_index([i, j - 1], [width, height])
                    if temp_lapmat[down_id, node_id] == -1:
                        plt.plot([i, i + 1], [j, j], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[down_id][0]:
                                plt.plot([i, i + 1], [j, j], linewidth=3, color=[0, 0, 0], linestyle='dashed')
                if j < height - 1:
                    up_id = np.ravel_multi_index([i, j + 1], [width, height])
                    if temp_lapmat[up_id, node_id] == -1:
                        plt.plot([i, i + 1], [j + 1, j + 1], linewidth=3, color=[0, 0, 0])
                    else:
                        for edge in Edge_list:
                            if edge[3] and edge[0] == Node_list[node_id][0] and edge[1] == Node_list[up_id][0]:
                                plt.plot([i, i + 1], [j + 1, j + 1], linewidth=3, color=[0, 0, 0], linestyle='dashed')
        plt.savefig('random_system.png')
        #plt.show()
    return Node_list, Edge_list, Initial_list, Potential_obstacle_list


def gen_random_system_task_specialized(width, height, ap_names, ap_nums, num_uncertainty, num_obstacle, num_initial,
                                       connectivity, max_connected_obstacles, automaton):
    while True:
        node_list, edge_list, initial_list, _ = gen_random_system_generalized(width, height, ap_names, ap_nums,
                                                                              0,
                                                                              num_obstacle,
                                                                              num_initial,
                                                                              connectivity,
                                                                              max_connected_obstacles)
        pk_wts = PK_WTS(node_list, edge_list, initial_list)
        if pk_wts.out_degree[initial_list[0]] == 0:
            continue
        PS = product(pk_wts, automaton)
        # judge the length of accepting run
        length = Dijskstra(PS, list(PS.graph["initial"])[0], list(PS.graph["acc"]))
        if width + 2 < length < 9998:
            # insert the uncertain obstacles
            path = shortest_path(PS, list(PS.graph["initial"])[0], list(PS.graph["acc"]))
            path.remove(path[0])
            for i in range(len(path)):
                path[i] = path[i][0]
            if num_uncertainty > 1:
                first_potential_obstacle_name = random.sample(list(path), 1)[0]
                in_degree_set = list(pk_wts.in_degree)
                non_obstacle_set = []
                for node in in_degree_set:
                    if node[1] > 0:
                        non_obstacle_set.append(node[0])
                non_obstacle_set.remove(first_potential_obstacle_name)
                potential_obstacle_names = random.sample(non_obstacle_set, num_uncertainty - 1)
                potential_obstacle_names.append(first_potential_obstacle_name)
            else:
                potential_obstacle_names = random.sample(list(path), 1)[0]
                potential_obstacle_names = [potential_obstacle_names]
            # check feasibility
            worst_edge_list = list(edge_list)
            for name in potential_obstacle_names:
                for edge in worst_edge_list:
                    if name in edge:
                        worst_edge_list.remove(edge)
            worst_wts = PK_WTS(node_list, worst_edge_list, initial_list)
            if worst_wts.out_degree[initial_list[0]] == 0:
                continue
            worst_PS = product(worst_wts, automaton)
            worst_length = 100000
            worst_length = Dijskstra(worst_PS, list(worst_PS.graph["initial"])[0], list(worst_PS.graph["acc"]))
            if length < worst_length < 9998:
                worst_path = shortest_path(worst_PS, list(worst_PS.graph["initial"])[0], list(worst_PS.graph["acc"]))
                for i in range(len(worst_path)):
                    worst_path[i] = worst_path[i][0]
                potential_obstacles = []
                for name in potential_obstacle_names:
                    potential_obstacles.append(np.ravel_multi_index([int(name[1]), int(name[3])], [width, height]))
                system_has_regret=True
                for worst_path_i in worst_path:
                    i = int(worst_path_i[1])
                    j = int(worst_path_i[3])
                    for potential_obstacle in potential_obstacles:
                        i_, j_ = np.unravel_index(potential_obstacle, [width, height])
                        if abs(i - i_) + abs(j - j_) == 1:
                            system_has_regret=False
                            break
                    if not system_has_regret:
                        break
                if not system_has_regret:
                    continue
                # update the edge list
                #print(potential_obstacle_names)
                potential_walls = []
                for node in potential_obstacle_names:
                    for i in range(len(edge_list)):
                        if edge_list[i][0] == node or edge_list[i][1] == node:
                            for j in range(len(edge_list)):
                                if edge_list[j][0]==edge_list[i][1] and edge_list[j][1]==edge_list[i][0]:
                                    #print(edge_list[j])
                                    edge_list[j] = (edge_list[j][0], edge_list[j][1], edge_list[j][2], True)
                                    #print(edge_list[j])
                                    potential_walls.append(edge_list[j])
                                    break
                            edge_list[i] = (edge_list[i][0], edge_list[i][1], edge_list[i][2], True)
                            #print(edge_list[i])
                            potential_walls.append(edge_list[i])
                            break
                print("specialized system generated.")
                break

    return node_list, edge_list, initial_list, potential_obstacles, potential_walls


def get_obstacles(edge_list, obstacles_with_probability, width, height):
    # return a list. e.g.: [('w1h2', 'w2h2'), ('w2h2','w1h2')]
    obstacles_nodes = []
    obstacles_edges = []
    for (o, p) in obstacles_with_probability:
        if np.random.rand(1) < p:
            continue
        else:
            (i, j) = np.unravel_index(o, (width, height))
            new_node = 'w' + str(i) + 'h' + str(j)
            obstacles_nodes.append(new_node)
            if i > 0:
                left_node = 'w' + str(i - 1) + 'h' + str(j)
                if (new_node, left_node, 1, True) in edge_list:
                    obstacles_edges.append((left_node, new_node))
                    obstacles_edges.append((new_node, left_node))
            if j > 0:
                down_node = 'w' + str(i) + 'h' + str(j - 1)
                if (new_node, down_node, 1, True) in edge_list:
                    obstacles_edges.append((down_node, new_node))
                    obstacles_edges.append((new_node, down_node))
            if i < width - 1:
                right_node = 'w' + str(i + 1) + 'h' + str(j)
                if (new_node, right_node, 1, True) in edge_list:
                    obstacles_edges.append((right_node, new_node))
                    obstacles_edges.append((new_node, right_node))
            if j < height - 1:
                up_node = 'w' + str(i) + 'h' + str(j + 1)
                if (new_node, up_node, 1, True) in edge_list:
                    obstacles_edges.append((up_node, new_node))
                    obstacles_edges.append((new_node, up_node))
    return obstacles_edges


def get_obstacles_xy(edge_list, obstacles_with_probability, width, height):
    # return a list. e.g.: [('w1h2', 'w2h2'), ('w2h2','w1h2')]
    obstacles_nodes = []
    obstacles_edges = []
    for (o, p) in obstacles_with_probability:
        if np.random.rand(1) < p:
            continue
        else:
            (i, j) = np.unravel_index(o, (width, height))
            new_node = 'x' + str(i) + 'y' + str(j)
            obstacles_nodes.append(new_node)
            if i > 0:
                left_node = 'x' + str(i - 1) + 'y' + str(j)
                if (new_node, left_node, 1, True) in edge_list:
                    obstacles_edges.append((left_node, new_node))
                    obstacles_edges.append((new_node, left_node))
            if j > 0:
                down_node = 'x' + str(i) + 'y' + str(j - 1)
                if (new_node, down_node, 1, True) in edge_list:
                    obstacles_edges.append((down_node, new_node))
                    obstacles_edges.append((new_node, down_node))
            if i < width - 1:
                right_node = 'x' + str(i + 1) + 'y' + str(j)
                if (new_node, right_node, 1, True) in edge_list:
                    obstacles_edges.append((right_node, new_node))
                    obstacles_edges.append((new_node, right_node))
            if j < height - 1:
                up_node = 'x' + str(i) + 'y' + str(j + 1)
                if (new_node, up_node, 1, True) in edge_list:
                    obstacles_edges.append((up_node, new_node))
                    obstacles_edges.append((new_node, up_node))
    return obstacles_edges


def get_obstacles_probability(potential_obstacles, probability):
    obstacles_with_probability = list(zip(potential_obstacles, probability))
    return obstacles_with_probability


def plot_determined_system(Node_list, Edge_list, Obstacle_edge_list, Initial_list, width, height, ap_names,
                           actual_path_1=None, actual_path_2=None):
    fig = plt.figure(1)
    plt.xlim([-1, width + 1])
    plt.ylim([-1, height + 1])
    plt.gca().add_patch(
        plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=[0, 0, 0], edgecolor=[0, 0, 0], fill=True,
                      linewidth=5))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # random_colors = np.random.rand(len(ap_names), 3)
    random_colors = np.random.rand(len(ap_names), 3)

    for i in range(width):
        for j in range(height):
            color = [0, 0, 0]
            new_node = 'w' + str(i) + 'h' + str(j)
            if i > 0:
                left_node = 'w' + str(i - 1) + 'h' + str(j)
                if ((new_node, left_node, 1, False) in Edge_list) or ((new_node, left_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j > 0:
                down_node = 'w' + str(i) + 'h' + str(j - 1)
                if ((new_node, down_node, 1, False) in Edge_list) or ((new_node, down_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if i < width - 1:
                right_node = 'w' + str(i + 1) + 'h' + str(j)
                if ((new_node, right_node, 1, False) in Edge_list) or ((new_node, right_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j < height - 1:
                up_node = 'w' + str(i) + 'h' + str(j + 1)
                if ((new_node, up_node, 1, False) in Edge_list) or ((new_node, up_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])

            node_id = np.ravel_multi_index([i, j], [width, height])
            ap_list = list(Node_list[node_id][1])

            if ap_list[0] is None:
                color = color
            else:
                color = np.array([0., 0., 0.])
                for ap_name in ap_list:
                    ap_i = ap_names.index(ap_name)
                    color += random_colors[ap_i, :]
                color /= len(ap_list)

            plt.gca().add_patch(
                plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                              linewidth=0))
    for node in Initial_list:
        i = int(node[1])
        j = int(node[3])
        plt.gca().add_patch(
            plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=[0, 0, 1], edgecolor=None, fill=True,
                          linewidth=0))

    def visual_path(actual_path, offset, linecolor):
        if actual_path is None:
            return -1
        path_data = []
        path_data.append(((int(actual_path[0][1]) + offset, int(actual_path[0][3]) + offset), path.Path.MOVETO))

        for node in actual_path:
            path_data.append(((int(node[1]) + offset, int(node[3]) + offset), path.Path.LINETO))

        verts, codes = zip(*path_data)
        Path = path.Path(verts, codes)
        patch = patches.PathPatch(Path, edgecolor=linecolor, fill=False, lw=2)
        ax.add_patch(patch)

    plt.savefig("determined_system.png")
    visual_path(actual_path_1, 0.3, 'b')
    visual_path(actual_path_2, 0.7, 'r')
    plt.savefig("determined_system_with_trace.png")
    # plt.show()


def plot_determined_system_xy(Node_list, Edge_list, Obstacle_edge_list, Initial_list, width, height, ap_names,
                           actual_path_1=None, actual_path_2=None):
    fig = plt.figure(1)
    plt.xlim([-1, width + 1])
    plt.ylim([-1, height + 1])
    plt.gca().add_patch(
        plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=[0, 0, 0], edgecolor=[0, 0, 0], fill=True,
                      linewidth=5))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # random_colors = np.random.rand(len(ap_names), 3)
    random_colors = np.random.rand(len(ap_names), 3)

    for i in range(width):
        for j in range(height):
            color = [0, 0, 0]
            new_node = 'x' + str(i) + 'y' + str(j)
            if i > 0:
                left_node = 'x' + str(i - 1) + 'y' + str(j)
                if ((new_node, left_node, 1, False) in Edge_list) or ((new_node, left_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j > 0:
                down_node = 'x' + str(i) + 'y' + str(j - 1)
                if ((new_node, down_node, 1, False) in Edge_list) or ((new_node, down_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if i < width - 1:
                right_node = 'x' + str(i + 1) + 'y' + str(j)
                if ((new_node, right_node, 1, False) in Edge_list) or ((new_node, right_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j < height - 1:
                up_node = 'x' + str(i) + 'y' + str(j + 1)
                if ((new_node, up_node, 1, False) in Edge_list) or ((new_node, up_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])

            node_id = np.ravel_multi_index([i, j], [width, height])
            ap_list = list(Node_list[node_id][1])

            if ap_list[0] is None:
                color = color
            else:
                color = np.array([0., 0., 0.])
                """ if ap_list[0] == 'f':
                    color = np.array([1,0,0])
                if ap_list[0] == 'e':
                    color = np.array([0,1,0]) """
                for ap_name in ap_list:
                    ap_i = ap_names.index(ap_name)
                    color = random_colors[ap_i, :]
                color /= len(ap_list)

            plt.gca().add_patch(
                plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                              linewidth=0))
    for node in Initial_list:
        i = int(node[1])
        j = int(node[3])
        plt.gca().add_patch(
            plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=[0, 0, 1], edgecolor=None, fill=True,
                          linewidth=0))

    def visual_path(actual_path, offset, linecolor):
        if actual_path is None:
            return -1
        path_data = []
        path_data.append(((int(actual_path[0][1]) + offset, int(actual_path[0][3]) + offset), path.Path.MOVETO))

        for node in actual_path:
            path_data.append(((int(node[1]) + offset, int(node[3]) + offset), path.Path.LINETO))

        verts, codes = zip(*path_data)
        Path = path.Path(verts, codes)
        patch = patches.PathPatch(Path, edgecolor=linecolor, fill=False, lw=2)
        ax.add_patch(patch)

    plt.savefig("determined_system.png")
    visual_path(actual_path_1, 0.3, 'g')
    visual_path(actual_path_2, 0.7, 'r')
    plt.savefig("determined_system_with_trace.png")
    # plt.show()


def plot_determined_system_wall(Node_list, Edge_list,potential_Walls, Obstacle_edge_list, Initial_list, width, height, ap_names,
                           actual_path_1=None, actual_path_2=None, actual_path_3=None):
    fig = plt.figure(1)
    plt.xlim([-1, width + 1])
    plt.ylim([-1, height + 1])
    plt.gca().add_patch(
        plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=[0, 0, 0], edgecolor=[0, 0, 0], fill=True,
                      linewidth=5))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # random_colors = np.random.rand(len(ap_names), 3)
    random_colors = np.random.rand(len(ap_names), 3)

    for i in range(width):
        for j in range(height):
            color = [0, 0, 0]
            new_node = 'w' + str(i) + 'h' + str(j)
            if i > 0:
                left_node = 'w' + str(i - 1) + 'h' + str(j)
                if ((new_node,left_node,1,True) in potential_Walls) and ((new_node,left_node) not in Obstacle_edge_list):
                     plt.plot([i, i], [j, j + 1], linewidth=3, color=[0, 0, 0])
                if ((new_node, left_node, 1, False) in Edge_list):
                    color = np.array([1., 1., 1.])
            if j > 0:
                down_node = 'w' + str(i) + 'h' + str(j - 1)
                if ((new_node,down_node,1,True) in potential_Walls) and ((new_node,down_node) not in Obstacle_edge_list):
                     plt.plot([i, i+1], [j, j], linewidth=3, color=[0, 0, 0])
                if ((new_node, down_node, 1, False) in Edge_list):
                    color = np.array([1., 1., 1.])
            if i < width - 1:
                right_node = 'w' + str(i + 1) + 'h' + str(j)
                if ((new_node,right_node,1,True) in potential_Walls) and ((new_node,right_node) not in Obstacle_edge_list):
                     plt.plot([i+1, i+1], [j, j + 1], linewidth=3, color=[0, 0, 0])
                if ((new_node, right_node, 1, False) in Edge_list):
                    color = np.array([1., 1., 1.])
            if j < height - 1:
                up_node = 'w' + str(i) + 'h' + str(j + 1)
                if ((new_node,up_node,1,True) in potential_Walls) and ((new_node,up_node) not in Obstacle_edge_list):
                     plt.plot([i, i+1], [j+1, j + 1], linewidth=3, color=[0, 0, 0])
                if ((new_node, up_node, 1, False) in Edge_list):
                    color = np.array([1., 1., 1.])

            node_id = np.ravel_multi_index([i, j], [width, height])
            ap_list = list(Node_list[node_id][1])

            if ap_list[0] is None:
                color = color
            else:
                color = np.array([0., 0., 0.])
                for ap_name in ap_list:
                    ap_i = ap_names.index(ap_name)
                    color += random_colors[ap_i, :]
                color /= len(ap_list)

            plt.gca().add_patch(
                plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                              linewidth=0))
    for node in Initial_list:
        i = int(node[1])
        j = int(node[3])
        plt.gca().add_patch(
            plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=[0, 0, 1], edgecolor=None, fill=True,
                          linewidth=0))

    def visual_path(actual_path, offset, linecolor):
        if actual_path is None:
            return -1
        path_data = []
        path_data.append(((int(actual_path[0][1]) + offset, int(actual_path[0][3]) + offset), path.Path.MOVETO))

        for node in actual_path:
            path_data.append(((int(node[1]) + offset, int(node[3]) + offset), path.Path.LINETO))

        verts, codes = zip(*path_data)
        Path = path.Path(verts, codes)
        patch = patches.PathPatch(Path, edgecolor=linecolor, fill=False, lw=2)
        ax.add_patch(patch)

    plt.savefig("determined_system.png")
    visual_path(actual_path_1, 0.25, 'r')
    visual_path(actual_path_2, 0.5, 'b')
    visual_path(actual_path_3,0.75,'g')
    plt.savefig("determined_system_with_trace.png")
    # plt.show()



def plot_system(Node_list, Edge_list, Obstacle_edge_list, Potential_obstacle_list, Initial_list, width, height,
                ap_names):
    Potential_obstacle_names = []
    for p_o_i in Potential_obstacle_list:
        i, j = np.unravel_index(p_o_i, [width, height])
        Potential_obstacle_names.append("w" + str(i) + "h" + str(j))

    fig = plt.figure(1)
    plt.xlim([-1, width + 1])
    plt.ylim([-1, height + 1])
    plt.gca().add_patch(
        plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=[0, 0, 0], edgecolor=[0, 0, 0], fill=True,
                      linewidth=5))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # random_colors = np.random.rand(len(ap_names), 3)
    random_colors = np.random.rand(len(ap_names), 3)

    for i in range(width):
        for j in range(height):
            color = [0, 0, 0]
            new_node = 'w' + str(i) + 'h' + str(j)
            if i > 0:
                left_node = 'w' + str(i - 1) + 'h' + str(j)
                if ((new_node, left_node, 1, False) in Edge_list) or ((new_node, left_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j > 0:
                down_node = 'w' + str(i) + 'h' + str(j - 1)
                if ((new_node, down_node, 1, False) in Edge_list) or ((new_node, down_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if i < width - 1:
                right_node = 'w' + str(i + 1) + 'h' + str(j)
                if ((new_node, right_node, 1, False) in Edge_list) or ((new_node, right_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j < height - 1:
                up_node = 'w' + str(i) + 'h' + str(j + 1)
                if ((new_node, up_node, 1, False) in Edge_list) or ((new_node, up_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if new_node in Potential_obstacle_names:
                color = np.array([0.5, 0.5, 0.5])

            node_id = np.ravel_multi_index([i, j], [width, height])
            ap_list = list(Node_list[node_id][1])

            if ap_list[0] is None:
                color = color
            else:
                color = np.array([0., 0., 0.])
                for ap_name in ap_list:
                    ap_i = ap_names.index(ap_name)
                    color += random_colors[ap_i, :]
                color /= len(ap_list)

            plt.gca().add_patch(
                plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                              linewidth=0))
    for node in Initial_list:
        i = int(node[1])
        j = int(node[3])
        plt.gca().add_patch(
            plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=[0, 0, 1], edgecolor=None, fill=True,
                          linewidth=0))
    plt.savefig("random_environment.png")
    #plt.show()

def plot_system_wall(Node_list, Edge_list, Potential_wall_list,Obstacle_edge_list, Initial_list, width, height,
                ap_names):


    fig = plt.figure(1)
    plt.xlim([-1, width + 1])
    plt.ylim([-1, height + 1])
    plt.gca().add_patch(
        plt.Rectangle(xy=(0, 0), width=width, height=height, facecolor=[0, 0, 0], edgecolor=[0, 0, 0], fill=True,
                      linewidth=5))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # random_colors = np.random.rand(len(ap_names), 3)
    random_colors = np.random.rand(len(ap_names), 3)

    for i in range(width):
        for j in range(height):
            color = [0, 0, 0]
            new_node = 'w' + str(i) + 'h' + str(j)
            if i > 0:
                left_node = 'w' + str(i - 1) + 'h' + str(j)
                if (new_node,left_node,1,True) in Potential_wall_list:
                    plt.plot([i, i], [j, j + 1], linewidth=3, color=[0.5, 0.5, 0.5], linestyle='dashed')
                if ((new_node, left_node, 1, False) in Edge_list) or ((new_node, left_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j > 0:
                down_node = 'w' + str(i) + 'h' + str(j - 1)
                if (new_node,down_node,1,True) in Potential_wall_list:
                    plt.plot([i, i+1], [j, j], linewidth=3, color=[0.5, 0.5, 0.5], linestyle='dashed')
                
                if ((new_node, down_node, 1, False) in Edge_list) or ((new_node, down_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if i < width - 1:
                right_node = 'w' + str(i + 1) + 'h' + str(j)
                if (new_node,right_node,1,True) in Potential_wall_list:
                    plt.plot([i+1, i+1], [j, j + 1], linewidth=3, color=[0.5, 0.5, 0.5], linestyle='dashed')
                
                if ((new_node, right_node, 1, False) in Edge_list) or ((new_node, right_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])
            if j < height - 1:
                up_node = 'w' + str(i) + 'h' + str(j + 1)
                if (new_node,up_node,1,True) in Potential_wall_list:
                    plt.plot([i, i+1], [j+1, j + 1], linewidth=3, color=[0.5, 0.5, 0.5], linestyle='dashed')
                
                if ((new_node, up_node, 1, False) in Edge_list) or ((new_node, up_node) in Obstacle_edge_list):
                    color = np.array([1., 1., 1.])


            node_id = np.ravel_multi_index([i, j], [width, height])
            ap_list = list(Node_list[node_id][1])

            if ap_list[0] is None:
                color = color
            else:
                color = np.array([0., 0., 0.])
                for ap_name in ap_list:
                    ap_i = ap_names.index(ap_name)
                    color += random_colors[ap_i, :]
                color /= len(ap_list)

            plt.gca().add_patch(
                plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=color, edgecolor=None, fill=True,
                              linewidth=0))
    for node in Initial_list:
        i = int(node[1])
        j = int(node[3])
        plt.gca().add_patch(
            plt.Rectangle(xy=(i, j), width=1, height=1, facecolor=[0, 0, 1], edgecolor=None, fill=True,
                          linewidth=0))
    plt.savefig("random_environment.png")
    #plt.show()

""" node_phi = ['q0', 'q1', 'q2']
edge_phi = [('q0', 'q0', [[]], [['flag']]), ('q0', 'q1', [['flag']], [[]]), ('q1', 'q2', [[]], [[]]),
            ('q2', 'q2', [[]], [[]])]
initial_phi = ['q0']
acc_phi = ['q1']

width = 10
height = 10
automaton = DFA(node_phi, edge_phi, initial_phi, acc_phi)

node_list, edge_list, initial_list, potential_obstacles_id, potential_Walls = gen_random_system_task_specialized(
    width, height,
    ["flag"],
    [1], 2,
    30, 1,
    1, 2, automaton)

potential_obstacles = list()
for o in potential_obstacles_id:
    potential_obstacles.append(node_list[o][0])

#get obs
potential_obstacles_id = []
for node in potential_obstacles:
    potential_obstacles_id.append(np.ravel_multi_index([int(node[1]), int(node[3])],[width,height]))
pro = [1,0]
pro_po = get_obstacles_probability(potential_obstacles_id,pro)
obs = get_obstacles(edge_list, pro_po,width, height)

print(node_list)
print(edge_list)
print(potential_obstacles)

print(obs)
plot_system_wall(node_list, edge_list, potential_Walls, obs, initial_list, width,height, ["flag"])
plot_determined_system_wall(node_list,edge_list,potential_Walls,obs,initial_list,width,height,['flag'])
 """