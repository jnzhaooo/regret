import time
import heapq
import networkx as nx
import numpy as np
import pandas as pd
import json
from networkx import DiGraph
import pickle
from nx_to_igraph import simplify_networkx
from basic_visualization import visual_Knowledge_Game_Arena
import itertools

# 从文件中加载自定义图
def load_kga_from_file(filename):
    with open(filename, 'rb') as f:
        kga = pickle.load(f)
    print("load file {} done.".format(filename))
    return kga

# 保存自定义图到文件
def save_kga_to_file(kga, filename):
    with open(filename, 'wb') as f:
        pickle.dump(kga, f)
    print("save file {} done.".format(filename))


def dijkstra_algorithm(graph, start_node):
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    shortest_paths = {start_node: (0, [])}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph[current_node]:
            weight = graph[current_node][neighbor]['cost']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                shortest_paths[neighbor] = (distance, shortest_paths[current_node][1] + [current_node])

    # shortest_length = 99999
    # shortest_path = None
    # for node in end_nodes:
    #     shortest_length = distances[node]
    #     shortest_path = shortest_paths[node][1]
    #     shortest_path.append(node)
    # return shortest_length, shortest_path
    for node in shortest_paths.keys():
        shortest_paths[node][1].append(node)
    return shortest_paths
    


def regret_min_strategy_synthesis(Arena,flag=None):
    
    t1 = time.process_time()
    reg_KGA = Reg_KGA(Arena)
    t2 = time.process_time()
    print(reg_KGA)

    print("time of constructing reg_KGA:", t2-t1)
    
    
    # reg-kga
    t3 = time.process_time()
    value_function, strategy_KGA,strategy_env_KGA = min_max_game(reg_KGA,flag=True)
    t4 = time.process_time()
    
    # # visualize
    # d = {}
    # d.update(strategy_KGA)      
    # d.update(strategy_env_KGA)
    visual_Knowledge_Game_Arena(reg_KGA,"reg_KGA.png", strategy_env_KGA)#
    print("reg-KGA visualized.")

    print("time of solving minmax in reg-KGA:", t4-t3)
    
    initial_info = list(reg_KGA.graph['initial'])[0]
    print("strategy synthesized!")
    print("regret =", value_function[initial_info])
    
    if flag==True:
        return strategy_KGA, value_function[initial_info]
    else:
        return strategy_KGA
    
def regret_min_strategy_synthesis_in_simple_reg_kga(Arena,flag=None):
    
    knowledge_dict = Arena.knowledge_dict
    best_responses_dict = Arena.best_responses_dict
    simple_kga, node_mapping,_ = simplify_networkx(Arena)
    t1 = time.process_time()
    reg_KGA = Reg_KGA_simple(simple_kga, node_mapping, best_responses_dict, knowledge_dict)
    # reg_KGA  =Reg_Tree(Arena)
    t2 = time.process_time()
    print(reg_KGA)

    print("time of constructing reg_KGA:", t2-t1)
    
    #visual_Knowledge_Game_Arena(reg_KGA,"reg_KGA.png")#

    # reg-kga
    t3 = time.process_time()
    value_function, strategy_KGA = min_max_game(reg_KGA)
    t4 = time.process_time()

    print("time of solving minmax in reg-KGA:", t4-t3)
    
    initial_info = list(reg_KGA.graph['initial'])[0]
    print("strategy synthesized!")
    print("regret =", value_function[initial_info])
    
    if flag==True:
        return strategy_KGA, value_function[initial_info]
    else:
        return strategy_KGA
    

def min_max_game(graph, flag = None):
    # 初始化每个节点的初始值和策略
    value_initial = {node: 0 if node in graph.graph['acc'] else float('inf') for node in graph.nodes}
    strategy_initial = {node: 'stop' if node in graph.graph['acc'] else None for node in graph.nodes}

    value_next = value_initial.copy()
    strategy_next = strategy_initial.copy()
    strategy_env_next = {node: None for node in graph.graph['env']}  # 初始化 env 策略

    # 计算初始的最大值 (env) 和最小值 (agent)
    for node in graph.graph['env']:
        cost_at_node = [
            (neighbor, value_initial[neighbor] + graph.edges[(node, neighbor)]['cost'])
            for neighbor in graph.successors(node)
        ]
        # 选择最大代价的后继节点
        best_neighbor, max_cost = max(cost_at_node, key=lambda x: x[1])
        value_next[node] = max_cost
        strategy_env_next[node] = best_neighbor  # 记录 env 的策略

    for node in graph.graph['agent']:
        if node in graph.graph['acc']:
            value_next[node] = 0
            strategy_next[node] = 'stop'
        else:
            cost_at_node = float('inf')
            next_node = None
            for neighbor in graph.successors(node):
                cost = value_initial[neighbor] + graph.edges[(node, neighbor)]['cost']
                if cost < cost_at_node:
                    cost_at_node = cost
                    next_node = neighbor
            value_next[node] = cost_at_node
            strategy_next[node] = next_node

    value_current = value_initial.copy()
    strategy_current = strategy_initial.copy()

    k = 1
    max_iterations = len(graph.nodes)  # 最大迭代次数
    while k <= max_iterations:
        updated = False  # 用于跟踪是否有值更新
        value_current = value_next.copy()
        strategy_current = strategy_next.copy()

        # 更新 env 节点的值和策略
        for node in graph.graph['env']:
            cost_at_node = [
                (neighbor, value_current[neighbor] + graph.edges[(node, neighbor)]['cost'])
                for neighbor in graph.successors(node)
            ]
            best_neighbor, max_cost = max(cost_at_node, key=lambda x: x[1])
            if max_cost != value_next[node]:
                value_next[node] = max_cost
                strategy_env_next[node] = best_neighbor  # 记录 env 的策略
                updated = True

        # 更新 agent 节点的值和策略
        for node in graph.graph['agent']:
            if node in graph.graph['acc']:
                value_next[node] = 0
                strategy_next[node] = 'stop'
            else:
                cost_at_node = value_current[node]
                next_node = strategy_current[node]
                for neighbor in graph.successors(node):
                    cost = value_current[neighbor] + graph.edges[(node, neighbor)]['cost']
                    if cost < cost_at_node:
                        cost_at_node = cost
                        next_node = neighbor
                if cost_at_node != value_next[node]:
                    value_next[node] = cost_at_node
                    strategy_next[node] = next_node
                    updated = True

        # 如果在本次迭代中没有任何值更新，则提前终止
        if not updated:
            break

        k += 1
    if(flag):
        return value_next, strategy_next, strategy_env_next
    else:
        return value_next, strategy_next

def min_max_game_igraph(graph):

    # 获取全局属性
    acc_nodes = graph['acc']
    env_nodes = graph['env']
    agent_nodes = graph['agent']
    initial_nodes = graph['initial']
    
    # 初始化每个节点的初始值和策略
    # print(v.label for v in graph.vs)
    value_initial = {v.index: 0 if v.index in acc_nodes else float('inf') for v in graph.vs}
    strategy_initial = {v.index: 'stop' if v.index in acc_nodes else None for v in graph.vs}


    value_next = value_initial.copy()
    strategy_next = strategy_initial.copy()

    # 计算初始的最大值 (env) 和最小值 (agent)
    for v_index in env_nodes:
        cost_at_node = [
            value_initial[neighbor] + graph.es[graph.get_eid(v_index, neighbor)]['cost']
            for neighbor in graph.successors(v_index)
        ]
        value_next[v_index] = max(cost_at_node)

    for v in agent_nodes:
        
        if v in acc_nodes:
            value_next[v] = 0
            strategy_next[v] = 'stop'
        else:
            cost_at_node = float('inf')
            next_node = None
            for neighbor in graph.successors(v):
                cost = value_initial[neighbor] + graph.es[graph.get_eid(v, neighbor)]['cost']
                if cost < cost_at_node:
                    cost_at_node = cost
                    next_node = neighbor
            value_next[v] = cost_at_node
            strategy_next[v] = next_node

    value_current = value_initial.copy()
    strategy_current = strategy_initial.copy()

    k = 1
    max_iterations = len(graph.vs)  # 最大迭代次数
    while k <= max_iterations:
        updated = False  # 用于跟踪是否有值更新
        value_current = value_next.copy()
        strategy_current = strategy_next.copy()

        # 更新 env 节点的值
        for v in env_nodes:

            cost_at_node = [
                value_current[neighbor] + graph.es[graph.get_eid(v, neighbor)]['cost']
                for neighbor in graph.successors(v)
            ]
            max_cost = max(cost_at_node)
            if max_cost != value_next[v]:
                value_next[v] = max_cost
                updated = True

        # 更新 agent 节点的值
        for v in agent_nodes:
            if v in acc_nodes:
                value_next[v] = 0
                strategy_next[v] = 'stop'
            else:
                cost_at_node = value_current[v]
                next_node = strategy_current[v]
                for neighbor in graph.successors(v):
                    # print(neighbor,v)
                    cost = value_current[neighbor] + graph.es[graph.get_eid(v, neighbor)]['cost']
                    if cost < cost_at_node:
                        cost_at_node = cost
                        next_node = neighbor
                if cost_at_node != value_next[v]:
                    value_next[v] = cost_at_node
                    strategy_next[v] = next_node
                    updated = True

        # 如果在本次迭代中没有任何值更新，则提前终止
        if not updated:
            break
        
        k += 1
    print("Iteration: ", k)
    return value_next, strategy_next



class Reg_KGA(DiGraph):
    def __init__(self, Arena):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())
        for edge in Arena.edges:
            self.add_edge(edge[0], edge[1])

        self.knowledge_dict = Arena.knowledge_dict
        initial_node = list(Arena.graph['initial'])[0]
        self.graph['initial'].add(initial_node)

        for node in self.nodes:
            if node in Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node in Arena.graph['acc']:
                self.graph['acc'].add(node)
                
                
        edges_in_shortest_path = set()
        shortest_paths = dijkstra_algorithm(Arena, initial_node)
        for acc_node in self.graph['acc']:
            shortest_path = shortest_paths[acc_node][1]
            for i in range(len(shortest_path)-1):
                edges_in_shortest_path.add((shortest_path[i], shortest_path[i+1]))

        for edge in self.edges:
            if edge[1] in Arena.graph['env']:
                self.edges[edge]['cost'] = 0
            else:
                if (edge[0],edge[1]) not in edges_in_shortest_path:
                    self.edges[edge]['cost'] = 9999
                else:
                    if edge[1] not in Arena.graph['acc']:
                        self.edges[edge]['cost'] = 0
                    else:

                        length = shortest_paths[edge[1]][0]
                
                        difference = length - Arena.best_responses_dict[edge[1][2]]
                        
                        self.edges[edge]['cost'] = difference
                        
      
class Reg_KGA_simple(DiGraph):
    def __init__(self, Arena, node_mapping, best_responses_dict, knowledge_dict):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())
        for edge in Arena.edges:
            self.add_edge(edge[0], edge[1])
        node_mapping_inverse = {v : k for k, v in node_mapping.items()}

        self.knowledge_dict = knowledge_dict
        initial_node = list(Arena.graph['initial'])[0]
        self.graph['initial'].add(initial_node)

        for node in self.nodes:
            if node in Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node in Arena.graph['acc']:
                self.graph['acc'].add(node)

                        
        edges_in_shortest_path = set()
        shortest_paths = dijkstra_algorithm(Arena, initial_node)
        for acc_node in self.graph['acc']:
            shortest_path = shortest_paths[acc_node][1]
            for i in range(len(shortest_path)-1):
                edges_in_shortest_path.add((shortest_path[i], shortest_path[i+1]))

        for edge in self.edges:
            if edge[1] in Arena.graph['env']:
                self.edges[edge]['cost'] = 0
            else:
                if (edge[0],edge[1]) not in edges_in_shortest_path:
                    self.edges[edge]['cost'] = 9999
                else:
                    if edge[1] not in Arena.graph['acc']:
                        self.edges[edge]['cost'] = 0
                    else:

                        length = shortest_paths[edge[1]][0]
                
                        difference = length - best_responses_dict[node_mapping_inverse[edge[1]][2]]
                        
                        self.edges[edge]['cost'] = difference
                        
   
def best_case_strategy_synthesis(Arena):
    strategy = dict()
    for node in Arena.graph['agent']:
        if node in Arena.graph['acc']:
            strategy[node] = 'stop'
        else:
            path = Dijkstra(Arena, node, Arena.graph['acc'])[1]
            if path:
                strategy[node] = path[1]
    return strategy


def Dijkstra(graph, start_node, end_nodes):
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    shortest_paths = {start_node: (0, [])}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph[current_node]:
            weight = graph[current_node][neighbor]['cost']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                shortest_paths[neighbor] = (distance, shortest_paths[current_node][1] + [current_node])

    shortest_length = 99999
    shortest_path = None
    for node in end_nodes:
        if node in shortest_paths:
            if shortest_paths[node][0] < shortest_length:
                shortest_length = shortest_paths[node][0]
                shortest_path = shortest_paths[node][1]
    return shortest_length, shortest_path


class Reg_Tree(DiGraph):
    def __init__(self, KGA):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())
        
        initial_node = list(KGA.graph['initial'])[0]
        self.graph['initial'].add(initial_node)

        shortest_paths = dijkstra_algorithm(KGA, initial_node)
        for node in KGA.nodes:
            if len(list(KGA.successors(node))) == 0:
                
                shortest_path = shortest_paths[node][1]
                for i in range(len(shortest_path)-1):
                        self.add_edge(shortest_path[i], shortest_path[i+1])
                if node in KGA.graph['acc']:
                    self.graph['acc'].add(node)
            
        for edge in self.edges:
            if edge[1] in self.graph['acc']:
                length = shortest_paths[edge[1]][0]
                difference = length - KGA.best_responses_dict[edge[1][2]]
                self.edges[edge]['cost'] = difference
            else:
                if len(list(self.successors(edge[1]))) == 0:
                    self.edges[edge]['cost'] = 9999
                else:
                    self.edges[edge]['cost'] = 0
        
        
        self.knowledge_dict = dict()
        for node in self.nodes:
            self.knowledge_dict[node[2]] = KGA.knowledge_dict[node[2]]
            if node in KGA.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)


class Reg_Tree_simple(DiGraph):
    def __init__(self, KGA, node_mapping, best_responses_dict, knowledge_dict):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())
        node_mapping_inverse = {v : k for k, v in node_mapping.items()}

        
        initial_node = list(KGA.graph['initial'])[0]
        self.graph['initial'].add(initial_node)

        shortest_paths = dijkstra_algorithm(KGA, initial_node)
        for node in KGA.nodes:
            if len(list(KGA.successors(node))) == 0:
                
                shortest_path = shortest_paths[node][1]
                for i in range(len(shortest_path)-1):
                        self.add_edge(shortest_path[i], shortest_path[i+1])
                if node in KGA.graph['acc']:
                    self.graph['acc'].add(node)
            
        for edge in self.edges:
            if edge[1] in self.graph['acc']:
                length = shortest_paths[edge[1]][0]
                # difference = length - KGA.best_responses_dict[edge[1][2]]
                difference = length - best_responses_dict[node_mapping_inverse[edge[1]][2]]
                self.edges[edge]['cost'] = difference
            else:
                if len(list(self.successors(edge[1]))) == 0:
                    self.edges[edge]['cost'] = 9999
                else:
                    self.edges[edge]['cost'] = 0
        
        
        self.knowledge_dict = dict()
        for node in self.nodes:
            node_id = node_mapping_inverse[node]
            self.knowledge_dict[node_id[2]] = knowledge_dict[node_id[2]]
            if node in KGA.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)





def regret_min_strategy_synthesis_in_tree(PK_WTS, Arena, flag=None): 
    knowledge_dict = Arena.knowledge_dict
    best_responses_dict = Arena.best_responses_dict
    simple_kga, node_mapping,_ = simplify_networkx(Arena)
    t1 = time.process_time()
    reg_KGA = Reg_Tree_simple(simple_kga, node_mapping, best_responses_dict, knowledge_dict)
    
    t2 = time.process_time()
    print(reg_KGA)

    print("time of constructing reg_KGA:", t2-t1)
    
    #visual_Knowledge_Game_Arena(reg_KGA,"reg_KGA.png")#

    # reg-kga
    t3 = time.process_time()
    value_function, strategy_KGA = min_max_game(reg_KGA)
    t4 = time.process_time()

    print("time of solving minmax in reg-KGA:", t4-t3)
    
    initial_info = list(reg_KGA.graph['initial'])[0]
    print("strategy synthesized!")
    print("regret =", value_function[initial_info])
    
    if flag==True:
        return strategy_KGA, value_function[initial_info]
    else:
        return strategy_KGA
    

class Reg_Tree_with_env_strategy(DiGraph):
    def __init__(self, Reg_Tree, env_strategy):
        DiGraph.__init__(self, initial=set(), acc=set())

        initial_node = list(Reg_Tree.graph['initial'])[0]
        self.graph['initial'].add(initial_node)

        todo_list = list()
        already_set = set()
        todo_list.append(initial_node)

        while todo_list:
            node = todo_list.pop()
            if node in Reg_Tree.graph['agent']:
                for successor in Reg_Tree.successors(node):
                    self.add_edge(node, successor)
            else:
                n = env_strategy[node]
                self.add_edge(node, n)

            for neighbor in self.successors(node):
                if node not in already_set:
                    todo_list.append(neighbor)
            already_set.add(node)
        
        for node in self.nodes:
            if node in Reg_Tree.graph['acc']:
                self.graph['acc'].add(node)

        for edge in self.edges:
            self.edges[edge]['cost'] = Reg_Tree.edges[edge]['cost']


