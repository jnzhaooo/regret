from networkx.classes.digraph import DiGraph
from basic_operations import dijkstra_algorithm
import time
from basic_visualization import visual_pk_wts, visual_PS, visual_Knowledge_Game_Arena

def best_case_strategy_synthesis(Arena):
    strategy = dict()
    for node in Arena.graph['agent']:
        if node in Arena.graph['acc']:
            strategy[node] = 'stop'
        else:
            path = dijkstra_algorithm(Arena, node, Arena.graph['acc'])[1]
            if path:
                strategy[node] = path[1]
    return strategy
    

def regret_min_strategy_synthesis(Arena,flag=None):
    
    t1 = time.process_time()
    reg_KGA = Reg_KGA(Arena)
    t2 = time.process_time()
    print(reg_KGA)

    print("time of constructing reg_KGA:", t2-t1)
    
    #visual_Knowledge_Game_Arena(reg_KGA,"reg_KGA.png")#


    t3 = time.process_time()
    value_function, strategy_KGA = min_max_game(reg_KGA)
    t4 = time.process_time()

    print("time of solving minmax:", t4-t3)

    initial_info = list(reg_KGA.graph['initial'])[0]
    print("strategy synthesized!")
    print("regret =", value_function[initial_info])
    if flag==True:
        return strategy_KGA, value_function[initial_info]
    else:
        return strategy_KGA

def min_max_strategy_synthesis(Arena):
    value, strategy = min_max_game(Arena)
    return strategy


def min_max_game(graph):
    # 初始化每个节点的初始值和策略
    value_initial = {node: 0 if node in graph.graph['acc'] else float('inf') for node in graph.nodes}
    strategy_initial = {node: 'stop' if node in graph.graph['acc'] else None for node in graph.nodes}

    value_next = value_initial.copy()
    strategy_next = strategy_initial.copy()

    # 计算初始的最大值 (env) 和最小值 (agent)
    for node in graph.graph['env']:
        cost_at_node = [
            value_initial[neighbor] + graph.edges[(node, neighbor)]['cost']
            for neighbor in graph.successors(node)
        ]
        value_next[node] = max(cost_at_node)

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

        # 更新 env 节点的值
        for node in graph.graph['env']:
            cost_at_node = [
                value_current[neighbor] + graph.edges[(node, neighbor)]['cost']
                for neighbor in graph.successors(node)
            ]
            max_cost = max(cost_at_node)
            if max_cost != value_next[node]:
                value_next[node] = max_cost
                updated = True

        # 更新 agent 节点的值
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

        for edge in self.edges:
            if edge[1] not in self.graph['acc']:
                self.edges[edge]['cost'] = 0
            else:
                length, path = dijkstra_algorithm(Arena, initial_node, {edge[1]}) 
                difference = length - Arena.best_responses_dict[edge[1][2]]
                last_second_node = path[len(path)-1]
                if edge[0] == last_second_node:
                    self.edges[edge]['cost'] = difference
                else:
                    self.edges[edge]['cost'] = 9999