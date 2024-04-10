from networkx.classes.digraph import DiGraph
from basic_operations import Dijskstra, shortest_path
import time
from basic_visualization import visual_pk_wts, visual_PS, visual_Knowledge_Game_Arena

def best_case_strategy_synthesis(Arena):
    strategy = dict()
    for node in Arena.graph['agent']:
        if node in Arena.graph['acc']:
            strategy[node] = 'stop'
        else:
            path = shortest_path(Arena, node, Arena.graph['acc'])
            if path:
                strategy[node] = path[1]
    return strategy
    

def regret_min_strategy_synthesis(Arena,flag=None):
    
    # theirs
    t3 = time.process_time()
    information_KGA = Information_KGA(Arena)
    print("theirs: ", information_KGA)
    
    
    
    # ours
    reg_KGA = Reg_KGA(Arena)
    
    #visual_Knowledge_Game_Arena(reg_KGA,"reg_KGA.png")#
    """ print("------reg_KGA------")
    for key,value in reg_KGA.knowledge_dict.items():
            print(value,":",key) """
    print("ours: ", reg_KGA)
    value_function, strategy_KGA = min_max_game(reg_KGA)
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
    value_initial = dict()
    strategy_initial = dict()
    for node in graph.nodes:
        if node in graph.graph['acc']:
            value_initial[node] = 0
            strategy_initial[node] = 'stop'
        else:
            value_initial[node] = 99999

    value_next = dict()
    strategy_next = dict()

    for node in graph.graph['env']:
        cost_at_node = list()
        for neighbor in graph.successors(node):
            cost_at_node.append(value_initial[neighbor] + graph.edges[(node, neighbor)]['cost'])
        value_next[node] = max(cost_at_node)
    for node in graph.graph['agent']:
        if node in graph.graph['acc']:
            value_next[node] = 0
            strategy_next[node] = 'stop'
        else:
            cost_at_node = 9999
            next_node = None
            for neighbor in graph.successors(node):
                if value_initial[neighbor] + graph.edges[(node, neighbor)]['cost'] < cost_at_node:
                    cost_at_node = value_initial[neighbor] + graph.edges[(node, neighbor)]['cost']
                    next_node = neighbor
            value_next[node] = cost_at_node
            strategy_next[node] = next_node

    value_current = value_initial

    k = 1
    while k <= len(graph.nodes):
        value_current = value_next
        strategy_current = strategy_next
        for node in graph.graph['env']:
            cost_at_node = list()
            for neighbor in graph.successors(node):
                cost_at_node.append(value_current[neighbor] + graph.edges[(node, neighbor)]['cost'])
            value_next[node] = max(cost_at_node)
        for node in graph.graph['agent']:
            if node in graph.graph['acc']:
                value_next[node] = 0
                strategy_next[node] = 'stop'
            else:
                cost_at_node = value_current[node]
                next_node = strategy_current[node]
                for neighbor in graph.successors(node):
                    if value_current[neighbor] + graph.edges[(node, neighbor)]['cost'] < cost_at_node:
                        cost_at_node = value_current[neighbor] + graph.edges[(node, neighbor)]['cost']
                        next_node = neighbor
                value_next[node] = cost_at_node
                strategy_next[node] = next_node
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
                difference = Dijskstra(Arena, initial_node, {edge[1]}) - Arena.best_responses_dict[edge[1][2]]
                path = shortest_path(Arena, initial_node, {edge[1]})
                last_second_node = path[len(path)-1]
                if edge[0] == last_second_node:
                    self.edges[edge]['cost'] = difference
                else:
                    self.edges[edge]['cost'] = 9999
                #self.edges[edge]['cost'] = Dijskstra(Arena, initial_node, {edge[1]})

class Information_KGA(DiGraph):
    def __init__(self, Arena):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())

        initial_Arena = list(Arena.graph['initial'])[0]
        initial_node = (initial_Arena, 0)
        self.graph['initial'].add(initial_node)

        todo_list = list()
        already_set = set()
        todo_list.append(initial_node)

        max_trans_cost = 0
        for edge in Arena.edges:
            trans_cost = Arena.edges[edge]['cost']
            if trans_cost > max_trans_cost:
                max_trans_cost = trans_cost
        bound = 2 * max_trans_cost * len(Arena.nodes)

        while todo_list:
            node = todo_list.pop()
            already_set.add(node)
            v = node[0]
            u = node[1]
            for v_next in Arena.successors(v):
                u_next = u + Arena.edges[(v, v_next)]['cost']
                if u_next < bound:
                    self.add_edge((v,u), (v_next, u_next))

            for neighbor in self.successors(node):
                if neighbor not in already_set:
                    todo_list.append(neighbor)

        for node in self.nodes:
            if node[0] in Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node[0] in Arena.graph['acc']:
                self.graph['acc'].add(node)

        for edge in self.edges:
            if edge[1] not in self.graph['acc']:
                self.edges[edge]['cost'] = 0
            else:
                self.edges[edge]['cost'] = edge[1][1] - Arena.best_responses_dict[edge[1][0][2]]
   




class Regret_KGA(DiGraph):
    def __init__(self, Info_Arena):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())

        initial_Info_Arena = list(Info_Arena.graph['initial'])[0]
        initial_node = (initial_Info_Arena, 9999)
        self.graph['initial'].add(initial_node)

        todo_list = list()
        already_set = set()
        todo_list.append(initial_node)

        while len(todo_list) != 0:
            node = todo_list.pop()
            already_set.add(node)

            v = node[0]
            b = node[1]

            for v_next in Info_Arena.successors(v):
                if v in Info_Arena.graph['agent']:
                    ba = Info_Arena.best_alternative((v, v_next))
                    b_next = min(b, ba)
                else:
                    b_next = b
                self.add_edge((v, b), (v_next, b_next))

            for neighbor in self.successors(node):
                if neighbor not in already_set:
                    todo_list.append(neighbor)

        for node in self.nodes:
            if node[0] in Info_Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node[0] in Info_Arena.graph['acc']:
                self.graph['acc'].add(node)

        for edge in self.edges:
            if edge[1] in self.graph['acc']:
                self.edges[edge]['cost'] = edge[1][0][1] - edge[1][1]
            else:
                self.edges[edge]['cost'] = 0
       


class Info_KGA(DiGraph):
    def __init__(self, Arena):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())

        initial_Arena = list(Arena.graph['initial'])[0]
        initial_node = (initial_Arena, 0)
        self.graph['initial'].add(initial_node)

        todo_list = list()
        already_set = set()
        todo_list.append(initial_node)

        max_trans_cost = 0
        for edge in Arena.edges:
            trans_cost = Arena.edges[edge]['cost']
            if trans_cost > max_trans_cost:
                max_trans_cost = trans_cost
        bound = 15

        while todo_list:
            node = todo_list.pop()
            already_set.add(node)
            v = node[0]
            u = node[1]
            for v_next in Arena.successors(v):
                u_next = u + Arena.edges[(v, v_next)]['cost']
                if u_next < bound:
                    self.add_edge((v,u), (v_next, u_next))

            for neighbor in self.successors(node):
                if neighbor not in already_set:
                    todo_list.append(neighbor)

        for node in self.nodes:
            if node[0] in Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node[0] in Arena.graph['acc']:
                self.graph['acc'].add(node)

        for edge in self.edges:
            if edge[1] not in self.graph['acc']:
                self.edges[edge]['cost'] = 0
            else:
                self.edges[edge]['cost'] = edge[1][1]

    def best_alternative(self, edge):
        ba = 9999
        for neighbor in self.successors(edge[0]):
            if neighbor != edge[1]:
                best = Dijskstra(self, neighbor, self.graph['acc'])
                if best < ba:
                    ba = best
        return ba
     

'''

class Aux_KGA(DiGraph):
    def __init__(self, Arena):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())

        initial_Arena = list(Arena.graph['initial'])[0]
        initial_node = (initial_Arena, tuple())
        self.graph['initial'].add(initial_node)

        todo_list = list()
        already_set = set()
        todo_list.append(initial_node)

        while len(todo_list) != 0:
            node = todo_list.pop()
            already_set.add(node)
            v = node[0]
            Y = set(node[1])
            for v_next in Arena.successors(v):
                Y_next = Y
                Y_next.add(v)
                if self.has_the_node((v_next, tuple(Y_next))):
                    self.add_edge(node, self.locate_the_node((v_next, tuple(Y_next))), cost=Arena.edges[(v, v_next)]['cost'])
                else:
                    self.add_edge(node, (v_next, tuple(Y_next)), cost=Arena.edges[(v, v_next)]['cost'])

            for neighbor in self.successors(node):
                if neighbor not in already_set:
                    todo_list.append(neighbor)
        

        for node in self.nodes:
            if node[0] in Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node[0] in Arena.graph['acc']:
                self.graph['acc'].add(node)

    def has_the_node(self, node):
        v = node[0]
        Y = node[1]
        for vertex in self.nodes:
            vv = vertex[0]
            YY = vertex[1]
            if v == vv and set(Y) == set(YY):
                return True
        return False
    
    def locate_the_node(self, node):
        v = node[0]
        Y = node[1]
        for vertex in self.nodes:
            vv = vertex[0]
            YY = vertex[1]
            if v == vv and set(Y) == set(YY):
                return vertex
        print("no such node")
        return None


class Information_KGA(DiGraph):
    def __init__(self, Aux_Arena):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())

        initial_Aux_Arena = list(Aux_Arena.graph['initial'])[0]
        initial_node = (initial_Aux_Arena, 0)
        self.graph['initial'].add(initial_node)

        todo_list = list()
        already_set = set()
        todo_list.append(initial_node)

        while len(todo_list) != 0:
            node = todo_list.pop()
            already_set.add(node[0])
            v = node[0]
            u = node[1]
            for v_next in Aux_Arena.successors(v):
                trans_cost = Aux_Arena.edges[(v, v_next)]['cost']
                self.add_edge(node, (v_next, trans_cost+u))

            for neighbor in self.successors(node):
                if neighbor[0] not in already_set:
                    todo_list.append(neighbor)

        for node in self.nodes:
            if node[0] in Aux_Arena.graph['agent']:
                self.graph['agent'].add(node)
            else:
                self.graph['env'].add(node)
            if node[0] in Aux_Arena.graph['acc']:
                self.graph['acc'].add(node)

        for edge in self.edges:
            if edge[1] in self.graph['acc']:
                self.edges[edge]['cost'] = edge[1][1]
            else:
                self.edges[edge]['cost'] = 0

    def best_response(self, edge):
        br = 9999
        for neighbor in self.successors(edge[0]):
            best = Dijskstra(self, neighbor, self.graph['acc'])
            if best < br:
                br = best
        return br
'''
                   


'''
    knowledge_dict = dict()
    knowledge_index = 0
    for node in Arena.nodes:
        if node[2] not in knowledge_dict:
            knowledge_dict[node[2]] = 'K' + str(knowledge_index)
            knowledge_index += 1

    Y_dict = dict()
    for vertex in aux_KGA.nodes:
        Y = vertex[1]
        Y_map = set()
        for node in set(Y):
            if node in Arena.graph['agent']:
                Y_map.add((node[0], node[1], knowledge_dict[node[2]]))
            else:
                Y_map.add((node[0], node[1], knowledge_dict[node[2]], node[3]))
        Y_dict[Y] = tuple(Y_map)

    for vertex in list(aux_KGA.nodes):
        print(vertex)
        print('\n')        

    for vertex in list(aux_KGA.nodes):
        if vertex[0] in Arena.graph['agent']:
            print(((vertex[0][0], vertex[0][1], knowledge_dict[vertex[0][2]]), Y_dict[vertex[1]]))
        else:
            print(((vertex[0][0], vertex[0][1], knowledge_dict[vertex[0][2]], vertex[0][3]), Y_dict[vertex[1]]))
'''    
