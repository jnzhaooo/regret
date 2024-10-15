from networkx.classes.digraph import DiGraph
from itertools import chain, combinations
import heapq

def WTS_of_PK_WTS(pk_wts, edge_list):
    generator = WTS(pk_wts, edge_list)
    return generator

class WTS(DiGraph):
    def __init__(self, pk_wts, edge_list):
        DiGraph.__init__(self, initial=set())
        self.graph['initial'].add(list(pk_wts.graph['initial'])[0])
        for node in pk_wts.nodes:
            self.add_node(node)
            self.nodes[node]['label'] = pk_wts.nodes[node]['label']
            
        for edge in pk_wts.edges:
            if pk_wts.edges[edge]['uncertainty'] == False:
                self.add_edge(edge[0], edge[1], cost=pk_wts.edges[edge]['cost'])
            else:
                if edge in edge_list:
                    self.add_edge(edge[0], edge[1], cost=pk_wts.edges[edge]['cost'])
                    self.add_edge(edge[1], edge[0], cost=pk_wts.edges[edge]['cost'])

def run_strategy_in_wts(wts, KGA, strategy_KGA):
    actual_path = list()
    actual_cost = 0

    initial_node = list(KGA.graph['initial'])[0]
    current_agent_node = initial_node
    current_position = current_agent_node[0]
    actual_path.append(current_position)

    decision = strategy_KGA[current_agent_node]


    while decision != 'stop':
        current_position = decision[3]
        transition_cost = KGA.edges[(current_agent_node, decision)]['cost']
        actual_successors = set(wts.successors(current_position))
        actual_cost += transition_cost
        actual_path.append(current_position)

        for next_agent_node in KGA.successors(decision):
            successors_at_current_position = set()
            knowledge = next_agent_node[2]
            for edge in knowledge[1]:
                if edge[0] == current_position:
                    successors_at_current_position.add(edge[1])
            if successors_at_current_position == actual_successors:
                current_agent_node = next_agent_node

        if current_agent_node:
            decision = strategy_KGA[current_agent_node]
        else:
            print("error!")
            break
            
    return actual_path, actual_cost


class PK_WTS(DiGraph):
    def __init__(self, node_list, edge_list, initial_list):
        DiGraph.__init__(self, initial=set())
        for item in node_list:
            self.add_node(item[0])
            self.nodes[item[0]]['label'] = item[1]

        for item in initial_list:
            self.graph['initial'].add(item)

        for item in edge_list:
            self.add_edge(item[0], item[1], cost=item[2], uncertainty=item[3])
            if item[3] == True:
                self.add_edge(item[1], item[0], cost=item[2], uncertainty=item[3])


        nodes_set = set(self.nodes)
        for node in nodes_set:
            uncertain_successors = set()
            certain_successors = set()
            for neighbor in self.successors(node):
                if self.edges[(node, neighbor)]['uncertainty'] == True:
                    uncertain_successors.add(neighbor)
                else:
                    certain_successors.add(neighbor)

            successor_patterns = list()
            subset_of_uncertain_succesors = powerset(uncertain_successors)
            for item in subset_of_uncertain_succesors:
                successors_item = set()
                for neighbor in item:
                    successors_item.add(neighbor)
                for neighbor_known in certain_successors:
                    successors_item.add(neighbor_known)
                successor_patterns.append(tuple(successors_item))
            self.nodes[node]['successor_patterns'] = successor_patterns

    

class DFA(DiGraph):
    def __init__(self, node_list, edge_list, initial_list, acc_list):
        DiGraph.__init__(self, initial=set(), acc=set())
        for item in node_list:
            self.add_node(item)
        for item in initial_list:
            self.graph['initial'].add(item)
        for item in acc_list:
            self.graph['acc'].add(item)

        for item in edge_list:
            self.add_edge(item[0], item[1], legal=item[2], illegal=item[3])

def product(WTS, DFA):
    generator = ProductSystem(WTS, DFA)
    return generator


class ProductSystem(DiGraph):
    def __init__(self, WTS, DFA):
        DiGraph.__init__(self, initial=set(), acc=set())
        WTS_init = list(WTS.graph['initial'])[0]
        DFA_init = list(DFA.graph['initial'])[0]
  
        for q_next in DFA.successors(DFA_init):
            if check_legal_transition(DFA, DFA_init, q_next, WTS.nodes[WTS_init]['label']):
                initial_node = (WTS_init, q_next)


        self.graph['initial'].add(initial_node)
        todo_list = list()
        already_set = set()

        todo_list.append(initial_node)

        while (len(todo_list)!=0):
            node = todo_list.pop()
            x = node[0]
            q = node[1]
            for q_next in DFA.successors(q):
                for x_next in WTS.successors(x):
                    label = WTS.nodes[x_next]['label']
                    if check_legal_transition(DFA, q, q_next, label):
                        self.add_edge((x,q), (x_next, q_next), cost=WTS.edges[(x, x_next)]['cost'])
                        continue
            for neighbor in self.successors((x,q)):
                if neighbor not in already_set:
                    todo_list.append(neighbor)
            already_set.add((x,q))

        for node in self.nodes:
            if node[1] in DFA.graph['acc']:
                self.graph['acc'].add(node)


def check_legal_transition(DFA, p, q, label):
    legal = DFA.edges[(p, q)]['legal']
    illegal = DFA.edges[(p, q)]['illegal']

    truth = list()
    for item in legal:
        for i in range(len(item)):
            if item[i] not in label:
                truth.append(True)
    if len(truth) != 0 and all(truth):
        return False

    illegal_truth = list()
    for item in illegal:
        for i in range(len(item)):
            if item[i] in label:
                illegal_truth.append(True)

    if len(illegal_truth) != 0 and all(illegal_truth):
        return False

    return True


def powerset(iterable):
    s = set(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def dijkstra_algorithm(graph, start_node, end_nodes):
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