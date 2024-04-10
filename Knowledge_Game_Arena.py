from networkx.classes.digraph import DiGraph
from basic_operations import Dijskstra, product
from basic_operations import WTS_of_PK_WTS

class Knowledge_Game_Arena(DiGraph):
    def __init__(self, PK_WTS, DFA, PS):
        DiGraph.__init__(self, initial=set(), agent=set(), env=set(), acc=set())

        initial_of_PS = list(PS.graph['initial'])[0]

        initial_known_transitions = set()
        for edge in PK_WTS.edges:
            if PK_WTS.edges[edge]['uncertainty'] == False:
                initial_known_transitions.add(edge)

        initial_explored_states = set()

        for node in PK_WTS.nodes:
            if len(PK_WTS.nodes[node]['successor_patterns']) < 2:
                initial_explored_states.add(node)
        
        initial_knowledge = (tuple(initial_explored_states), tuple(initial_known_transitions))

        agent_nodes = set()
        env_nodes = set()

        initial_node = (initial_of_PS[0], initial_of_PS[1], initial_knowledge)
        self.graph['initial'].add(initial_node)
        agent_nodes.add(initial_node)
    
        todo_list = list()
        already_set = set()

        todo_list.append(initial_node)


        while len(todo_list) != 0:
            node = todo_list.pop()
            already_set.add(node)
            if node in agent_nodes:
                x = node[0]
                q = node[1]
                knowledge = node[2]
                for x_next in PK_WTS.successors(x):
                    if (x, x_next) in set(knowledge[1]) and self.legal_transition(PS, (x,q), x_next):
                        flag, vertex = self.has_the_node((x,q,knowledge,x_next))
                        if flag:
                            self.add_edge((x,q,knowledge), vertex, cost=PK_WTS.edges[(x, x_next)]['cost'])
                        else:
                            self.add_edge((x,q,knowledge), (x,q,knowledge,x_next), cost=PK_WTS.edges[(x, x_next)]['cost'])
                            env_nodes.add((x,q,knowledge,x_next))
            else:
                x = node[0]
                q = node[1]
                knowledge = node[2]
                x_next = node[3]
                for neighbor in list(PS.successors((x,q))):
                    if neighbor[0] == x_next:
                        q_next = neighbor[1]
                if x_next in set(knowledge[0]):
                    flag, vertex = self.has_the_node((x_next, q_next, knowledge))
                    if flag:
                        self.add_edge((x,q,knowledge,x_next), vertex, cost=0)
                    else:
                        self.add_edge((x,q,knowledge,x_next), (x_next, q_next, knowledge), cost=0)
                        agent_nodes.add((x_next,q_next,knowledge))
                else:
                    for successor_pattern in PK_WTS.nodes[x_next]['successor_patterns']:
                        explored_states = set(knowledge[0])
                        explored_states.add(x_next)
                        known_transitions = set(knowledge[1])
                        for x_next_next in set(successor_pattern):
                            if x_next_next not in explored_states:
                                observation_set = {(x_next, x_next_next), (x_next_next, x_next)}
                                known_transitions = known_transitions | observation_set
                        knowledge_next = (tuple(explored_states), tuple(known_transitions))
                        flag, vertex = self.has_the_node((x_next, q_next, knowledge_next))
                        if flag:
                            self.add_edge((x,q,knowledge,x_next), vertex, cost=0)
                        else:
                            self.add_edge((x,q,knowledge,x_next), (x_next, q_next, knowledge_next), cost=0)
                            agent_nodes.add((x_next,q_next,knowledge_next))
                            
            for neighbor in self.successors(node):
                if neighbor not in already_set:
                    todo_list.append(neighbor)

        for node in agent_nodes:
            self.graph['agent'].add(node)
            if (node[0], node[1]) in PS.graph['acc']:
                self.graph['acc'].add(node)

        for node in env_nodes:
            self.graph['env'].add(node)

        self.knowledge_dict = dict()
        self.generate_knowledge_dict()
        self.best_responses_dict = dict()
        self.generate_best_responses_dict(PK_WTS, DFA)

    def best_response(self, PK_WTS, DFA, knowledge):
        wts = WTS_of_PK_WTS(PK_WTS, knowledge[1])
        ps = product(wts, DFA)

        initial_node = list(ps.graph['initial'])[0]
        return Dijskstra(ps, initial_node, ps.graph['acc'])
        

    def generate_best_responses_dict(self, PK_WTS, DFA):
        knowledges_list = list(self.knowledge_dict.keys())
        for knowledge in knowledges_list:
            self.best_responses_dict[knowledge] = self.best_response(PK_WTS, DFA, knowledge)
        

    def legal_transition(self, PS, node, x_next):
        for neighbor in PS.successors(node):
            if neighbor[0] == x_next:
                return True
        return False

    def print_nodes(self):
        for node in self.nodes:
            if node in self.graph['agent']:
                print((node[0], node[1], self.knowledge_dict[node[2]]))
            else:
                print((node[0], node[1], self.knowledge_dict[node[2]], node[3]))

    def print_edges(self):
        for edge in self.edges:
            cost = self.edges[edge]['cost']
            if edge[0] in self.graph['agent']:
                print(((edge[0][0], edge[0][1], self.knowledge_dict[edge[0][2]]), (edge[1][0], edge[1][1], self.knowledge_dict[edge[1][2]], edge[1][3])), cost)
            else:
                print(((edge[0][0], edge[0][1], self.knowledge_dict[edge[0][2]], edge[0][3]), (edge[1][0], edge[1][1], self.knowledge_dict[edge[1][2]])), cost)

    def generate_knowledge_dict(self):
        knowledge_index = 0
        for node in self.nodes:
            if node[2] not in self.knowledge_dict:
                self.knowledge_dict[node[2]] = 'K' + str(knowledge_index)
                knowledge_index += 1

    def has_the_node(self, node):
        if len(node) == 3:
            x = node[0]
            q = node[1]
            knowledge = node[2]
            for vertex in self.nodes:
                if len(vertex) == 3:
                    xx = vertex[0]
                    qq = vertex[1]
                    kknowledge = vertex[2]
                    if x == xx and q == qq and set(knowledge[0]) == set(kknowledge[0]) and set(knowledge[1]) == set(kknowledge[1]):
                        return True, vertex
            return False, None
        else:
            x = node[0]
            q = node[1]
            knowledge = node[2]
            x_next = node[3]
            for vertex in self.nodes:
                if len(vertex) == 4:
                    xx = vertex[0]
                    qq = vertex[1]
                    kknowledge = vertex[2]
                    xx_next = vertex[3]
                    if x == xx and q == qq and set(knowledge[0]) == set(kknowledge[0]) and set(knowledge[1]) == set(kknowledge[1]) and x_next == xx_next:
                        return True, vertex
            return False, None

                        
