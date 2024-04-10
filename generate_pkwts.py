import networkx as nx
import random
from itertools import chain, combinations
from basic_operations import PK_WTS, DFA, product, run_strategy_in_wts, WTS_of_PK_WTS, Dijskstra, PK_WTS_Grid
from strategy_synthesis import regret_min_strategy_synthesis, min_max_strategy_synthesis, best_case_strategy_synthesis
from basic_visualization import visual_pk_wts, visual_PS, visual_Knowledge_Game_Arena
from KGA import KGA
import time
import numpy as np

f = open("./a.log","w+")

def random_generate_pk_wts(num_node, num_edge, num_initial,num_uncertain, ap, edge_cost, num_isolated):
    #random_generate_pk_wts(num_node=20, num_edge=50,num_initial=1,num_uncertain=5,ap={'flag':5,'fire':3},edge_cost=1)

    G = nx.DiGraph() 
    path_nodes = []
    cnt1 = 0
    num_path_node = num_node - num_isolated
    #add node
    for i in range(num_node):
        name = 'x' + str(i)
        G.add_node(name)
        cnt1 = cnt1+1
        if cnt1<num_path_node:
            path_nodes.append(name)

    # initial node
    
    G.graph['initial'] = []
    initial_nodes = random.sample(path_nodes, num_initial)
    G.graph['initial'] = initial_nodes

    """     # add edge and make sure "strongly connected"
    start_name='x0'
    cnt = 0
    #注释掉
    for i in range(1,num_node):
        node_name = 'x' + str(i)
        G.add_edge(start_name, node_name)
   
    while True:
        u_name = random.choice(list(G.nodes()))
        v_name = random.choice(list(G.nodes()))

        if u_name!=v_name and not G.has_edge(u_name,v_name):
            if nx.has_path(G,v_name,u_name):  
                G.add_edge(u_name,v_name)
                cnt = cnt+1
            else:
                G.add_edge(u_name,v_name)
                G.add_edge(v_name,u_name)
                cnt = cnt+2
            if cnt==num_edge:
                break
            else:
                continue
            
    print("Strongly connected!")
    """

    for i in range(0,num_path_node-1):
        G.add_edge(('x'+str(i)),('x'+str(i+1)))
    G.add_edge(('x'+str(num_path_node-1)),('x'+str(0)))

    # process nodes: randomly select nodes, add labels 
    for node in G.nodes:
        G.nodes[node]['label'] = 'None'
    for (name,n_ap) in ap.items():
        nodes_ = random.sample(range(num_path_node), n_ap)
        nodes = []
        for node in nodes_:
            nodes.append('x'+str(node))
        for node in nodes:
            G.nodes[node]['label'] = name

    #process edges: add cost and uncertainty and other edges
    for u,v in G.edges():
        G[u][v]['cost'] = edge_cost
    for u,v in G.edges():
        G[u][v]['uncertainty'] = False   
    cnt = num_edge-num_path_node
    cnt2 = 0
    uncertain_edges = []
    while(cnt > 0):
         u = random.choice(list(path_nodes))
         v = random.choice(list(G.nodes()))
         if (u != v) and ((u,v) not in G.edges):
             G.add_edge(u,v)
             G[u][v]['cost'] = edge_cost
             if cnt2 < num_uncertain:
                cnt2 = cnt2+1
                uncertain_edges.append((u,v))
                if (v,u) not in G.edges:
                    G.add_edge(v,u)
                    cnt = cnt-2
                    G[v][u]['cost'] = edge_cost
                    G[v][u]['uncertainty'] = True
                    G[u][v]['uncertainty'] = True
                else:
                    cnt = cnt-1
                    G[v][u]['uncertainty'] = True
                    G[u][v]['uncertainty'] = True
             else:
                 cnt = cnt-1
                 G[u][v]['uncertainty'] = False
            


    """ uncertain_edges = random.sample(list(G.edges()), num_uncertain)
    for u,v in uncertain_edges:
        G[u][v]['uncertainty'] = True
        if (v,u) not in G.edges:
            G.add_edge(v,u)
            G[v][u]['cost'] = edge_cost
            G[v][u]['uncertainty'] = True
        else:
            G[v][u]['uncertainty'] = True
     """

    #process node:add successor_patterns
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

    for node in G:
        uncertain_successors = set()
        certain_successors = set()
        
        for neighbor in G.neighbors(node):
            if G.edges[(node, neighbor)]['uncertainty']:
                uncertain_successors.add(neighbor)
            else: 
                certain_successors.add(neighbor)
                
        successor_patterns = []
        subset = powerset(uncertain_successors) 
        
        for items in subset:
            successors = set()
            for neighbor in items:
                successors.add(neighbor)
            for neighbor in certain_successors: 
                successors.add(neighbor)
            successor_patterns.append(tuple(successors))
                
        G.nodes[node]['successor_patterns'] = successor_patterns

    
    #print(G)
    #visual_pk_wts(G)
    return G, uncertain_edges

#在用版本（在最短路上加不确定，并调整两条通路的cost差）
def random_generate_pk_wts_generalize(num_node,num_initial,num_potential,ap, out_min, out_max,cost_min,cost_max):
    flag = True
    while(flag):
        flag=False
        G = nx.DiGraph()
        unproc_nodes = list()
        proced_nodes = list()
        nodes = list()
        for i in range(num_node):
            node = 'x'+str(i+1)
            nodes.append(node)
            unproc_nodes.append(node)
            G.add_node(node,label='None')

        while(len(unproc_nodes)>0):
            node = unproc_nodes.pop()
            e = random.randint(out_min,out_max)
            while(e>0):
                next = random.choice(nodes)
                if next==node:
                    continue
                c = random.randint(cost_min,cost_max)
                G.add_edge(node,next,cost=c,uncertainty=False)
                e = e-1
            proced_nodes.append(node)
        
        G.graph['initial'] = []
        initial_nodes = random.sample(list(G.nodes), num_initial)
        G.graph['initial'] = initial_nodes

        for (name,n_ap) in ap.items():
            nodes_ = random.sample(nodes, n_ap)
            for node in nodes_:
                G.nodes[node]['label'] = name

        if (nx.has_path(G,initial_nodes[0], nodes_[0])==False):
            flag = True
            continue

        s_path = nx.dijkstra_path(G, initial_nodes[0], nodes_[0], weight='cost')
        s_lenth = nx.dijkstra_path_length(G, initial_nodes[0], nodes_[0], weight='cost')
        if len(s_path)<=4:
            flag=True
            continue
        uncer_nodes = []
        cnt = 0
        while(cnt < num_potential):
            un_node_id = random.randint(1,len(s_path)-3)
            next_node_id = un_node_id+1
            un_node = s_path[un_node_id]
            next_node = s_path[next_node_id]
         
            if ((un_node in uncer_nodes) or (next_node in uncer_nodes)):
                continue
            if (un_node,next_node) not in uncer_nodes:
                uncer_nodes.append((un_node,next_node))
                cnt = cnt+1 
            if (un_node,next_node) in G.edges:
                G.remove_edge(un_node,next_node)
            if (next_node,un_node) in G.edges:
                G.remove_edge(next_node,un_node)
            
        if (nx.has_path(G,initial_nodes[0], nodes_[0])==False):
            flag = True
            continue
        h_path = nx.dijkstra_path(G, initial_nodes[0], nodes_[0], weight='cost')
        h_lenth = nx.dijkstra_path_length(G, initial_nodes[0], nodes_[0], weight='cost')
        """ if (h_lenth - s_lenth) < 200  :
            flag = True
            continue
        else:
            print("The difference between two path: ", h_lenth-s_lenth) """
        for (un_node,next_node) in uncer_nodes:
            c = random.randint(cost_min,cost_max)
            G.add_edge(un_node,next_node,cost=c,uncertainty=True)
            G.add_edge(next_node,un_node,cost=c,uncertainty=True)

        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

        for node in G:
            uncertain_successors = set()
            certain_successors = set()
            
            for neighbor in G.neighbors(node):
                if G.edges[(node, neighbor)]['uncertainty']:
                    uncertain_successors.add(neighbor)
                else: 
                    certain_successors.add(neighbor)
                    
            successor_patterns = []
            subset = powerset(uncertain_successors) 
            
            for items in subset:
                successors = set()
                for neighbor in items:
                    successors.add(neighbor)
                for neighbor in certain_successors: 
                    successors.add(neighbor)
                successor_patterns.append(tuple(successors))
                    
            G.nodes[node]['successor_patterns'] = successor_patterns
            
        #print(s_path)
        #print(h_path)
        if nx.is_directed_acyclic_graph(G):
            flag=True
            print("not DAG")
            continue
        #print(G_)
    #visual_pk_wts(G)
    """ for node in G.nodes:
        print(node,G.nodes[node]['successor_patterns']) """
    pro_best_case_path = s_path
    return G,uncer_nodes, pro_best_case_path

#完全随机版本
def sheer_random_generate_pk_wts_generalize(num_node,num_initial,num_potential,ap, out_min, out_max,cost_min,cost_max):
    flag = True
    while(flag):
        flag = False
        G = nx.DiGraph()
        unproc_nodes = list()
        proced_nodes = list()
        nodes = list()
        for i in range(num_node):
            node = 'x'+str(i+1)
            nodes.append(node)
            unproc_nodes.append(node)
            G.add_node(node,label='None')

        while(len(unproc_nodes)>0):
            node = unproc_nodes.pop()
            e = random.randint(out_min,out_max)
            while(e>0):
                next = random.choice(nodes)
                if next==node:
                    continue
                c = random.randint(cost_min,cost_max)
                G.add_edge(node,next,cost=c,uncertainty=False)
                e = e-1
            proced_nodes.append(node)
        
        G.graph['initial'] = []
        initial_nodes = random.sample(list(G.nodes), num_initial)
        G.graph['initial'] = initial_nodes

        for (name,n_ap) in ap.items():
            nodes_ = random.sample(nodes, n_ap)
            for node in nodes_:
                G.nodes[node]['label'] = name

        uncer_nodes = []
        uncer_nodes = random.sample(list(G.edges), num_potential)
        for (un_node,next_node) in uncer_nodes:
            if (un_node,next_node) in G.edges:
                G.remove_edge(un_node,next_node)

        if (nx.has_path(G,initial_nodes[0], nodes_[0])==False):
            flag = True
            continue
        for (un_node,next_node) in uncer_nodes: 
            c = random.randint(cost_min,cost_max)
            G.add_edge(un_node,next_node,cost=c,uncertainty=True)
            G.add_edge(next_node,un_node,cost=c,uncertainty=True)

        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

        for node in G:
            uncertain_successors = set()
            certain_successors = set()
            
            for neighbor in G.neighbors(node):
                if G.edges[(node, neighbor)]['uncertainty']:
                    uncertain_successors.add(neighbor)
                else: 
                    certain_successors.add(neighbor)
                    
            successor_patterns = []
            subset = powerset(uncertain_successors) 
            
            for items in subset:
                successors = set()
                for neighbor in items:
                    successors.add(neighbor)
                for neighbor in certain_successors: 
                    successors.add(neighbor)
                successor_patterns.append(tuple(successors))
                    
            G.nodes[node]['successor_patterns'] = successor_patterns
            
            #print(s_path)
            #print(h_path)
        
        
    #print(G)
    #visual_pk_wts(G)
    """ for node in G.nodes:
        print(node,G.nodes[node]['successor_patterns']) """
    pro_best_case_path = nx.dijkstra_path(G, initial_nodes[0], nodes_[0], weight='cost')    
    return G,uncer_nodes,pro_best_case_path

#加的edge均为双向版本
def random_generate_pk_wts_generalize_bi(num_node,num_initial,num_potential,ap, out_min, out_max,cost_min,cost_max):
    flag = True
    while(flag):
        flag=False
        G = nx.DiGraph()
        unproc_nodes = list()
        proced_nodes = list()
        nodes = list()
        for i in range(num_node):
            node = 'x'+str(i+1)
            nodes.append(node)
            unproc_nodes.append(node)
            G.add_node(node,label='None')

        while(len(unproc_nodes)>0):
            node = unproc_nodes.pop()
            e = random.randint(out_min,out_max)
            while(e>0):
                next = random.choice(nodes)
                if next==node:
                    continue
                c = random.randint(cost_min,cost_max)
                G.add_edge(node,next,cost=c,uncertainty=False)
                G.add_edge(next,node,cost=c,uncertainty=False)
                e = e-1
            proced_nodes.append(node)
        
        G.graph['initial'] = []
        initial_nodes = random.sample(list(G.nodes), num_initial)
        G.graph['initial'] = initial_nodes

        for (name,n_ap) in ap.items():
            nodes_ = random.sample(nodes, n_ap)
            for node in nodes_:
                G.nodes[node]['label'] = name

        if (nx.has_path(G,initial_nodes[0], nodes_[0])==False):
            flag = True
            continue

        s_path = nx.dijkstra_path(G, initial_nodes[0], nodes_[0], weight='cost')
        if len(s_path)<=4:
            flag=True
            continue
        uncer_nodes = []
        cnt = 0
        while(cnt < num_potential):
            un_node_id = random.randint(1,len(s_path)-3)
            next_node_id = un_node_id+1
            un_node = s_path[un_node_id]
            next_node = s_path[next_node_id]
         
            if ((un_node in uncer_nodes) or (next_node in uncer_nodes)):
                continue
            if (un_node,next_node) not in uncer_nodes:
                uncer_nodes.append((un_node,next_node))
                cnt = cnt+1
            if (un_node,next_node) in G.edges:
                G.remove_edge(un_node,next_node)
            if (next_node,un_node) in G.edges:
                G.remove_edge(next_node,un_node)
            
        if (nx.has_path(G,initial_nodes[0], nodes_[0])==False):
            flag = True
            continue
        h_path = nx.dijkstra_path(G, initial_nodes[0], nodes_[0], weight='cost')
        for (un_node,next_node) in uncer_nodes:
            c = random.randint(cost_min,cost_max)
            G.add_edge(un_node,next_node,cost=c,uncertainty=True)
            G.add_edge(next_node,un_node,cost=c,uncertainty=True)

        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

        for node in G:
            uncertain_successors = set()
            certain_successors = set()
            
            for neighbor in G.neighbors(node):
                if G.edges[(node, neighbor)]['uncertainty']:
                    uncertain_successors.add(neighbor)
                else: 
                    certain_successors.add(neighbor)
                    
            successor_patterns = []
            subset = powerset(uncertain_successors) 
            
            for items in subset:
                successors = set()
                for neighbor in items:
                    successors.add(neighbor)
                for neighbor in certain_successors: 
                    successors.add(neighbor)
                successor_patterns.append(tuple(successors))
                    
            G.nodes[node]['successor_patterns'] = successor_patterns
            
        #print(s_path)
        #print(h_path)
        if nx.is_directed_acyclic_graph(G):
            flag=True
            print("not DAG")
            continue
        #print(G_)
    #visual_pk_wts(G)
    """ for node in G.nodes:
        print(node,G.nodes[node]['successor_patterns']) """
    pro_best_case_path = s_path
    return G,uncer_nodes,pro_best_case_path

def best_case_strategy(PK_WTS: nx.DiGraph, WTS: nx.DiGraph ,pro_best_case_path: list):
    start_node = pro_best_case_path[0]
    end_node = pro_best_case_path[-1]
    next_node = pro_best_case_path[0]
    actual_best_case_path = []
    actual_best_case_cost = 0
    actual_best_case_path.append(start_node)
    flag = True
    new_path = pro_best_case_path
    i = 0
    while(flag):
        next_node = new_path[i+1]
        if (start_node,next_node) in WTS.edges:
            actual_best_case_path.append(next_node)
            actual_best_case_cost = actual_best_case_cost + WTS.edges[(start_node,next_node)]['cost']
            start_node = next_node
            i = i + 1
        else:
            PK_WTS.remove_edge(start_node,next_node)
            if nx.has_path(PK_WTS,start_node,end_node):
                new_path = nx.dijkstra_path(PK_WTS, start_node, end_node, weight='cost')
                i = 0
            else:
                print("ERROR!")
                actual_best_case_cost = 999
                break
                start_node = pro_best_case_path[0]
                new_path = nx.dijkstra_path(PK_WTS, start_node, end_node, weight='cost')
                next_node = pro_best_case_path[0]
                actual_best_case_path = []
                actual_best_case_cost = 0
                actual_best_case_path.append(start_node)
                i = 0
        if start_node == end_node:
            flag = False
            
    return actual_best_case_path,actual_best_case_cost




#random_generate_pk_wts_generalize(num_node=20,num_initial=1,num_potential=1,ap={'flag':1},out_min=2,out_max=3,cost_min=1,cost_max=5)


""" node_list = [
            ('x1', {'flag'}), ('x2', {None}), ('x3', {None}), ('x4', {None}),('x5', {None})
            ]
edge_list = [
            ('x1', 'x2', 56, False), ('x2', 'x1', 90, False), ('x1', 'x5', 90, False),
            ('x2', 'x3', 16, False), ('x2', 'x5', 24, True), 
            ('x3', 'x1', 70, False), ('x3', 'x5', 84, False),
            ('x4', 'x5', 50, False), 
            ('x5', 'x3', 63, False), ('x5', 'x2', 24, True)
            ]
initial_list = ['x4']
potential_list = [ ('x2','x5'),('x5','x2') ] """

#pk_wts = PK_WTS(node_list,edge_list,initial_list)

node_phi = ['q0', 'q1', 'q2']
edge_phi = [('q0', 'q0', [[]], [['flag']]), ('q0', 'q1', [['flag']], [[]]), ('q1', 'q2', [[]], [[]]),
            ('q2', 'q2', [[]], [[]])]
initial_phi = ['q0']
acc_phi = ['q1']

K1 = (('x4', 'x3', 'x1', 'x5'), 
      (('x5', 'x3'), ('x2', 'x3'), ('x3', 'x1'), ('x4', 'x5'), 
       ('x2', 'x1'), ('x1', 'x2'), ('x3', 'x5'), ('x1', 'x5')))


automaton = DFA(node_phi, edge_phi, initial_phi, acc_phi)
def testing_3(times,num_nodes, pro):
    cost_regret = []
    cost_minmax = []
    cost_best_case = []
    cost_best_case_ = []
    REG = []
    for i in range(times):
        print("epoch:{}/{}".format(i+1,times))
        t0 = time.time()
        #pk_wts, potential_list=random_generate_pk_wts(num_node=25, num_edge=25,num_initial=1,num_uncertain=2,ap={'flag':1},edge_cost=1, num_isolated=10)
        pk_wts, potential_list,pro_best_case_path=random_generate_pk_wts_generalize(num_node=num_nodes,num_initial=1,num_potential=1  ,ap={'flag':1},out_min=1,out_max=2,cost_min=1,cost_max=5)
        #pk_wts, potential_list=sheer_random_generate_pk_wts_generalize(num_node=50,num_initial=1,num_potential=2,ap={'flag':1},out_min=1,out_max=2,cost_min=1,cost_max=100)
        PS = product(pk_wts, automaton)

        #potential_edge_list = random.sample(potential_list, random.randint(0,len(potential_list)))
        potential_edge_list = []
        for po in potential_list:
            if np.random.rand(1) > pro:
                potential_edge_list.append(po)
        print("potential_edge_list: ", potential_edge_list)
        wts = WTS_of_PK_WTS(pk_wts, potential_edge_list)
        #wts = WTS_of_PK_WTS(pk_wts, [])
        print(pk_wts)
        #visual_pk_wts(pk_wts)#
        print(PS)

        ps = product(wts, automaton)
        initial_node = list(ps.graph['initial'])[0]
        

        #game_arena = KGA_Grid(pk_wts, automaton, PS)
        game_arena = KGA(pk_wts, automaton, PS)
        #visual_Knowledge_Game_Arena(game_arena)

        print(game_arena)

        t1 = time.time()
        regret_strategy,reg = regret_min_strategy_synthesis(game_arena,True)
        t2 = time.time()
        minmax_strategy = min_max_strategy_synthesis(game_arena)
        t3 = time.time()
        actual_path_best_case, actual_cost_best_case = best_case_strategy(pk_wts,wts,pro_best_case_path)
        #best_case_strategy_ = best_case_strategy_synthesis(game_arena)
        t4 = time.time()


        actual_path_regret, actual_cost_regret = run_strategy_in_wts(wts, game_arena, regret_strategy)
        t5 = time.time()
        print("actual_path:", actual_path_regret)
        print("actual_cost:", actual_cost_regret)

        print("----------------------")
        print("minmax strategy")

        actual_path_minmax, actual_cost_minmax = run_strategy_in_wts(wts, game_arena, minmax_strategy)
        t6 = time.time()
        print("actual_path:", actual_path_minmax)
        print("actual_cost:", actual_cost_minmax)
        REG.append(reg)
        print("----------------------")
        print("best_case strategy")
        """ try:
            actual_path_best_case_, actual_cost_best_case_ = run_strategy_in_wts(wts, game_arena, best_case_strategy_)
        except:
            continue """
        print("actual_path:", actual_path_best_case)
        print("actual_cost:", actual_cost_best_case)
        t7 = time.time()


        if actual_cost_regret == 0:
            i = i - 1
            continue
        print("regret time: ", t2-t1+t5-t4)
        print("minmax time: ", t3-t2+t6-t5)
        print("best_case time: ", t4-t3+t7-t6)
        print("total time: ", t7-t0)
        cost_regret.append(actual_cost_regret)
        cost_minmax.append(actual_cost_minmax)
        cost_best_case.append(actual_cost_best_case)
        #cost_best_case_.append(actual_cost_best_case_)
        print("difference: ", [(cost_regret[i]-cost_minmax[i]) for i in range(len(cost_regret))])
        print("regret: ", REG)
        #visual_Knowledge_Game_Arena(game_arena,"KGA.png")#
        #visual_PS(PS)#
        """ print("------KGA------")
        for key,value in game_arena.knowledge_dict.items():
            print(value,":",key) """
        ave_regret = sum(cost_regret)/len(cost_regret)
        ave_minmax = sum(cost_minmax)/len(cost_minmax)
        ave_best_case = sum(cost_best_case)/len(cost_best_case)
    
        print("ave_regret: {}\n ave_minmax: {}\n ave_best_case: {}\n".format(ave_regret,ave_minmax,ave_best_case))

    return cost_regret, cost_minmax, cost_best_case



""" testing_3(100,0.2)
time.sleep(10)
testing_3(100,0.4)
time.sleep(10)
testing_3(100,0.6)
time.sleep(10) """
testing_3(1,15,0.5)
testing_3(1,20,0.5)
testing_3(1,30,0.5)
testing_3(1,50,0.5)
testing_3(1,80,0.5)
testing_3(1,100,0.5)
#time.sleep(10)
#testing_3(100,1)
#sheer_random_generate_pk_wts_generalize(num_node=20,num_initial=1,num_potential=2,ap={'flag':1},out_min=2,out_max=3,cost_min=1,cost_max=5)

