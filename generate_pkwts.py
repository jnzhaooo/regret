import networkx as nx
import random
from itertools import chain, combinations
from basic_operations import PK_WTS, DFA, product, run_strategy_in_wts, WTS_of_PK_WTS
from strategy_synthesis import regret_min_strategy_synthesis, min_max_strategy_synthesis, best_case_strategy_synthesis
from basic_visualization import visual_pk_wts, visual_PS, visual_Knowledge_Game_Arena
from KGA import KGA
import time
import numpy as np
import json

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


def save_graph_to_json(G, filename):
    data = nx.node_link_data(G)  # 将图转换为 JSON 格式的数据结构
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_graph_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    G = nx.node_link_graph(data)  # 从 JSON 格式的数据结构恢复图
    return G

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
