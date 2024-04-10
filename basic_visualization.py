import networkx as nx
import pylab
import matplotlib.pyplot as plt

def visual_pk_wts(pk_wts):
    pos1=nx.circular_layout(pk_wts)          # 生成圆形节点布局
    pos2=nx.random_layout(pk_wts)            # 生成随机节点布局
    pos3=nx.shell_layout(pk_wts)             # 生成同心圆节点布局
    pos4=nx.spring_layout(pk_wts)            # 利用Fruchterman-Reingold force-directed算法生成节点布局
    pos5=nx.spectral_layout(pk_wts)          # 利用图拉普拉斯特征向量生成节点布局
    pos=nx.kamada_kawai_layout(pk_wts)      #使用Kamada-Kawai路径长度代价函数生成布局

    #pos = nx.random_layout(pk_wts)
    initial_nodes = pk_wts.graph['initial']
    costs = nx.get_edge_attributes(pk_wts, "cost")
    uncertainty = nx.get_edge_attributes(pk_wts, "uncertainty")
    euncertain = [key for key,value in uncertainty.items() if value==True]
    nx.set_node_attributes(pk_wts, {node:{'label':('({},{})'.format(node,pk_wts.nodes[node]['label']))} for node in pk_wts.nodes})
    nx.set_edge_attributes(pk_wts, {k:{'label':costs[k]} for k in costs.keys()})
    G = nx.nx_agraph.to_agraph(pk_wts)
    for i in initial_nodes:
        node = G.get_node(i)
        node.attr['color']='red'
    for (u,v) in euncertain:
        edge = G.get_edge(u,v)
        edge.attr['color']='blue'

    G.draw("pk_wts.png", prog="dot")
    


def visual_PS(PS):
    
    acc_nodes = PS.graph['acc']
    initial_nodes = PS.graph['initial']
    costs = nx.get_edge_attributes(PS, "cost")
    nx.set_edge_attributes(PS, {k:{'label':costs[k]} for k in costs.keys()})
    G = nx.nx_agraph.to_agraph(PS)
    for i in initial_nodes:
        node = G.get_node(i)
        node.attr['color']='red'
    for i in acc_nodes:
        node = G.get_node(i)
        node.attr['shape']='doublecircle'

    G.draw("ProductSystem.png", prog="dot")#'dot,neato,fdp,sfdp,circo,twopi,nop,osage,patchwork,..'


def visual_Knowledge_Game_Arena(game_arena,filename=None):
    acc_nodes = game_arena.graph['acc']
    initial_nodes = game_arena.graph['initial']
    agent_nodes = game_arena.graph['agent']
    env_nodes = game_arena.graph['env']
    costs = nx.get_edge_attributes(game_arena, 'cost')
    labels = {}
    for node in game_arena.nodes:
            if node in game_arena.graph['agent']:
                labels[node]=(node[0], node[1],game_arena.knowledge_dict[node[2]])
            else:
                labels[node]=(node[0], node[1],game_arena.knowledge_dict[node[2]],node[3])

    nx.set_node_attributes(game_arena, {k:{'label':labels[k]} for k in labels.keys()})
    nx.set_edge_attributes(game_arena, {k:{'label':costs[k]} for k in costs.keys()})

    G = nx.nx_agraph.to_agraph(game_arena)
    for i in initial_nodes:
        node = G.get_node(i)
        node.attr['color']='red'
    for i in acc_nodes:
        node = G.get_node(i)
        node.attr['shape']='doublecircle'
    for i in env_nodes:
        node = G.get_node(i)
        node.attr['shape']='box'
    if filename==None:
        G.draw("Knowledge_Game_Arena.png", prog="dot")#'dot,neato,fdp,sfdp,circo,twopi,nop,osage,patchwork,..'
    else:
        G.draw(filename, prog="dot")

        