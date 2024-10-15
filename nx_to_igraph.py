import igraph as ig
from networkx.classes.digraph import DiGraph
import networkx as nx

def convert_networkx_to_igraph(nx_digraph:DiGraph):
    # 创建一个 igraph 的空有向图
    ig_graph = ig.Graph(directed=True)

    # 添加所有节点
    node_mapping = {node: idx for idx, node in enumerate(nx_digraph.nodes)}
    ig_graph.add_vertices(len(nx_digraph.nodes))
    # 将节点的属性添加到 igraph
    for node, node_data in nx_digraph.nodes(data=True):
        for attr_key, attr_value in node_data.items():
            ig_graph.vs[node_mapping[node]][attr_key] = attr_value
    

    # 添加所有边
    for edge in nx_digraph.edges(data=True):
        
        source, target = edge[0], edge[1]
        ig_graph.add_edges([(node_mapping[source], node_mapping[target])])

        # 如果有额外的边属性，也将其添加到 igraph 中
        for attr_key, attr_value in edge[2].items():
            ig_graph.es[-1][attr_key] = attr_value



    # 将图的全局属性从 networkx 添加到 igraph
    for attr_key, attr_value in nx_digraph.graph.items():
        attr_value_mapping = []
        for a in attr_value:
            attr_value_mapping.append(node_mapping[a])
        ig_graph[attr_key] = attr_value_mapping
    ig_graph.graph = ig_graph
    ig_graph.nodes = ig_graph.vs
    ig_graph.edges = ig_graph.es
    return ig_graph

def simplify_networkx(graph:DiGraph):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
    node_mapping_inverse = {idx: node for (node,idx) in node_mapping.items()}
    G = nx.DiGraph(graph)
    G = nx.relabel_nodes(G, node_mapping)
    for attr_key, attr_value in G.graph.items():
        attr_value_mapping = []
        for a in attr_value:
            attr_value_mapping.append(node_mapping[a])
        G.graph[attr_key] = attr_value_mapping
    return G, node_mapping, node_mapping_inverse