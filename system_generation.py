import matplotlib.pyplot as plt
import numpy as np

def generate_system(width,height,initial,ap,obstacles,pro_obstacles,pro_list):
    '''
    e.g.:

    input:

    width = 3
    height = 3
    obstacles = [(1,2),(1,1)]
    initial = [(0,0)]
    pro_obstacles = [(0,1)]
    ap = {'flag':(2,0)}

    return:

    node_list:  [('x2y0', {'flag'})]
    edge_list:  [('x0y0', 'x1y0', 1, False), ('x0y0', 'x0y1', 1, True), ('x0y1', 'x0y0', 1, True), ('x0y1', 'x0y2', 1, True), ('x0y2', 'x0y1', 1, True), ('x1y0', 
    'x0y0', 1, False), ('x1y0', 'x2y0', 1, False), ('x2y0', 'x1y0', 1, False), ('x2y0', 'x2y1', 1, False), ('x2y1', 'x2y0', 1, False), ('x2y1', 'x2y2', 1, False), ('x2y2', 'x2y1', 1, False)]
    initial_list:  ['x0y0']
    potential_list:  ['x0y1']
    '''

    node_list = []
    ini_list = []
    obs_list = []
    edge_list = []
    potential_list = []
    obs_list = []

    for (x,y) in obstacles:
        obs_list.append('x{}y{}'.format(x,y))

    for (x,y) in initial:
        ini_list.append('x{}y{}'.format(x,y))

    for (x,y) in pro_obstacles:
        potential_list.append('x{}y{}'.format(x,y))

    def getDictKey_1(myDict, value):
        return [k for k, v in myDict.items() if value in v]

    def add_edge(node_1, node_2):
        if ((node_1 not in obs_list) and (node_2 not in obs_list)):
            if (node_1 in potential_list or node_2 in potential_list):
                edge_list.append((node_1, node_2, 1, True))
                
            else:
                edge_list.append((node_1, node_2, 1, False))
        else:
            return        
        return
    def in_ap(i,j):
        flag = False
        for v in ap.values():
            if (i,j) in v:
                flag = True
                return flag
        return flag
    for i in range(width):
            for j in range(height):
                new_node = 'x' + str(i) + 'y' + str(j)
                if in_ap(i,j):
                    print(i,j)
                    ap_name = getDictKey_1(ap,(i,j))
                    node_list.append(('x{}y{}'.format(i,j), {ap_name[0]}))
                else:
                    node_list.append(('x{}y{}'.format(i,j), {None}))
                                
                if i > 0:
                    left_node = 'x' + str(i - 1) + 'y' + str(j)
                    add_edge(new_node,left_node)
                if j > 0:
                    down_node = 'x' + str(i) + 'y' + str(j - 1)
                    add_edge(new_node,down_node)
                if i < width - 1:
                    right_node = 'x' + str(i + 1) + 'y' + str(j)
                    add_edge(new_node,right_node)
                if j < height - 1:
                    up_node = 'x' + str(i) + 'y' + str(j + 1)
                    add_edge(new_node,up_node)
    for (u,v) in pro_list:
        edge_list.append((u, v, 1, True))
    
    return node_list,edge_list,ini_list,potential_list




def plot_system(Node_list, Edge_list, Potential_list, Initial_list, width, height,
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
            new_node = 'x' + str(i) + 'y' + str(j)
            if i > 0:
                left_node = 'x' + str(i - 1) + 'y' + str(j)
                if ((new_node, left_node, 1, False) in Edge_list) or ((new_node, left_node, 1, True) in Edge_list):
                    color = np.array([1., 1., 1.])
            if j > 0:
                down_node = 'x' + str(i) + 'y' + str(j - 1)
                if ((new_node, down_node, 1, False) in Edge_list) or ((new_node, down_node, 1, True) in Edge_list):
                    color = np.array([1., 1., 1.])
            if i < width - 1:
                right_node = 'x' + str(i + 1) + 'y' + str(j)
                if ((new_node, right_node, 1, False) in Edge_list) or ((new_node, right_node, 1, True) in Edge_list):
                    color = np.array([1., 1., 1.])
            if j < height - 1:
                up_node = 'x' + str(i) + 'y' + str(j + 1)
                if ((new_node, up_node, 1, False) in Edge_list) or ((new_node, up_node, 1, True) in Edge_list):
                    color = np.array([1., 1., 1.])
            if new_node in Potential_list:
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
    
    plt.savefig("test_environment.png")
    #plt.show()
