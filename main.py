from networkx.classes.digraph import DiGraph
from basic_operations import PK_WTS, DFA, product, run_strategy_in_wts, WTS_of_PK_WTS, Dijskstra, PK_WTS_Grid, shortest_path
from Knowledge_Game_Arena import Knowledge_Game_Arena
from strategy_synthesis import regret_min_strategy_synthesis, min_max_strategy_synthesis, best_case_strategy_synthesis
#from basic_visualization import visual_pk_wts, visual_PS, visual_Knowledge_Game_Arena
from KGA import KGA
from KGA_Grid import KGA_Grid
from system_generation import generate_system, plot_system
import random_system_generation
import numpy as np
from pandas import DataFrame
import time
import multiprocessing

def write_data(filename,str1,cost1,str2=None,cost2=None,str3=None, cost3=None, str4=None,cost4=None):

    data = { str1:cost1, str2:cost2, str3:cost3, str4:cost4}
    df = DataFrame(data)
    df.to_excel(filename)


node_phi = [ 'q0', 'q1', 'q2' ]
edge_phi = [ ('q0', 'q0', [[]], [['flag']]), ('q0', 'q1', [['flag']], [[]]), ('q1', 'q2', [[]], [[]]), ('q2', 'q2', [[]], [[]]) ]
initial_phi = [ 'q0' ]
acc_phi = [ 'q1' ]

automaton = DFA(node_phi, edge_phi, initial_phi, acc_phi)

def testing_1(times, probability):
    '''
    根据给定的pk-wts, 生成多个确定的wts
    '''

    cost_regret = []
    cost_minmax = []
    cost_best_case = []
    
    for i in range(times):
        #get obs
        potential_obstacles_id = []
        for node in potential_obstacles:
            potential_obstacles_id.append(np.ravel_multi_index([int(node[1]), int(node[3])],[width,height]))
        pro = probability
        pro_po = random_system_generation.get_obstacles_probability(potential_obstacles_id,pro)
        obs = random_system_generation.get_obstacles_2(edge_list, pro_po,width, height)


        wts = WTS_of_PK_WTS(pk_wts, obs)
        actual_path_regret, actual_cost_regret = run_strategy_in_wts(wts, game_arena, regret_strategy)

        print("actual_path:", actual_path_regret)
        print("actual_cost:", actual_cost_regret)

        print("----------------------")
        print("minmax strategy")

        actual_path_minmax, actual_cost_minmax = run_strategy_in_wts(wts, game_arena, minmax_strategy)

        print("actual_path:", actual_path_minmax)
        print("actual_cost:", actual_cost_minmax)



        print("----------------------")
        print("best-case strategy")

        actual_path_best_case, actual_cost_best_case = run_strategy_in_wts(wts, game_arena, best_case_strategy)

        print("actual_path:", actual_path_best_case)
        print("actual_cost:", actual_cost_best_case)

        #plot path
        #plot_determined_system_2(node_list,edge_list,obs,initial_list,width, height,['flag'],actual_path_best_case, actual_path_minmax)

        cost_regret.append(actual_cost_regret)
        cost_minmax.append(actual_cost_minmax)
        cost_best_case.append(actual_cost_best_case)

    return cost_regret, cost_minmax, cost_best_case
    

def testing_2(times,width,height,num_obstacle):
    '''
    多个pk-wts.每个pk-wts生成一个wts
    '''
    
    cost_regret = []
    cost_minmax = []
    cost_best_case = []
    
    for i in range(times):
        print("Epoch: {}/{}\n".format(i+1,times))
        t0 = time.time()
        node_list, edge_list, initial_list, potential_obstacles_id, potential_walls = random_system_generation.gen_random_system_task_specialized(
            width, height,
            ["flag"],
            [1], num_obstacle,
            45, 1,
            1, 2, automaton)


        potential_obstacles = list()
        for o in potential_obstacles_id:
            potential_obstacles.append(node_list[o][0])


        
        #pk_wts = PK_WTS_Grid(node_list, edge_list, initial_list, potential_obstacles)
        pk_wts = PK_WTS(node_list, edge_list, initial_list)
        PS = product(pk_wts, automaton)

        print(pk_wts)

        print(PS)

        game_arena = KGA(pk_wts, automaton, PS)

        #visual_Knowledge_Game_Arena(game_arena)

        print(game_arena)

        regret_strategy = regret_min_strategy_synthesis(game_arena)
        minmax_strategy = min_max_strategy_synthesis(game_arena)
        best_case_strategy = best_case_strategy_synthesis(game_arena)

        #get obs
        potential_obstacles_id = []
        for node in potential_obstacles:
            potential_obstacles_id.append(np.ravel_multi_index([int(node[1]), int(node[3])],[width,height]))
        pro = []
        for obs in potential_obstacles:
            pro.append(np.random.rand(1))
        pro_po = random_system_generation.get_obstacles_probability(potential_obstacles_id,pro)
        obs = random_system_generation.get_obstacles(edge_list, pro_po,width, height)
        random_system_generation.plot_system_wall(node_list, edge_list,potential_walls, obs, initial_list, width,height, ["flag"])



        wts = WTS_of_PK_WTS(pk_wts, obs)
        actual_path_regret, actual_cost_regret = run_strategy_in_wts(wts, game_arena, regret_strategy)

        print("actual_path:", actual_path_regret)
        print("actual_cost:", actual_cost_regret)

        print("----------------------")
        print("minmax strategy")

        actual_path_minmax, actual_cost_minmax = run_strategy_in_wts(wts, game_arena, minmax_strategy)

        print("actual_path:", actual_path_minmax)
        print("actual_cost:", actual_cost_minmax)

        print("----------------------")
        #print("best-case strategy")

        #actual_path_best_case, actual_cost_best_case = run_strategy_in_wts(wts, game_arena, best_case_strategy)

        #print("actual_path:", actual_path_best_case)
        #print("actual_cost:", actual_cost_best_case)

        #random_system_generation.plot_determined_system_wall(node_list,edge_list,potential_walls, obs,initial_list,width,height,['flag'],actual_path_regret,actual_path_minmax)
        cost_regret.append(actual_cost_regret)
        cost_minmax.append(actual_cost_minmax)
        #cost_best_case.append(actual_cost_best_case)
        t1 = time.time()
        print("total time: ", (t1-t0))

    return cost_regret, cost_minmax, cost_best_case
    
   


#一个pk-wts生成多个wts
width = 5
height = 5
obstacles = [(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(1,2),(3,3)]
initial = [(4,2)]
pro_obstacles = [(2,2)]
pro = []
for obs in pro_obstacles:
    pro.append(np.random.rand(1))
ap = {'flag':(0,0)}
""" node_list,edge_list,initial_list,potential_obstacles = generate_system(width, height, initial,ap,obstacles,pro_obstacles)
#plot_system(node_list, edge_list, potential_obstacles, initial_list, width, height, ['flag'])
pk_wts = PK_WTS_Grid(node_list, edge_list, initial_list, potential_obstacles)
PS = product(pk_wts, automaton)
print(pk_wts)
print(PS)
#game_arena = Knowledge_Game_Arena(pk_wts, automaton, PS)
game_arena = KGA_Grid(pk_wts, automaton, PS)
#visual_Knowledge_Game_Arena(game_arena)
print(game_arena)
regret_strategy = regret_min_strategy_synthesis(game_arena)
minmax_strategy = min_max_strategy_synthesis(game_arena)
best_case_strategy = best_case_strategy_synthesis(game_arena)
cost_regret,cost_minmax,cost_best_case = testing_1(1,pro) """
#write_data('cost_1.xlsx','cost_regret',cost_regret,'cost_minmax',cost_minmax,'cost_best_case',cost_best_case)



#多个pk-wts.每个pk-wts生成一个wts
#proba = [0.5]
cost_regret,cost_minmax,cost_best_case = testing_2(1,width=10,height=10,num_obstacle=2)

#write_data('cost_2.xlsx','cost_regret',cost_regret,'cost_minmax',cost_minmax,'cost_best_case',cost_best_case)

ave_regret = sum(cost_regret)/len(cost_regret)
ave_minmax = sum(cost_minmax)/len(cost_minmax)
ave_best_case = 0
#ave_best_case = sum(cost_best_case)/len(cost_best_case)

print("ave_regret: {}\n ave_minmax: {}\n ave_best_case: {}\n".format(ave_regret,ave_minmax,ave_best_case))

