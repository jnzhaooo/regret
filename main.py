from basic_operations import DFA, product, run_strategy_in_wts, WTS_of_PK_WTS
from KGA import KGA
from generate_pkwts import random_generate_pk_wts_repeat_game,best_case_strategy, random_generate_pk_wts_multi_label
from new_operations import min_max_game,regret_min_strategy_synthesis, regret_min_strategy_synthesis_in_simple_reg_kga,best_case_strategy_synthesis,Dijkstra, regret_min_strategy_synthesis_in_tree 
import time
import numpy as np
from nx_to_igraph import simplify_networkx
from basic_visualization import visual_pk_wts
from new_operations import save_kga_to_file, load_kga_from_file
import os
import plot_box



def testing_3(times,num_nodes, pro):
    cost_regret = []
    cost_minmax = []
    cost_best_case = []
    cost_best_case_ = []
    minmax_cost =[]
    REG = []
    file = './data/data_KGA/KGA_{}_2'.format(num_nodes)
    sub_files = [os.path.join(file, f) for f in os.listdir(file)]
    pk_wts_path = './data/data_pkwts/pkwts_{}_2'.format(num_nodes)
    pkwts_files = [os.path.join(pk_wts_path, f) for f in os.listdir(pk_wts_path)]
    for i in range(len(sub_files)):
        if(i>times):
            break
        print("epoch:{}/{}".format(i+1,times))
        t0 = time.time()
        print(pro)

        # load from file
        
        game_arena = load_kga_from_file(sub_files[i])
        pk_wts = load_kga_from_file(pkwts_files[i])
        
        print(pk_wts)

        potential_edge_list = []
        for po in pk_wts.graph['uncer_edges']:
            if np.random.rand(1) > pro:
                potential_edge_list.append(po)
        # potential_edge_list = []
        print("potential_edge_list: ", potential_edge_list)
        unexist_edges = list(set(pk_wts.graph['uncer_edges']) - set(potential_edge_list))
        wts = WTS_of_PK_WTS(pk_wts, potential_edge_list)
        
        print(game_arena)

        t1 = time.time()
        regret_strategy,reg = regret_min_strategy_synthesis_in_simple_reg_kga(game_arena,True)  
        t2 = time.time()
        kga_simple, node_dict, node_dict_inverse = simplify_networkx(game_arena)
        res, minmax_strategy = min_max_game(kga_simple)
        t3 = time.time()
        try:
            actual_path_best_case, actual_cost_best_case = best_case_strategy(pk_wts,wts, pk_wts.graph['label_nodes'][1],pk_wts.graph['label_nodes'][2])
        except:
            continue
        # best_case_strategy_ = best_case_strategy_synthesis(game_arena)
        t4 = time.time()

        # remapping the strategy
        regret_strategy_node = dict()
        minmax_strategy_node = dict()
        for (k,v) in regret_strategy.items():
            if v==None:
                regret_strategy_node[node_dict_inverse[k]] = None
            elif v=='stop':
                regret_strategy_node[node_dict_inverse[k]] = 'stop'
            else:
                regret_strategy_node[node_dict_inverse[k]] = node_dict_inverse[v]
        for (k,v) in minmax_strategy.items():
            if v==None:
                minmax_strategy_node[node_dict_inverse[k]] = None
            elif v=='stop':
                minmax_strategy_node[node_dict_inverse[k]] = 'stop'
            else:
                minmax_strategy_node[node_dict_inverse[k]] = node_dict_inverse[v]

        actual_path_regret, actual_cost_regret = run_strategy_in_wts(wts, game_arena, regret_strategy_node)
        t5 = time.time()
        print("actual_path:", actual_path_regret)
        print("actual_cost:", actual_cost_regret)

        print("----------------------")
        print("minmax strategy")

        actual_path_minmax, actual_cost_minmax = run_strategy_in_wts(wts, game_arena, minmax_strategy_node)
        t6 = time.time()
        print("actual_path:", actual_path_minmax)
        print("actual_cost:", actual_cost_minmax)
        REG.append(reg)
        print("----------------------")
        print("best_case strategy")
        print("actual_path:", actual_path_best_case)
        print("actual_cost:", actual_cost_best_case)
        t7 = time.time()



        print("regret time: ", t2-t1+t5-t4)
        print("minmax time: ", t3-t2+t6-t5)
        print("best_case time: ", t4-t3+t7-t6)
        print("total time: ", t7-t0)
        

        cost_regret.append(actual_cost_regret)
        cost_minmax.append(actual_cost_minmax)
        cost_best_case.append(actual_cost_best_case)
        # cost_best_case_.append(actual_cost_best_case_)
        print("difference: ", [(cost_regret[i]-cost_minmax[i]) for i in range(len(cost_regret))])
        print("regret: ", REG)
        print("bast case: ",cost_best_case)
        print("bast case_: ",cost_best_case_)

        ave_regret = sum(cost_regret)/len(cost_regret)
        ave_minmax = sum(cost_minmax)/len(cost_minmax)
        ave_best_case = sum(cost_best_case)/len(cost_best_case)
    
        print("ave_regret: {}\n ave_minmax: {}\n ave_best_case: {}\n".format(ave_regret,ave_minmax,ave_best_case))

    return cost_regret, cost_minmax, cost_best_case

n_node = 20

y1,y2,y3 = testing_3(100,n_node,0)

# save data
np.savetxt('./result/cost_regret_0.txt', y1)
np.savetxt('./result/cost_minmax_0.txt', y2)
np.savetxt('./result/cost_best_case_0.txt', y3)

# #plot
# plot_box.plot_violin(y1,y2,y3,'box_0.png')

y1,y2,y3 = testing_3(100,n_node,0.2)

# save data
np.savetxt('./result/cost_regret_2.txt', y1) 
np.savetxt('./result/cost_minmax_2.txt', y2)
np.savetxt('./result/cost_best_case_2.txt', y3)


# #plot
# plot_box.plot_violin(y1,y2,y3,'box_2.png')

y1,y2,y3 = testing_3(100,n_node,0.5)

# save data
np.savetxt('./result/cost_regret_5.txt', y1)
np.savetxt('./result/cost_minmax_5.txt', y2)
np.savetxt('./result/cost_best_case_5.txt', y3)

# #plot
# plot_box.plot_violin(y1,y2,y3,'box_5.png')

y1,y2,y3 = testing_3(100,n_node,0.8)

# save data
np.savetxt('./result/cost_regret_8.txt', y1)
np.savetxt('./result/cost_minmax_8.txt', y2)
np.savetxt('./result/cost_best_case_8.txt', y3)

# #plot
# plot_box.plot_violin(y1,y2,y3,'box_8.png')

y1,y2,y3 = testing_3(100,n_node,1)

# save data
np.savetxt('./result/cost_regret_10.txt', y1)
np.savetxt('./result/cost_minmax_10.txt', y2)
np.savetxt('./result/cost_best_case_10.txt', y3)

# #plot
# plot_box.plot_violin(y1,y2,y3,'box_10.png')
