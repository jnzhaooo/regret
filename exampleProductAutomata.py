import time

import igraph as ig
import matplotlib.pyplot as plt
import random
import numpy as np
import copy


def gen_product_automata(ts_, automata_):
    # type:(ig.Graph,ig.Graph)->ig.Graph
    n_ts = len(ts_.vs)
    n_a = len(automata_.vs)
    n_pa = n_ts * n_a
    pa_ = ig.Graph(n=n_pa, directed=True)
    for i_pa in range(n_pa):
        i_ts, i_a = np.unravel_index(i_pa, (n_ts, n_a))
        pa_.vs[i_pa]["accepting"] = automata_.vs[i_a]["accepting"]
        # i_pa_ = np.ravel_multi_index((i_ts,i_a),(n_ts,n_a))
        if len(automata_.es.select(_source=i_a)(condition_eq=ts_.vs[i_ts]["AP"])) != 0:
            continue
        for j_pa in range(n_pa):
            j_ts, j_a = np.unravel_index(j_pa, (n_ts, n_a))
            try:
                ts_eid = ts_.get_eid(i_ts, j_ts)
            except:
                continue
            try:
                a_eid = automata_.get_eid(i_a, j_a)
            except:
                if i_a == j_a and len(automata_.es.select(_source=i_a)(condition_eq=ts_.vs[j_ts]["AP"])) == 0:
                    pa_.add_edges([(i_pa, j_pa)])
                    pa_.es[-1]["cost"] = ts_.es[ts_eid]["cost"]
            else:
                if ts_.vs[j_ts]["AP"] == automata_.es[a_eid]["condition"]:
                    pa_.add_edges([(i_pa, j_pa)])
                    pa_.es[-1]["cost"] = ts_.es[ts_eid]["cost"]

    return pa_


def gen_extended_system(ts_):
    # type: (ig.Graph) -> ig.Graph
    for i_ts in range(ts.vcount()):
        ts_.vs[i_ts]["potential_posts"] = [[]]
    for i_ts in range(ts.vcount()):
        neighbors = ts_.vs[i_ts].neighbors()
        for i_neighbor in neighbors:
            ni_ts = i_neighbor.index
            eid = ts_.get_eid(i_ts, ni_ts)
            if ts_.es[eid]["uncertainty"] is True:
                temp = copy.deepcopy(ts_.vs[i_ts]["potential_posts"])
                for temp_i in range(len(temp)):
                    temp[temp_i].append(ni_ts)
                ts_.vs[i_ts]["potential_posts"].extend(temp)
            else:
                for temp_i in range(len(ts_.vs[i_ts]["potential_posts"])):
                    ts_.vs[i_ts]["potential_posts"][temp_i].append(ni_ts)
    print(ts_.vs["potential_posts"])
    new_ts_ = copy.deepcopy(ts_)
    for i_ts in range(new_ts_.vcount()):
        new_ts_.vs[i_ts]["visited"] = False
        new_ts_.vs[i_ts]["knowledge"] = copy.deepcopy(ts_.vs["potential_posts"])
    i_ts = 0
    while True:
        new_ts_.vs[i_ts]["originalid"] = i_ts

        current_knowledge = new_ts_.vs[i_ts]["knowledge"]
        if len(current_knowledge[i_ts]) > 1:
            new_ts_.add_vertices(len(current_knowledge[i_ts]) - 1)
            for i in range(len(current_knowledge[i_ts]) - 1):
                new_ts_.vs[-i - 1]["originalid"] = i_ts
                new_ts_.vs[-i - 1]["knowledge"] = copy.deepcopy(new_ts_.vs[i_ts]["knowledge"][i])
            new_ts_.vs[i_ts]["knowledge"] = copy.deepcopy(new_ts_.vs[i_ts]["knowledge"][-1])
        i_ts += 1
        if i_ts >= new_ts_.vcount():
            break
    return new_ts_


def gen_random_ts(num_states,
                  num_init,
                  min_trans_per_state,
                  max_trans_per_state,
                  min_uncertain_trans,
                  max_uncertain_trans,
                  min_cost,
                  max_cost,
                  label_names,
                  label_num_vec):  # -> List ig.Graph()
    random.seed(time.ctime())
    ts = ig.Graph(n=num_states)
    # basic check
    if max_trans_per_state > num_states - 1:
        max_trans_per_state = num_states - 1
    # build a connected graph
    while True:
        ts.delete_edges(ts.es)
        # add edges
        for i in range(num_states):
            num_trans = max(random.randint(min_trans_per_state, max_trans_per_state), len(ts.es.select(_target=i)))
            num_new_trans = num_trans - len(ts.es.select(_target=i))
            num_added_trans = 0
            while True:
                if num_added_trans == num_new_trans:
                    break
                elif num_new_trans < 0:
                    print("???")
                    break
                temp_v_id = random.randint(0, num_states-1)
                try:
                    ts.get_eid(i, temp_v_id)
                except:
                    if max_trans_per_state - len(ts.es.select(_target=temp_v_id)) == 0:
                        continue
                    ts.add_edges([(i, temp_v_id)])
                    ts.es[ts.ecount() - 1]["cost"] = random.randint(min_cost, max_cost)
                    ts.es[ts.ecount() - 1]["uncertainty"] = False
                    num_added_trans += 1
        if ts.is_connected():
            break
    # gen init states
    init_states_id = random.sample(range(num_states), num_init)
    # gen uncertain edges and check
    while True:
        num_uncertain_trans = random.randint(min_uncertain_trans, max_uncertain_trans)
        uncertain_edges_id = random.sample(range(ts.ecount()), num_uncertain_trans)
        ts_copy = copy.deepcopy(ts)
        ts_copy.delete_edges(uncertain_edges_id)
        if ts_copy.is_connected():
            for i in uncertain_edges_id:
                ts.es[i]["uncertainty"] = True
            break
    # gen AP
    num_labeled_total = np.sum(label_num_vec)
    if num_labeled_total > num_states:
        print("too many labels")
        num_labeled_total = num_states
    selectedIDs = random.sample(range(num_states), num_labeled_total)
    current_class = 0
    for i in range(len(selectedIDs)):
        if i >= label_num_vec[current_class]:
            current_class += 1
            label_num_vec[current_class] += label_num_vec[current_class - 1]
        ts.vs[selectedIDs[i]]["AP"] = label_names[current_class]

    return init_states_id, ts


if __name__ == "__main__":
    # ts = ig.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (0, 5), (5, 3), (5, 6), (6, 3), (6, 4)])
    # ts.vs["AP"] = [None, None, None, "distinguisher", "fire", None, None]
    # ts.es["cost"] = [10, 40, 50, 100, 2, 20, 35, 10, 40, 100]
    # ts.es["uncertainty"] = [False, False, False, False, True, False, False, True, False, False, False]

    # ts = gen_extended_system(ts)
    # ts = ig.Graph([(0, 1), (0, 2), (1, 2)])
    # ts.vs["AP"] = [None, "distinguisher", "fire"]
    # ts.es["cost"] = [10, 40, 50]
    # ts.es["uncertainty"] = [False, False, False]

    init_states_id, ts = gen_random_ts(8, 1, 1, 3, 1, 2, 1, 50, ["distinguisher", "fire"], [2, 1])
    print(ts)
    print(ts.vs["AP"])
    print(ts.es["uncertainty"])
    print(ts.es["cost"])
    fire_id=ts.vs.select(AP_eq="fire").indices[0]
    start_id=init_states_id[0]
    # dijkstra
    print(ts.get_shortest_path(start_id,fire_id,ts.es["cost"]))
    # automata = ig.Graph([(0, 1), (1, 2), (0, 3)], directed=True)
    # automata.vs["accepting"] = [False, False, True, False]
    # automata.es["condition"] = ["distinguisher", "fire", "fire"]
    #
    # pa = gen_product_automata(ts, automata)
    # print(pa)
    # print(pa.vs["accepting"])
    # print(pa.es["cost"])
    # print(pa.vs(accepting_eq=True).indices)
