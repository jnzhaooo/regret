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
        new_ts_.vs[i_ts]["originalid"]=i_ts

        current_knowledge=new_ts_.vs[i_ts]["knowledge"]
        if len(current_knowledge[i_ts])>1:
            new_ts_.add_vertices(len(current_knowledge[i_ts])-1)
            for i in range(len(current_knowledge[i_ts])-1):
                new_ts_.vs[-i-1]["originalid"]=i_ts
                new_ts_.vs[-i-1]["knowledge"]=copy.deepcopy(new_ts_.vs[i_ts]["knowledge"][i])
            new_ts_.vs[i_ts]["knowledge"] = copy.deepcopy(new_ts_.vs[i_ts]["knowledge"][-1])
        i_ts += 1
        if i_ts >= new_ts_.vcount():
            break
    return new_ts_


if __name__ == "__main__":
    ts = ig.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (0, 5), (5, 3), (5, 6), (6, 3), (6, 4)])
    ts.vs["AP"] = [None, None, None, "distinguisher", "fire", None, None]
    ts.es["cost"] = [10, 40, 50, 100, 2, 20, 35, 10, 40, 100]
    ts.es["uncertainty"] = [False, False, False, False, True, False, False, True, False, False, False]

    # ts = gen_extended_system(ts)
    # ts = ig.Graph([(0, 1), (0, 2), (1, 2)])
    # ts.vs["AP"] = [None, "distinguisher", "fire"]
    # ts.es["cost"] = [10, 40, 50]
    # ts.es["uncertainty"] = [False, False, False]

    automata = ig.Graph([(0, 1), (1, 2), (0, 3)], directed=True)
    automata.vs["accepting"] = [False, False, True, False]
    automata.es["condition"] = ["distinguisher", "fire", "fire"]

    pa = gen_product_automata(ts, automata)
    print(ts)
    print(ts.es["cost"])
    #print(pa.vs["accepting"])
    #print(pa.es["cost"])
    #print(pa.vs(accepting_eq=True).indices)
