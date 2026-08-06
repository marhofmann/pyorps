import networkx as nx
import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from check_tree_topologies import get_line_gdf, print_result_summary, net_valid


def create_nk_graph(net, line_param=None, respect_switches=True):
    nxg = nx.Graph()
    if respect_switches:
        closed_lines = net.switch.loc[(net.switch.et == 'l') & net.switch.closed].element.values
        lines = net.line.loc[closed_lines]
        res_lines = net.res_line.loc[closed_lines]
    else:
        lines = net.line
        res_lines = net.res_line
    from_nodes = lines.from_bus.values
    to_nodes = lines.to_bus.values
    if line_param is not None:
        if line_param in net.line.columns:
            weight = lines[line_param].values
        elif line_param in net.res_line.columns:
            weight = res_lines[line_param].values
        else:
            raise ValueError("Invalid line parameter")
        nxg.add_weighted_edges_from(zip(from_nodes, to_nodes, weight))
    else:
        nxg.add_edges_from(zip(from_nodes, to_nodes))
    return nxg


def find_tree(net, line_param, slack=None, degree_per_level=None):
    if degree_per_level is None:
        degree_per_level = [5, 3, 2]
    degree_per_level = iter(degree_per_level)

    nxg = create_nk_graph(net, line_param)

    if slack is None:
        slack = net.ext_grid.bus.values[0]
    if line_param in net.line.columns:
        line_data = net.line[line_param].values
    else:
        line_data = net.res_line[line_param].values

    edge_lkp = dict(zip(zip(net.line.from_bus.values, net.line.to_bus.values), line_data))
    edge_lkp.update(dict(zip(zip(net.line.to_bus.values, net.line.from_bus.values), line_data)))

    neighbours_weight = {n: sum(nxg[n][nei]['weight'] for nei in nxg[n]) for n in nxg.nodes()}

    tree = nx.Graph()
    levels = {}
    max_deg = next(degree_per_level)
    level = 0
    visited = {slack}
    level_nodes = [slack]
    all_lvl_nodes = set()
    while tree.number_of_nodes() < nxg.number_of_nodes():
        all_lvl_nodes.update(level_nodes)
        next_level_nodes = []
        for node in level_nodes:
            level_edges = {}
            for neighbour in nxg[node]:
                if neighbour in visited or tree.has_node(neighbour):
                    continue
                weight = edge_lkp[(neighbour, node)]
                if max_deg == len(level_edges):
                    if weight > (max_weight := min(level_edges.values())):
                        max_edge = [edge for edge, weight in level_edges.items() if weight == max_weight][0]
                        del level_edges[max_edge]
                        level_edges[(neighbour, node)] = weight
                else:
                    level_edges[(neighbour, node)] = weight
            if level in levels:
                levels[level].update(level_edges)
            else:
                levels.update({level: level_edges})
            edge_data = [(uv[0], uv[1], w)for uv, w in level_edges.items()]
            tree.add_weighted_edges_from(edge_data)
            next_level_nodes.extend(n for u, v in level_edges.keys() for n in (u, v) if n not in all_lvl_nodes)
        level_nodes = sorted(next_level_nodes, key=neighbours_weight.get)
        visited.update(level_nodes)
        level += 1
        try:
            max_deg = next(degree_per_level)
        except StopIteration:
            pass
        print(levels)
    from_buses, to_buses = np.array(list(tree.edges())).T
    tree_lines = ((net.line.from_bus.isin(from_buses) & net.line.to_bus.isin(to_buses)) |
                 (net.line.to_bus.isin(from_buses) & net.line.from_bus.isin(to_buses)))
    oos = net.line.loc[~tree_lines].index.values
    net.line.loc[oos, 'in_service'] = False
    return net



def get_switches_on_path(net, sp):
    path_lines = net.line.loc[net.line.from_bus.isin(sp) & net.line.to_bus.isin(sp)].index.values
    switches = net.res_switch.loc[net.switch.element.isin(path_lines) &
                                  net.switch.bus.isin(sp) &
                                  net.switch.closed &
                                  (net.switch.et == 'l')]
    return switches


def open_simple_cycles_lowest_current(net, opt_func):
    nxg = create_nk_graph(net, None, respect_switches=False)
    print("Running open-simple-cycles-lowest-current-algorithm.")
    iterations = 0
    while cycles := nx.cycle_basis(nxg):
        print(f"Iteration {iterations} - {len(cycles)} cycles left")
        #cycles = sorted(cycles, key=len)
        all_switches = [get_switches_on_path(net, cycle) for cycle in cycles]
        switches = pd.concat(all_switches)
        open_switch = opt_func(net, switches)
        net.switch.at[open_switch, 'closed'] = False
        pp.runpp(net)
        nxg = create_nk_graph(net, None, respect_switches=True)
        iterations += 1
    print(f"Done! Optimization took {iterations} iterations")


def min_switch_current(net, cycle_switches):
    min_switch = cycle_switches.i_ka.idxmin()
    return min_switch


def min_cost_per_cycle(net, cycle_switches):
    lines = net.switch.loc[cycle_switches.index].element.values
    min_cost_line = (net.line.loc[lines].cost / net.res_line.loc[lines].i_ka).idxmin()
    min_switch = net.switch.loc[net.switch.element == min_cost_line].index.values[0]
    return min_switch


def min_cost_current_per_cycle(net, cycle_switches):
    line_idx = net.switch.loc[cycle_switches.index].element.values
    lines = net.line.loc[line_idx]
    res_lines = net.res_line.loc[line_idx]

    normalized_cost = lines.cost / lines.cost.max()
    normalized_current = res_lines.i_ka / res_lines.i_ka.max()

    return (normalized_cost / normalized_current).idxmin()


def find_topology(net, slack=None, plot=False, max_line_loading=50, max_slack_lines=5, **kwargs):
    if slack is None:
        slack = net.ext_grid.bus.values[0]

    slack_lines = net.line.loc[((net.line.from_bus == slack) | (net.line.to_bus == slack))].index.values
    edge_lkp = dict(zip(net.line.index.values, zip(net.line.from_bus.values, net.line.to_bus.values)))

    line_lkp = dict(zip(zip(net.line.from_bus.values, net.line.to_bus.values), net.line.index.values))
    line_lkp.update(dict(zip(zip(net.line.to_bus.values, net.line.from_bus.values), net.line.index.values)))

    iterations = 0
    power_flow_results_valid = False
    plt.ion()
    while True:
        full_graph = create_nk_graph(net, 'cost', respect_switches=False)
        checked_lines = set()
        was_overloaded = set()
        while True:
            print(f"Iteration {iterations}")
            cost_tree = nx.minimum_spanning_tree(full_graph)
            tree_lines = [line_lkp[(u, v)] for u, v in cost_tree.edges()]

            co_tree_lines = net.line.loc[~net.line.index.isin(tree_lines)].index.values
            net.line.in_service = True
            net.line.loc[co_tree_lines, 'in_service'] = False
            pp.runpp(net)
            net.res_line.loc[co_tree_lines, 'loading_percent'] = np.nan
            if plot:
                lines = get_line_gdf(net)
                #lines.to_file(f"results\min_cost_tree.geojson")
                ax = lines.plot(column="loading_percent", legend=True, cmap="jet", figsize=(13, 8),
                                legend_kwds={'label': "Line loading in %"})
                ax.set_title(f"Minimum cost tree iteration {iterations}")
                plt.tight_layout()
                #plt.savefig(f"results\my_method_{iterations}.png", dpi=300)
                plt.pause(1)  # Pause for 1 second
                plt.close()

            #   plt.close()
            if net_valid(net, max_line_loading=max_line_loading, **kwargs):
                print(f"Power flow valid!\n Max. line loading = {net.res_line.loading_percent.max():.2} %")
                power_flow_results_valid = True
                break
            print(f"Power flow NOT valid!\n Max. line loading = {net.res_line.loading_percent.max():.2} %")
            was_overloaded.update(net.res_line.loc[net.res_line.loading_percent > max_line_loading].index.values)
            slack_tree_line = np.intersect1d(np.array(tree_lines), slack_lines)
            max_loaded_line = net.res_line.loading_percent.idxmax()
            checked_lines.update(slack_tree_line)
            print(net.line.loc[list(checked_lines), ['from_bus', 'to_bus']])
            max_line_current = net.line.at[max_loaded_line, 'max_i_ka']
            over_loading = net.res_line.at[max_loaded_line, 'loading_percent'] - max_line_loading
            over_current = (max_line_current * over_loading) / 100
            net.line.in_service = True

            deg1 = [n for n in cost_tree.nodes() if cost_tree.degree(n) == 1]
            fb = net.line.at[max_loaded_line, 'from_bus']
            tb = net.line.at[max_loaded_line, 'from_bus']
            fb_path_length = nx.shortest_path_length(cost_tree, slack, fb)
            tb_path_length = nx.shortest_path_length(cost_tree, slack, tb)
            node = fb if fb_path_length > tb_path_length else tb
            deg1_paths = {n: nx.shortest_path(cost_tree, n, node) for n in deg1}

            for n in sorted(deg1_paths, key=lambda k: len(deg1_paths[k]), reverse=True):
                path = deg1_paths[n]
                break_outer = False
                for u, v in zip(path[:-1], path[1:]):
                    if node != slack and slack in path:
                        continue
                    line = line_lkp[(u, v)]
                    if net.res_line.at[line, 'i_ka'] > over_current:
                        full_graph.remove_edge(u, v)
                        cost_tree.remove_edge(u, v)
                        break_outer = True
                        break
                if break_outer:
                    break
            if len(checked_lines) >= max_slack_lines:
                break
            iterations += 1
        if not power_flow_results_valid:
            checked_slack_lines = net.line.loc[list(checked_lines)]
            net.line.loc[net.line.index.isin(checked_slack_lines.index) &
                         net.line.index.isin(was_overloaded) &
                         net.line.in_service, 'parallel'] += 1
            checked_lines_buses = np.concatenate([checked_slack_lines.from_bus.values,
                                                  checked_slack_lines.to_bus.values])
            max_parallel = net.line.parallel.max()
            parallel_neighbour_slack_lines = []
            for nr_of_neighbours in range(1, max_parallel):
                for n in checked_lines_buses:
                    if n == slack:
                        continue
                    for neighbour in cost_tree[n]:
                        line = line_lkp[(neighbour, n)]
                        if line in was_overloaded and not pd.isnull(net.res_line.at[line, 'loading_percent']):
                            parallel_neighbour_slack_lines.append(line)
            net.line.loc[parallel_neighbour_slack_lines, 'parallel'] += 1
            print(f"Parallel lines:\n{net.line.loc[net.line.parallel > 1, ['parallel', 'from_bus', 'to_bus']]}")
        else:
            break

    parallel_lines = net.line.loc[net.line.parallel > 1]
    for line, parallel in zip(parallel_lines.index.values, parallel_lines.parallel.values):
        if pd.isnull(net.res_line.at[line, 'loading_percent']):
            net.line.loc[line, 'parallel'] = 1
            continue
        for p in range(parallel - 1, 0, -1):
            net.line.loc[line, 'parallel'] = p
            pp.runpp(net)
            if net.res_line.at[line, 'loading_percent'] > max_line_loading:
                net.line.loc[line, 'parallel'] = p + 1
                break
    print("\n\nOptimization complete!")
    net.res_line.loc[~net.line.in_service, 'loading_percent'] = np.nan
    print(f"System cost: {net.line.loc[net.line.in_service, 'cost'].sum()}")
    total_line_length = net.line.loc[net.line.in_service, 'length_km'] * net.line.loc[net.line.in_service, 'parallel']
    print(f"Total line length km: {(total_line_length).sum()}")
    p = net.res_ext_grid.p_mw.sum()
    q = net.res_ext_grid.q_mvar.sum()
    s = np.sqrt(np.power(p, 2) + np.power(q, 2))
    print(f"P = {p:.3f} MW\tQ = {q:.3f} MVAR\tS = {s:.3f}MVA")
    lines = get_line_gdf(net)
    ax = lines.plot(column="loading_percent", legend=True, cmap="jet", figsize=(13, 8),
                    legend_kwds={'label': "Line loading in %"})
    ax.set_title(f"Result")
    plt.tight_layout()
    #plt.savefig(f"results\my_method_result_05.png", dpi=300)
    plt.pause(5)  # Pause for 1 second
    plt.close()
    return net


def topology_valid(net):
    nxg = create_nk_graph(net, respect_switches=True)
    try:
        nx.find_cycle(nxg)
        return False
    except nx.NetworkXNoCycle:
        pass
    connected = {n for bus in net.ext_grid.bus.values for n in nx.node_connected_component(nxg, bus)}
    if any(bus not in connected for bus in net.load.bus):
        return False
    else:
        return True


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    net = pp.from_json((r"C:\Users\mhnn82\Documents\3_python_projects\pyorps_private"
                   r"\case_studies\substation_planning\substation_planning_pp_grid"
                        r".json"))

    find_topology(net, plot=False, max_line_loading=50)
    [print(l) for l in net.line.loc[~net.line.in_service].index.values]
    lines = get_line_gdf(net)
    #lines.to_file("my_method__.geojson")
    #pp.runpp(net)
    #net = find_tree(net, 'i_ka')
    #pp.runpp(net)
    #print_result_summary(net)
    #net.line = net.line.loc[net.line.in_service]
    #net.res_line = net.res_line.loc[net.line.index]
    #get_line_gdf(net).to_file('find_tree_test.geojson')
    #opt_func = min_cost_current_per_cycle
    #open_simple_cycles_lowest_current(net, opt_func=opt_func)
    #save_path = rf"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\results
    # \open_simple_cycles_lowest_current_{opt_func.__name__}.geojson"
    #get_line_gdf(net).to_file(save_path)

