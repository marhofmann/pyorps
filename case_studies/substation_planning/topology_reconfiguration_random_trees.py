
import pandapower as pp
import networkx as nx
import numpy as np
from numba import njit, prange, bool as nb_bool
import pandas as pd
from networkx.algorithms.tree.mst import SpanningTreeIterator
from itertools import chain, pairwise
import rustworkx as rx

from progressbar import progressbar


def voltage_bounds_violated(net: pp.pandapowerNet, min_vm_pu: float = 0.9, max_vm_pu: float = 1.1) -> bool:
    """
    Aufgabe 3 d) iv)
    Überprüft die Knotenspannungen in den Berechnungsergebnissen nach einem Lastfluss. Gibt False zurück,
    wenn die Grenzwerte nicht eingehalten werden un True, sofern sie eingehalten sind.

    :param net: Das zu überprüfende Pandapower Netz
    :param min_vm_pu: Die untere Grenze der Knotenspannungen - alle Knotenspannungen müssen über diesem Wert liegen
    :param max_vm_pu: Die obere Grenze der Knotenspannungen - alle Knotenspannungen müssen darunter liegen
    :return: True, wenn die Grenzwerte eingehalten werden, False, wenn sie verletzt werden
    """
    return ((net.res_bus.vm_pu < min_vm_pu) | (net.res_bus.vm_pu > max_vm_pu)).any()


def lines_overloaded(net: pp.pandapowerNet, max_line_loading: float = 100) -> bool:
    """
    Aufgabe 3 d) v)
    Überprüft ob die Auslastung aller Leitungen unter einem Grenzwert liegen. Gibt False zurück, wenn
    Leitungsauslastungen über diesem Grenzwert liegen und True, wenn alle Leitungsauslastungen unter diesem Grenzwert
    liegen.

    :param net: Das Pandapower Netz, welches überprüft werden soll.
    :param max_line_loading: Der Auslastungsgrenzwert, unter dem die Leitungsauslastungen liegen müssen.
    :return: True, wenn der Grenzwert eingehalten wird, False, wenn nicht.
    """
    return (net.res_line.loading_percent > max_line_loading).any()


def trafos_overloaded(net: pp.pandapowerNet, max_trafo_loading: float = 100) -> bool:
    """
    Aufgabe 3 d) vi)
    Überprüft, ob die Auslastung aller Transformatoren unter einem Grenzwert liegen. Gibt False zurück,
    wenn die Grenzwerte nicht eingehalten werden und True, sofern sie eingehalten sind.

    :param net: Das zu überprüfende Pandapower Netz
    :param max_trafo_loading: Der Auslastungsgrenzwert
    :return: True, wenn der Grenzwert eingehalten wird, False, wenn nicht.
    """
    return (net.res_trafo.loading_percent > max_trafo_loading).any()


def net_valid(net: pp.pandapowerNet, min_vm_pu: float = 0.9, max_vm_pu: float = 1.1,
              max_line_loading_percent: float = 100., max_trafo_loading: float = 100.) -> bool:
    """
    Zusammenfassung der Aufgabe 3 d) iv) - vi)
    Sammelt die Funktionen zur Prüfung der Lastflussergebnisse. Ergibt die Prüfung eine Grenzwertverletzung so wird
    False zurückgegeben und True, wenn alle Grenzwerte eingehalten werden.

    :param net: Das zu überprüfende Pandapower Netz
    :param min_vm_pu: Die minimale Knotenspannung die nicht unterschritten werden soll
    :param max_vm_pu: Die maximale Knotenspannung die nicht überschritten werden soll
    :param max_line_loading_percent: Die maximale Leitungsauslastung, die nicht überschritten werden soll
    :param max_trafo_loading: Die maximale Transformatorauslastung, die nicht überschritten werden soll
    :return: True wenn alle Grenzwerte eingehalten werden und False, wenn nicht
    """
    if voltage_bounds_violated(net, min_vm_pu, max_vm_pu):
        return False
    elif lines_overloaded(net, max_line_loading_percent):
        return False
    elif trafos_overloaded(net, max_trafo_loading):
        return False
    else:
        return True


@njit(parallel=True)
def check_array_match(flat_array, arrays_2d, arrays_to_check):
    """
    Check if flat_array matches any row in the 2D array, regardless of element order.

    Args:
        flat_array: 1D numpy array of integers
        arrays_2d: 2D numpy array where each row is compared with flat_array

    Returns:
        bool: True if a match is found, False otherwise
    """
    if arrays_to_check == 0:
        return False
    # Pre-sort the input array once
    sorted_flat = np.sort(flat_array)
    result_array = np.zeros((arrays_to_check,), dtype=nb_bool)
    # Use parallel processing for better performance
    for i in prange(arrays_to_check):
        row = arrays_2d[i]
        # Only sort and do full comparison if sizes match
        sorted_row = np.sort(row)
        if np.array_equal(sorted_flat, sorted_row):
            result_array[i] = True

    return np.any(result_array)


@njit()
def find_augmenting_path(list_idx, lol, flat_set, element_to_list, assigned_element, visited):
    """Find an augmenting path starting from list_idx using an iterative approach."""
    if visited[list_idx]:
        return False

    visited[list_idx] = True

    for elem_idx in range(len(lol[list_idx])):
        elem = lol[list_idx][elem_idx]
        if elem in flat_set:
            # Check if element is not assigned yet (value -1 means unassigned)
            assigned_to = element_to_list[elem]
            if assigned_to == -1:
                element_to_list[elem] = list_idx
                assigned_element[list_idx] = elem
                return True

            # If element is already assigned, try to reassign it
            other_list_idx = element_to_list[elem]
            temp_visited = visited.copy()  # Need a copy for recursive-like behavior
            if find_augmenting_path(other_list_idx, lol, flat_set, element_to_list, assigned_element, temp_visited):
                element_to_list[elem] = list_idx
                assigned_element[list_idx] = elem
                return True

    return False


def reorder_flat_list_numba(lol, flat_list_unordered):
    """
    Numba-accelerated version of reorder_flat_list.

    Parameters:
    -----------
    lol : list of lists or list of numpy arrays
        List of lists containing integers.
    flat_list_unordered : numpy.ndarray
        1D numpy array containing integers.

    Returns:
    --------
    numpy.ndarray
        Array containing assigned elements.
    """
    # Convert lol to a format that works well with Numba
    n = len(lol)
    max_len = max(len(sublist) for sublist in lol)

    # Create a 2D array with padding
    lol_array = np.full((n, max_len), -1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)

    for i, sublist in enumerate(lol):
        length = len(sublist)
        lengths[i] = length
        lol_array[i, :length] = sublist if isinstance(sublist, np.ndarray) else np.array(sublist, dtype=np.int64)

    # Call the Numba-optimized core function
    return _reorder_flat_list_core(lol_array, lengths, flat_list_unordered)


@njit()
def _reorder_flat_list_core(lol_array, lengths, flat_list_unordered):
    """Core algorithm optimized with Numba."""
    n = len(lol_array)

    # Create a set from flat_list_unordered using a simple array-based approach
    max_elem = np.max(flat_list_unordered)
    flat_set = np.zeros(max_elem + 1, dtype=np.int8)
    for elem in flat_list_unordered:
        if elem >= 0:  # Only handle non-negative integers
            flat_set[elem] = 1

    # Count matching elements for each list
    constraints = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(lengths[i]):
            elem = lol_array[i, j]
            if elem >= 0 and elem < len(flat_set) and flat_set[elem] == 1:
                constraints[i] += 1

    # Sort list indices by constraint count
    list_indices = np.argsort(constraints)

    # Create bipartite matching structures
    assigned_element = np.full(n, -1, dtype=np.int64)
    element_to_list = np.full(len(flat_set), -1, dtype=np.int64)  # Maps elements to list index

    # Process lists in order of fewest matching elements
    for idx in range(n):
        list_idx = list_indices[idx]
        visited = np.zeros(n, dtype=np.bool_)
        _find_augmenting_path(list_idx, lol_array, lengths, flat_set, element_to_list, assigned_element, visited)

    return assigned_element


@njit()
def _find_augmenting_path(list_idx, lol_array, lengths, flat_set, element_to_list, assigned_element, visited):
    """Find an augmenting path starting from list_idx."""
    if visited[list_idx]:
        return False

    visited[list_idx] = True

    for j in range(lengths[list_idx]):
        elem = lol_array[list_idx, j]
        if elem >= 0 and elem < len(flat_set) and flat_set[elem] == 1:
            # If element is not assigned yet
            if element_to_list[elem] == -1:
                element_to_list[elem] = list_idx
                assigned_element[list_idx] = elem
                return True

            # If element is already assigned, try to reassign it
            other_list_idx = element_to_list[elem]
            if _find_augmenting_path(other_list_idx, lol_array, lengths, flat_set, element_to_list, assigned_element,
                                     visited):
                element_to_list[elem] = list_idx
                assigned_element[list_idx] = elem
                return True

    return False


# Define Jaccard similarity function (ratio of intersection to union)
@njit()
def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def sort_by_increasing_similarity(list_of_lists):
    """
    Sorts a list of lists by their average similarity to all other lists,
    placing lists with lowest average similarity first.

    Args:
        list_of_lists: A list containing sublists of integers

    Returns:
        A new list of lists sorted by increasing average similarity
    """
    n = len(list_of_lists)
    if n <= 1:
        return list_of_lists.copy()  # No sorting needed

    # Calculate average similarity of each list to all others
    similarity_scores = []
    for i, current_list in enumerate(list_of_lists):
        total_similarity = sum(
            jaccard_similarity(current_list, other_list)
            for j, other_list in enumerate(list_of_lists)
            if i != j
        )
        avg_similarity = total_similarity / (n - 1)
        similarity_scores.append((i, avg_similarity))

    # Sort indices by their average similarity scores
    similarity_scores.sort(key=lambda x: x[1])

    # Return a new list sorted by increasing similarity
    return [list_of_lists[idx] for idx, _ in similarity_scores]


def calculate_jaccard_threshold(items_list, subset_size, num_subsets):
    # Flatten the list of lists to get all unique elements
    all_items = set(chain(*items_list))
    total_items = len(all_items)

    # Theoretical maximum similarity when sets differ by one element
    theoretical_max = (subset_size - 1) / (subset_size + 1)

    # Adjustment based on total items vs subset size
    coverage_ratio = (subset_size * num_subsets) / total_items

    if total_items < subset_size * num_subsets:
        # Not enough elements to create completely distinct subsets
        min_overlap = (subset_size * num_subsets - total_items) / num_subsets
        min_similarity = min_overlap / (2 * subset_size - min_overlap)
        # Add a small buffer to the minimum
        return min_similarity + 0.05
    else:
        # We could potentially create completely distinct subsets
        # but we'll allow some similarity for practical purposes
        base_threshold = 0.3 * theoretical_max
        # Adjust based on how many subsets vs. available combinations
        from math import comb
        total_possible = comb(total_items, subset_size)
        subset_ratio = num_subsets / total_possible if total_possible > 0 else 1
        return base_threshold + (0.5 * theoretical_max * subset_ratio)


def count_spanning_trees(graph):
    """
    Count the number of spanning trees in a graph using Kirchhoff's Matrix Theorem.

    Parameters:
        graph: A rustworkx PyGraph object

    Returns:
        int: The number of spanning trees in the graph
    """
    # Get actual node indices (which might not be 0,1,2,...,n-1)
    node_indices = list(graph.node_indices())

    if len(node_indices) <= 1:
        return len(node_indices)  # 0 for empty graph, 1 for single node

    # Check connectivity
    if not rx.is_connected(graph):
        return 0

    # Create a mapping from actual node indices to matrix positions
    idx_to_pos = {idx: pos for pos, idx in enumerate(node_indices)}
    n = len(node_indices)

    # Initialize Laplacian matrix
    laplacian = np.zeros((n, n), dtype=float)

    # Handle diagonal entries (degrees)
    for i, node_idx in enumerate(node_indices):
        laplacian[i, i] = graph.degree(node_idx)

    # Handle off-diagonal entries (-1 for connected nodes)
    for node_idx in node_indices:
        # Get neighbors for this node
        for neighbor_idx in graph.neighbors(node_idx):
            i = idx_to_pos[node_idx]
            j = idx_to_pos[neighbor_idx]
            laplacian[i, j] = -1

    # Per Kirchhoff's theorem, get the cofactor by removing first row and column
    cofactor = laplacian[1:, 1:]

    # Calculate determinant
    det_value = np.linalg.det(cofactor)

    # Return the absolute value, rounded to an integer
    return int(round(abs(det_value)))


def create_trees_with_cycle_switches(net, n_trees=10, resistance_first=True):
    """
    Creates a NetworkX graph from a pandapower network, finds all cycles,
    identifies line-switches in cycles, and generates trees with excluded switches.

    Uses SpanningTreeIterator with line resistances as weights and decreasing sorting strategy.

    Parameters:
    -----------
    net : pandapower.auxiliary.pandapowerNet
        The pandapower network
    n_trees : int, optional
        Number of trees to generate (default: 10)

    Returns:
    --------
    tuple
        - cycle_line_switches: List of lists, each containing line-switch indices for a cycle
        - tree_excluded_switches: List of lists, each containing line-switch indices excluded from a tree
    """
    print(f"Extracting trees from a given pandapower network...")
    # Map from bus1-bus2 pair to line_idx
    from_buses = net.line.from_bus.values
    to_buses = net.line.to_bus.values

    edge_line = dict(zip(zip(net.line.from_bus, net.line.to_bus), net.line.index))
    complete_edge_line = dict(edge_line)
    complete_edge_line.update(dict(zip(zip(net.line.to_bus, net.line.from_bus), net.line.index)))

    # Map from line_idx to list of switch indices on that line
    line_switches = net.switch.loc[net.switch.et == 'l']
    line_switches = dict(zip(line_switches.element, line_switches.index))
    graph = rx.PyGraph()
    unique_nodes = list(range(np.unique(np.concatenate([from_buses, to_buses])).max() + 1))
    graph.add_nodes_from(unique_nodes)
    edge_switch = {edge: line_switches[line] for edge, line in complete_edge_line.items()}
    if resistance_first:
        net.line['r_ohm'] = net.line.length_km * net.line.r_ohm_per_km
        r_dict = net.line.r_ohm.to_dict()
        edge_data = [(u_v[0], u_v[1], r_dict[li]) for u_v, li in edge_line.items()]
        graph.add_edges_from(edge_data)

    else:
        weights = np.random.random_sample(from_buses.size)
        graph.add_edges_from(list(zip(from_buses, to_buses, weights)))
    # Step 3: Create an undirected graph with resistance values as edge weights

    ext_grid_connection = [(s1, s2, 0) for s1, s2 in pairwise(net.ext_grid.bus)]
    graph.add_edges_from(ext_grid_connection)
    num_spanning_trees = count_spanning_trees(graph)
    graph.remove_edges_from([(u, v) for u, v, _ in ext_grid_connection])

    print(f"There are {num_spanning_trees} possible spanning trees which can be formed by this network!")
    # Step 4: Find all cycles in the graph
    cycles = rx.cycle_basis(graph)

    cycle_switches = []
    for cycle in cycles:
        cycle_lines = net.line.loc[net.line.from_bus.isin(cycle) & net.line.to_bus.isin(cycle)].index.values
        switches = net.switch.loc[net.switch.bus.isin(cycle) & net.switch.element.isin(cycle_lines) &
                                  (net.switch.et == 'l')].index.values
        cycle_switches.append([int(sw) for sw in switches])

    nr_of_cycles = len(cycle_switches)
    tree_count = 0
    tree_excluded_switches = np.zeros((n_trees, nr_of_cycles), dtype=int)
    while tree_count < n_trees:
        mse = rx.minimum_spanning_edges(graph, weight_fn=float)
        arr = mse.__array__().T
        el = list(zip(arr[0], arr[1]))
        graph.clear_edges()
        weights = np.random.random_sample(from_buses.size)
        graph.add_edges_from(list(zip(from_buses, to_buses, weights)))
        co_tree = np.array([edge_switch[(u, v)] for u, v in graph.edge_list() if (u, v) not in el and (v, u) not in el])
        if co_tree.size != np.unique(co_tree).size or co_tree.size != nr_of_cycles:
            continue
        ordered_tree_switches = reorder_flat_list_numba(cycle_switches, co_tree)
        no_none = all(si is not None for si in ordered_tree_switches)
        if no_none and not check_array_match(co_tree, tree_excluded_switches, tree_count):
            tree_excluded_switches[tree_count]  = ordered_tree_switches
            tree_count += 1
        else:
            continue

    print(f"Extracted {tree_count} different spanning trees from the net data!")
    return cycle_switches, tree_excluded_switches


def create_trees_with_cycle_switches_one_by_one(net, func, parameter, n_trees=10, **kwargs):
    """
    Creates a NetworkX graph from a pandapower network, finds all cycles,
    identifies line-switches in cycles, and generates trees with excluded switches.

    Uses SpanningTreeIterator with line resistances as weights and decreasing sorting strategy.

    Parameters:
    -----------
    net : pandapower.auxiliary.pandapowerNet
        The pandapower network
    n_trees : int, optional
        Number of trees to generate (default: 10)

    Returns:
    --------
    tuple
        - cycle_line_switches: List of lists, each containing line-switch indices for a cycle
        - tree_excluded_switches: List of lists, each containing line-switch indices excluded from a tree
    """
    print(f"Extracting trees from a given pandapower network...")
    # Map from bus1-bus2 pair to line_idx
    from_buses = net.line.from_bus.values
    to_buses = net.line.to_bus.values

    edge_line = dict(zip(zip(net.line.from_bus, net.line.to_bus), net.line.index))
    complete_edge_line = dict(edge_line)
    complete_edge_line.update(dict(zip(zip(net.line.to_bus, net.line.from_bus), net.line.index)))

    # Map from line_idx to list of switch indices on that line
    line_switches = net.switch.loc[net.switch.et == 'l']
    line_switches = dict(zip(line_switches.element, line_switches.index))

    edge_switch = {edge: line_switches[line] for edge, line in complete_edge_line.items()}
    net.line['r_ohm'] = net.line.length_km * net.line.r_ohm_per_km
    r_dict = net.line.r_ohm.to_dict()

    # Step 3: Create an undirected graph with resistance values as edge weights
    graph = rx.PyGraph()
    unique_nodes = list(range(np.unique(np.concatenate([from_buses, to_buses])).max() + 1))
    graph.add_nodes_from(unique_nodes)
    edge_data = [(u_v[0], u_v[1], r_dict[li]) for u_v, li in edge_line.items()]
    graph.add_edges_from(edge_data)
    ext_grid_connection = [(s1, s2, 0) for s1, s2 in pairwise(net.ext_grid.bus)]
    graph.add_edges_from(ext_grid_connection)
    num_spanning_trees = count_spanning_trees(graph)
    graph.remove_edges_from([(u, v) for u, v, _ in ext_grid_connection])

    print(f"There are {num_spanning_trees} possible spanning trees which can be formed by this network!")
    # Step 4: Find all cycles in the graph
    cycles = rx.cycle_basis(graph)

    cycle_switches = []
    for cycle in cycles:
        cycle_lines = net.line.loc[net.line.from_bus.isin(cycle) & net.line.to_bus.isin(cycle)].index.values
        switches = net.switch.loc[net.switch.bus.isin(cycle) & net.switch.element.isin(cycle_lines) &
                                  (net.switch.et == 'l')].index.values
        cycle_switches.append([int(sw) for sw in switches])

    nr_of_cycles = len(cycle_switches)
    tree_count = 0
    overall_min = np.inf
    min_cost = np.inf
    best_option = None
    tree_excluded_switches = np.zeros((n_trees, nr_of_cycles), dtype=int)
    while tree_count < n_trees:
        mse = rx.minimum_spanning_edges(graph, weight_fn=float)
        arr = mse.__array__().T
        el = list(zip(arr[0], arr[1]))
        graph.clear_edges()
        weights = np.random.random_sample(from_buses.size)
        graph.add_edges_from(list(zip(from_buses, to_buses, weights)))
        co_tree = np.array([edge_switch[(u, v)] for u, v in graph.edge_list() if (u, v) not in el and (v, u) not in el])
        if co_tree.size != np.unique(co_tree).size or co_tree.size != nr_of_cycles:
            continue
        ordered_tree_switches = reorder_flat_list_numba(cycle_switches, co_tree)
        no_none = all(si is not None for si in ordered_tree_switches)
        if no_none and not check_array_match(co_tree, tree_excluded_switches, tree_count):
            tree_excluded_switches[tree_count]  = ordered_tree_switches
            tree_count += 1
            net.switch.closed = True
            net.switch.loc[ordered_tree_switches, 'closed'] = False
            pp.runpp(net)
            if not net_valid(net, **kwargs):
                continue
            min_current = func(net)
            if min_current <= overall_min:
                lines_cost = net.line.loc[net.switch.loc[ordered_tree_switches, 'element'], 'cost'].sum()
                if lines_cost < min_cost:  # or min_line_loading_topology < min_loading:
                    best_option = ordered_tree_switches
                    overall_min = min_current
                    min_cost = lines_cost
                    print(f"\nNew minimum {round(min_current, 3)} {parameter}"
                          f"with total line cost {lines_cost} € "
                          f"for switch option {tree_count}\n")
    if best_option is None:
        print("No valid grid option found!")
        return net
    net.switch.closed = True
    net.switch.loc[best_option, 'closed'] = False
    pp.runpp(net)
    pp.to_json(net, f'{n_trees}_{round(min_cost)}_{round(overall_min)}_{parameter}.json')
    print(f"Extracted {tree_count} different spanning trees from the net data!")

    print(np.array(best_option))
    return net


def create_trees_with_cycle_switches_all_trees(net, n_trees=10, max_jaccard_similarity=0.97):
    """
    Creates a NetworkX graph from a pandapower network, finds all cycles,
    identifies line-switches in cycles, and generates trees with excluded switches.

    Uses SpanningTreeIterator with line resistances as weights and decreasing sorting strategy.

    Please note, that it takes very long time to find trees with a low jaccard_similarity index when iterating
    through all spanning trees! For example, in a network with 205 cycles the following similarity indices appear
    after n Trees to be compared:

        Max. jaccard similarity index   ->  Number of compared trees
        0.9903                          ->  2
        0.9807                          ->  4
        0.9712                          ->  10
        0.9617                          ->  59
        0.9524                          ->  235
        0.9431                          ->  854
        0.9340                          ->  2931
        0.9249                          ->  14167
        0.9159                          ->  69104

    Parameters:
    -----------
    net : pandapower.auxiliary.pandapowerNet
        The pandapower network
    n_trees : int, optional
        Number of trees to generate (default: 10)

    Returns:
    --------
    tuple
        - cycle_line_switches: List of lists, each containing line-switch indices for a cycle
        - tree_excluded_switches: List of lists, each containing line-switch indices excluded from a tree
    """

    # Map from bus1-bus2 pair to line_idx
    edge_line = dict(zip(zip(net.line.from_bus, net.line.to_bus), net.line.index))
    edge_line.update(dict(zip(zip(net.line.to_bus, net.line.from_bus), net.line.index)))

    # Map from line_idx to list of switch indices on that line
    line_switches = net.switch.loc[net.switch.et == 'l']
    line_switches = dict(zip(line_switches.element, line_switches.index))

    edge_switches = {edge: line_switches[line] for edge, line in edge_line.items()}
    net.line['r_ohm'] = net.line.length_km * net.line.r_ohm_per_km
    r_dict = net.line.r_ohm.to_dict()
    # Step 3: Create an undirected graph with resistance values as edge weights
    graph = nx.Graph()
    graph.add_edges_from([(u_v[0], u_v[1], {"weight": r_dict[li]}) for u_v, li in edge_line.items()])

    # Step 4: Find all cycles in the graph
    cycles = nx.cycle_basis(graph)
    cycles_increasing_similarity = sort_by_increasing_similarity(cycles)

    cycle_switches = []
    for cycle in cycles_increasing_similarity:
        cycle_lines = net.line.loc[net.line.from_bus.isin(cycle) & net.line.to_bus.isin(cycle)].index.values
        switches = net.switch.loc[net.switch.bus.isin(cycle) & net.switch.element.isin(cycle_lines) &
                                  (net.switch.et == 'l')].index.values
        cycle_switches.append([int(sw) for sw in switches])

    # Step 6: Generate trees using SpanningTreeIterator with decreasing sorting strategy
    tree_excluded_switches = []
    nr_of_cycles = len(cycle_switches)
    tree_count = 0
    similarities = []
    min_similarity = 1
    total_tree_count = 0
    for tree in SpanningTreeIterator(graph, minimum=True):
        total_tree_count += 1
        if tree_count >= n_trees:
            break
        dg = nx.difference(graph, tree)
        non_tree_switches = [edge_switches[(u, v)] for u, v in dg.edges()]
        if len(tree_excluded_switches) > 0:
            max_similarity = max(jaccard_similarity(non_tree_switches, tse) for tse in tree_excluded_switches)
            similarities.append(max_similarity)
            if min_similarity > (new_min := min(similarities)):
                min_similarity = new_min
                print(new_min, total_tree_count)
            if max_similarity > max_jaccard_similarity:
                continue
        ordered_tree_switches = reorder_flat_list_numba(cycle_switches, non_tree_switches)
        if all(si is not None for si in ordered_tree_switches):
            tree_excluded_switches.append(ordered_tree_switches)
        else:
            continue
        tree_count += 1
    return cycle_switches, tree_excluded_switches


def print_result_summary(net: pp.pandapowerNet) -> pd.DataFrame:
    """
    Print a result summary after a power flow!
    :param net: The pandapowerNet instance of which the results should be printed
    :return: A result Dataframe which is returned
    """
    print("Berechnungsergebnisse (Knotenspannungen und Betriebsmittelauslastungen):")
    bus_results = net.res_bus.loc[:, 'vm_pu'].describe()
    line_results = net.res_line.loc[:, 'loading_percent'].describe()
    trafo_results = net.res_trafo.loc[:, 'loading_percent'].describe()
    result_frame = pd.DataFrame(data={"Knotenspannungen in p.u.": bus_results,
                                      "Leitungsauslastung in %": line_results,
                                      "Transformatorauslastung in %": trafo_results})
    print(result_frame.T)
    return result_frame


def find_best_option_of_n_trees(net, n_trees, func, parameter, **kwargs):
    cycle_switches, switch_options = create_trees_with_cycle_switches(net, n_trees)
    overall_min = np.inf
    min_cost = np.inf
    best_option = None
    for i in progressbar(range(switch_options.shape[0])):
        switches = switch_options[i]
        net.switch.closed = True
        net.switch.loc[switches, 'closed'] = False
        pp.runpp(net)
        if not net_valid(net, **kwargs):
            continue
        min_current = func(net)
        if min_current <= overall_min:
            lines_cost = net.line.loc[net.switch.loc[switches, 'element'], 'cost'].sum()
            if lines_cost < min_cost:# or min_line_loading_topology < min_loading:
                best_option = switches
                overall_min = min_current
                min_cost = lines_cost
                print(f"\nNew minimum {round(min_current, 3)} {parameter}"
                      f"with total line cost {lines_cost} € "
                      f"for switch option {i}\n")
    if best_option is None:
        print("No valid grid option found!")
        return net
    net.switch.closed = True
    net.switch.loc[best_option, 'closed'] = False
    pp.runpp(net)
    pp.to_json(net, f'{n_trees}_{round(min_cost)}_{round(overall_min)}_{parameter}.json')
    print(np.array(best_option))
    return net

def max_line_loading(nw):
    return nw.res_line.loading_percent.max()


def sum_losses(nw):
    return nw.res_line.pl_mw.sum()


def s_ext_grid(net):
    return np.sqrt(net.res_ext_grid.p_mw.sum() ** 2 + 2 * net.res_ext_grid.q_mvar.sum() ** 2)


if __name__ == '__main__':

    np.random.seed(1234)



    net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\substation_planning_pp_grid.json")
    #Just see whats the best option with random trees! (without GA)
    #
    #!!!IDee:!!!
    #Wie wäre es wenn man die crossover/mutation ergebnisse nur verwendet um damit zufällige Bäume zu erstellen?
    #-> Die Äste die durch crossover und mutation prozesse entstehen, dienen nur der eingabe für random trees! die
    #durch den prozess rausfliegenden Äste werden mit hohem gewicht im graph berücksichtigt -> gewichtung der Kanten
    #anhand der vorkommnisse im Graph (siehe dynamic mutation!)
    #-> Vorteil: Keine topologieprüfungen notwendig! Power Grid Model kann Batchprocess durchführen!
    #
    #pp.runpp(self.net)
    #
    #if not net_valid(self.net, **self.net_valid_kwargs):
    find_best_option_of_n_trees(net, 100, s_ext_grid, 'apparent_power_slack_mva',
                                min_vm_pu=0.95, max_vm_pu=1.05, max_line_loading_percent=100.)
    print_result_summary(net)


