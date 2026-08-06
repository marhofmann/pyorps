import rustworkx as rx
import pandapower as pp
import numpy as np
from IPython.extensions.autoreload import update_function
from numba import njit, prange, bool as nb_bool
from itertools import pairwise
import pandas as pd
from geopandas import GeoDataFrame
from rustworkx.visit import BFSVisitor
from shapely.geometry import LineString


def net_valid(net, min_vm_pu=0.9, max_vm_pu=1.1,
              max_line_loading=100., max_trafo_loading=100.):
    """Check if network is valid with respect to operational constraints."""
    if voltage_bounds_violated(net, min_vm_pu, max_vm_pu):
        return False
    elif lines_overloaded(net, max_line_loading):
        return False
    elif trafos_overloaded(net, max_trafo_loading):
        return False
    else:
        return True


def voltage_bounds_violated(net, min_vm_pu=0.9, max_vm_pu=1.1):
    """Check if voltage bounds are violated."""
    return ((net.res_bus.vm_pu < min_vm_pu) |
            (net.res_bus.vm_pu > max_vm_pu)).any()


def lines_overloaded(net, max_line_loading=100.):
    """Check if any lines are overloaded."""
    return (net.res_line.loading_percent > max_line_loading).any()


def trafos_overloaded(net, max_trafo_loading=100.):
    """Check if any transformers are overloaded."""
    return (net.res_trafo.loading_percent > max_trafo_loading).any()

@njit()
def weighted_random_sample(values, n_samples, weights=None):
    """
    Generate a weighted random sample from an array of values.

    Parameters:
    values (array-like): The values to sample from
    n_samples (int): Number of samples to generate
    weights (array-like, optional): Weight for each value. If not provided, equal weights are used.

    Returns:
    numpy.ndarray: Array of randomly sampled values according to the weights
    """
    # Convert inputs to numpy arrays if they aren't already
    values = np.array(values)

    if weights is None:
        # If no weights provided, use uniform weights
        return np.random.choice(values, size=n_samples, replace=True)
    else:
        # Convert weights to numpy array if it isn't already
        weights = np.array(weights)

        # Normalize weights to probabilities (sum to 1)
        probabilities = weights / np.sum(weights)

        # Generate weighted random sample
        return np.random.choice(values, size=n_samples, replace=True, p=probabilities)


# Define Numba functions outside the class
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
            if _find_augmenting_path(other_list_idx, lol_array, lengths, flat_set,
                                     element_to_list, assigned_element, visited):
                element_to_list[elem] = list_idx
                assigned_element[list_idx] = elem
                return True

    return False


@njit()
def reorder_flat_list_core(lol_array, lengths, flat_list_unordered):
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
        _find_augmenting_path(list_idx, lol_array, lengths, flat_set,
                              element_to_list, assigned_element, visited)

    return assigned_element


@njit(parallel=True)
def check_array_match(flat_array, arrays_2d, arrays_to_check):
    """
    Check if flat_array matches any row in the 2D array, regardless of element order.

    Args:
        flat_array: 1D numpy array of integers
        arrays_2d: 2D numpy array where each row is compared with flat_array
        arrays_to_check: Number of arrays to check

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


def get_line_gdf(net, crs='EPSG:32632'):
    geoms = net.line.geo.apply(lambda x: LineString(eval(x)['coordinates']))
    lines = net.line.loc[:, ['length_km', 'from_bus', 'to_bus', 'std_type', 'max_i_ka', 'cost', 'parallel']]
    res_lines = net.res_line.loc[:, ['loading_percent', 'i_ka', 'i_from_ka', 'i_to_ka']]
    line_data = pd.concat([lines, res_lines], axis=1)
    gdf = GeoDataFrame(line_data, geometry=geoms, crs=crs)
    return gdf


class TreeEdgesRecorder(BFSVisitor):

    def __init__(self):
        self.edges = []

    def tree_edge(self, edge):
        self.edges.append(edge)


class TreeTopologyGenerator:
    """
    Modified version that avoids copying the network.
    """

    def __init__(self, net, evaluation_func, update_func, n_trees=10, **kwargs):
        """
        Initialize with the same parameters but now works with a single network instance.
        """
        self.net = net  # Use the original network
        self.evaluation_func = evaluation_func
        self.n_trees = n_trees
        self.update_func = update_func
        self.validation_params = kwargs
        self.total_line_cost = self.net.line.cost.sum()
        self.total_tree_count = {(fb, tb): 1 for fb, tb in zip(net.line.from_bus.values, net.line.to_bus.values)}

        # Save original switch states to restore after analysis
        self.original_switch_states = net.switch.closed.copy()

        # Will be set during initialization
        self.edge_line = None
        self.complete_edge_line = None
        self.line_switches = None
        self.edge_switch = None
        self.graph = None
        self.cycles = None
        self.cycle_switches = None
        self.nr_of_cycles = None
        self.r_dict = None
        self.edge_traversal_counts = None

        # Results
        self.tree_excluded_switches = None
        self.best_option = None
        self.overall_min = np.inf
        self.min_cost = np.inf

    # Most other methods can remain the same

    def _evaluate_tree(self, switch_config, tree_count):
        """
        Modified version that avoids copying the network.
        """
        # Save current switch states
        current_switch_states = self.net.switch.closed.copy()

        # Set all switches closed, then open the specified ones
        self.net.switch.closed = True
        self.net.switch.loc[switch_config, 'closed'] = False

        # Run power flow
        pp.runpp(self.net)

        # Check if network is valid with this configuration
        if net_valid(self.net, **self.validation_params):
            # Get evaluation metric
            metric_value = self.evaluation_func(self.net)

            lines = self.net.switch.loc[switch_config, 'element']
            # Calculate cost of the configuration
            lines_cost = self.total_line_cost - self.net.line.loc[lines, 'cost'].sum()

            # Check if this is a better solution
            if metric_value <= self.overall_min:
                if lines_cost < self.min_cost:
                    self.best_option = switch_config.copy()  # Make a copy of the switch config
                    self.overall_min = metric_value
                    self.min_cost = lines_cost
                    print(f"New minimum {round(metric_value, 3)} {self.evaluation_func.__name__} "
                          f"with total line cost {round(lines_cost)} € "
                          f"for switch option {tree_count}")

        # Restore original switch states if this isn't the best solution
        if switch_config is not self.best_option:
            self.net.switch.closed = current_switch_states

    def generate(self):
        """
        Modified version that avoids copying the network.
        """
        print(f"Extracting trees from a given pandapower network...")

        # Initialize mappings
        from_buses, to_buses = self._initialize_mappings()

        # Create graph
        self._create_graph(from_buses, to_buses)

        # Find cycles and switches
        self._find_cycles()

        # Generate trees
        try:
            self._generate_trees()
        except KeyboardInterrupt:
            pass

        # Apply best configuration before returning
        if self.best_option is not None:
            self.net.switch.closed = True
            self.net.switch.loc[self.best_option, 'closed'] = False
            pp.runpp(self.net)
            print("Applied best switching option.")
        else:
            # Restore original switch states if no valid solution was found
            self.net.switch.closed = self.original_switch_states
            print("No valid grid option found! Restored original switch states.")

        return self.net

    def tree_count_edges(self):
        switches = self.net.switch.loc[self.net.switch.closed, 'element']
        from_bus = self.net.line.loc[switches, 'from_bus']
        to_bus = self.net.line.loc[switches, 'to_bus']
        for fb, tb in zip(from_bus, to_bus):
            if (fb, tb) not in self.total_tree_count:
                self.total_tree_count[(fb, tb)] = 1
            else:
                self.total_tree_count[(fb, tb)] += 1
        total_count = sum(self.total_tree_count.values()) / max(self.total_tree_count.values())
        weights = np.array([total_count / w for w in self.total_tree_count.values()])

        return weights

    def _initialize_mappings(self):
        """Initialize mappings between lines, buses, and switches."""
        from_buses = self.net.line.from_bus.values
        to_buses = self.net.line.to_bus.values

        # Map from bus1-bus2 pair to line_idx
        self.edge_line = dict(zip(zip(self.net.line.from_bus, self.net.line.to_bus),
                                  self.net.line.index))
        self.complete_edge_line = dict(self.edge_line)
        self.complete_edge_line.update(dict(zip(zip(self.net.line.to_bus, self.net.line.from_bus),
                                                self.net.line.index)))

        # Map from line_idx to switch indices
        line_switches = self.net.switch.loc[self.net.switch.et == 'l']
        self.line_switches = dict(zip(line_switches.element, line_switches.index))

        self.edge_switch = {edge: self.line_switches[line]
                            for edge, line in self.complete_edge_line.items()
                            if line in self.line_switches}
        return from_buses, to_buses

    def _create_graph(self, from_buses, to_buses):
        """Create a graph representation of the network."""
        self.graph = rx.PyGraph()
        unique_nodes = list(range(np.unique(np.concatenate([from_buses, to_buses])).max() + 1))
        self.graph.add_nodes_from(unique_nodes)

        # Set resistance values for lines
        self.net.line['r_ohm'] = self.net.line.length_km * self.net.line.r_ohm_per_km
        self.r_dict = self.net.line.r_ohm.to_dict()

        # Add edges with resistance values as weights
        edge_data = [(u_v[0], u_v[1], self.r_dict[li])
                     for u_v, li in self.edge_line.items()]
        self.graph.add_edges_from(edge_data)

        # Add external grid connections with weight 0
        ext_grid_connection = [(s1, s2, 0) for s1, s2 in pairwise(self.net.ext_grid.bus)]
        self.graph.add_edges_from(ext_grid_connection)

        # Count number of possible spanning trees
        num_spanning_trees = self._count_spanning_trees()
        print(f"There are {num_spanning_trees} possible spanning trees which can be formed by this network!")

        # Remove external grid connections for tree generation
        self.graph.remove_edges_from([(u, v) for u, v, _ in ext_grid_connection])

        cm = rx.degree_centrality(self.graph)
        self.centrality = np.array([(cm[fb] + cm[tb]) for fb, tb in zip(from_buses, to_buses)])

    def _count_spanning_trees(self):
        """Count the number of spanning trees in the graph using Kirchhoff's Matrix Theorem."""
        # Get actual node indices (which might not be 0,1,2,...,n-1)
        node_indices = list(self.graph.node_indices())

        if len(node_indices) <= 1:
            return len(node_indices)  # 0 for empty graph, 1 for single node

        # Check connectivity
        if not rx.is_connected(self.graph):
            return 0

        # Create a mapping from actual node indices to matrix positions
        idx_to_pos = {idx: pos for pos, idx in enumerate(node_indices)}
        n = len(node_indices)

        # Initialize Laplacian matrix
        laplacian = np.zeros((n, n), dtype=float)

        # Handle diagonal entries (degrees)
        for i, node_idx in enumerate(node_indices):
            laplacian[i, i] = self.graph.degree(node_idx)

        # Handle off-diagonal entries (-1 for connected nodes)
        for node_idx in node_indices:
            # Get neighbors for this node
            for neighbor_idx in self.graph.neighbors(node_idx):
                i = idx_to_pos[node_idx]
                j = idx_to_pos[neighbor_idx]
                laplacian[i, j] = -1

        # Per Kirchhoff's theorem, get the cofactor by removing first row and column
        cofactor = laplacian[1:, 1:]

        # Calculate determinant
        det_value = np.linalg.det(cofactor)

        # Return the absolute value, rounded to an integer
        return int(round(abs(det_value)))

    def _find_cycles(self):
        """Find all cycles in the graph and identify switches in each cycle."""
        self.cycles = rx.cycle_basis(self.graph)

        self.cycle_switches = []
        for cycle in self.cycles:
            cycle_lines = self.net.line.loc[
                self.net.line.from_bus.isin(cycle) &
                self.net.line.to_bus.isin(cycle)
                ].index.values

            switches = self.net.switch.loc[
                self.net.switch.bus.isin(cycle) &
                self.net.switch.element.isin(cycle_lines) &
                (self.net.switch.et == 'l')
                ].index.values

            self.cycle_switches.append([int(sw) for sw in switches])

        self.nr_of_cycles = len(self.cycle_switches)

    def _reorder_flat_list(self, lol, flat_list_unordered):
        """
        Wrapper for the Numba-accelerated reorder_flat_list_core function.

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

        # Call the Numba-optimized standalone function
        return reorder_flat_list_core(lol_array, lengths, flat_list_unordered)

    def _generate_trees(self):
        """Generate different spanning trees by selecting switches to open."""
        self._tree_count = 0
        from_buses = self.net.line.from_bus.values
        to_buses = self.net.line.to_bus.values

        self.tree_excluded_switches = np.zeros((self.n_trees, self.nr_of_cycles), dtype=int)
        while self._tree_count < self.n_trees:
            # Generate minimum spanning tree
            mse = rx.minimum_spanning_edges(self.graph, weight_fn=float)
            arr = mse.__array__().T
            el = list(zip(arr[0], arr[1]))
            # Find co-tree edges (not in spanning tree)
            co_tree = np.array([
                self.edge_switch[(u, v)]
                for u, v in self.graph.edge_list()
                if (u, v) not in el and (v, u) not in el
            ])

            # Skip if not valid
            if co_tree.size != np.unique(co_tree).size or co_tree.size != self.nr_of_cycles:
                continue

            # Order tree switches using independent function
            ordered_tree_switches = self._reorder_flat_list(self.cycle_switches, co_tree)

            no_none = all(si is not None for si in ordered_tree_switches)
            if no_none and not check_array_match(co_tree, self.tree_excluded_switches, self._tree_count):
                self.tree_excluded_switches[self._tree_count] = ordered_tree_switches
                self._tree_count += 1

                self._evaluate_tree(ordered_tree_switches, self._tree_count)

            self._update_graph(from_buses, to_buses)
        return self.best_option

    def _create_shortest_path_mapping(self):
        """
        Creates a mapping of edges to how frequently they are used in shortest paths from each node
        to the closest slack bus (external grid).

        This method:
        1. Finds the shortest path from each node to the closest slack bus (ext_grid)
        2. Counts how many times each edge appears in these paths
        3. Creates a mapping with edge tuples as keys and traversal counts as values

        Returns:
        --------
        dict:
            A dictionary with edge tuples (from_node, to_node) as keys and traversal counts as values
        """
        # Get all slack buses (external grid connections)
        slack_buses = set(int(bus) for bus in self.net.ext_grid.bus.values)
        if not slack_buses:
            print("Warning: No slack buses (external grids) found in the network")
            self.edge_traversal_counts = {}
            return {}

        # Initialize edge traversal counts dictionary
        if self.edge_traversal_counts is None:
            zipper = zip(self.net.line.from_bus.values, self.net.line.to_bus.values)
            self.edge_traversal_counts = {(u, v): 1 for u, v in zipper}

        # Get all nodes in the graph
        all_nodes = list(self.graph.node_indices())

        # For each node, find shortest path to closest slack bus
        for node in all_nodes:
            if node in slack_buses:
                continue  # Skip slack buses themselves

            path = None
            for slack_bus in slack_buses: # Find shortest path to this slack bus
                removed = False
                if self.graph.has_edge(node, slack_bus):
                    ed = self.graph.get_edge_data(node, slack_bus)
                    self.graph.remove_edge(node, slack_bus)
                    removed = True
                slack_path = rx.dijkstra_shortest_paths(
                    self.graph,
                    node,
                    slack_bus,
                    weight_fn=float
                )
                if removed:
                    self.graph.add_edge(node, slack_bus, ed)
                try:
                    path = list(slack_path[slack_bus])
                    break
                except IndexError:
                    continue
            if not path:
                continue
            else:
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    reversed_edge = (path[i + 1], path[i])

                    # Update count for this edge (accounting for undirected edges)
                    if edge in self.edge_traversal_counts:
                        self.edge_traversal_counts[edge] += 1
                    else:
                        self.edge_traversal_counts[reversed_edge] += 1

        print(f"Created edge traversal mapping with {len(self.edge_traversal_counts)} edges")

        # Optional: Print some statistics about the edge traversals
        if self.edge_traversal_counts:
            counts = list(self.edge_traversal_counts.values())
            print(f"Edge traversal statistics: min={min(counts)}, max={max(counts)}, "
                  f"avg={sum(counts) / len(counts):.2f}")

        return self.edge_traversal_counts

    def _update_graph(self, from_buses, to_buses):
        """Update graph edges with weights from the update function."""
        # Clear edges and add with updated weights to get different trees
        self.graph.clear_edges()

        # Get weights from the update function
        weights = self.update_func(self.net)

        # Make sure we have the correct number of weights
        nr_elements = len(from_buses)
        if len(weights) != nr_elements:
            print(f"Warning: update_func returned {len(weights)} weights, but {nr_elements} were expected.")
            # Fill with random weights if necessary
            if len(weights) < nr_elements:
                additional = np.random.random_sample(nr_elements - len(weights))
                weights = np.concatenate([weights, additional])
            weights = weights[:nr_elements]

        # Add small random noise to avoid identical weights
        weights += np.random.random_sample(nr_elements) * 0.01

        # Create edge data with updated weights
        edge_data = list(zip(from_buses, to_buses, weights))
        self.graph.add_edges_from(edge_data)

        # Debug info
        #print(f"Graph updated with {len(edge_data)} edges and weights range: {np.min(weights):.4f} to
        # {np.max(weights):.4f}")

    def _end(self):
        if self.best_option is None:
            print("No valid grid option found!")
        else:
            print("Applying best switching option and saving before closing...")
            # Apply best configuration
            self.net.switch.closed = True
            self.net.switch.loc[self.best_option, 'closed'] = False
            pp.runpp(self.net)


            # Save best configuration
            file_name = (f'{self.n_trees}_{round(self.min_cost)}_{round(self.overall_min)}_'
                         f'{self.evaluation_func.__name__}_'
                         f'{self.update_func.__name__}')
            pp.to_json(self.net, file_name + '.json')
            get_line_gdf(net).to_file(file_name + '.geojson')

            print(f"Extracted {self._tree_count} different spanning trees from the net data!")
            print(np.array(self.best_option))
            print(f"Results saved to file:\n{file_name}")


def print_result_summary(net: pp.pandapowerNet) -> pd.DataFrame:
    print("Calculation results (node voltages and equipment utilization):")
    print(f"Slack:\n{net.res_ext_grid}")
    bus_results = net.res_bus.loc[:, 'vm_pu'].describe()
    line_results = net.res_line.loc[:, 'loading_percent'].describe()
    trafo_results = net.res_trafo.loc[:, 'loading_percent'].describe()
    result_frame = pd.DataFrame(data={"Node voltages in p.u.": bus_results,
                                      "Line loading in %": line_results,
                                      "Transformer loading in %": trafo_results})
    print(result_frame.T)
    return result_frame


def max_line_loading(nw):
    return nw.res_line.loading_percent.max()


def sum_losses(nw):
    return nw.res_line.pl_mw.sum()


def s_ext_grid_cost(net):
    cost = net.line.loc[net.switch.loc[~net.switch.closed, 'element'], 'cost'].sum() / 1_000_000
    return np.sqrt(net.res_ext_grid.p_mw.sum() ** 2 + net.res_ext_grid.q_mvar.sum() ** 2) + cost


def total_line_loading_squared(net):
    lines = net.switch.loc[net.switch.closed, 'element']
    return np.sum(np.power(net.res_line.loc[lines, 'loading_percent'].values, 2))


def s_ext_grid(net):
    return np.sqrt(net.res_ext_grid.p_mw.sum() ** 2 + net.res_ext_grid.q_mvar.sum() ** 2)


def random(net):
    weights = np.random.random_sample(net.line.shape[0])
    return weights


def random_choice(net):
    nr_elements = net.line.shape[0]
    sample = np.arange(1, 100, 1)
    weights = np.random.choice(sample, nr_elements)
    return weights


def cost_current(net):
    weights = np.random.random_sample(net.line.shape[0])
    return (net.res_line.i_ka / net.line.cost) * weights


def cost(net):
    weights = np.random.random_sample(net.line.shape[0])
    return (net.line.cost * weights) / 1000


def normalized_cost_current(net):
    lines = net.line
    res_lines = net.res_line
    normalized_cost = lines.cost / lines.cost.max()
    normalized_current = res_lines.i_ka / res_lines.i_ka.max()
    return (normalized_cost / normalized_current).values


def line_losses_s_mva(net):
    pl_mw = net.res_line.pl_mw.values
    ql_mvar = net.res_line.ql_mvar.values
    s_mva = np.sqrt(np.power(pl_mw, 2) + np.power(ql_mvar, 2))
    return np.sum(s_mva)


if __name__ == '__main__':
    np.random.seed(1234)


    net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\substation_planning_pp_grid.json")

    # Create the generator and generate trees
    generator = TreeTopologyGenerator(
        net=net,
        evaluation_func=s_ext_grid,
        update_func=normalized_cost_current,
        n_trees=50_000,
        min_vm_pu=0.95,
        max_vm_pu=1.05,
        max_line_loading_percent=50.,
    )

    optimized_net = generator.generate()

    # Print result summary
    print_result_summary(optimized_net)
    get_line_gdf(optimized_net).to_file("simulated_annealing.geojson")


