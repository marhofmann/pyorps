
from check_tree_topologies import *
import matplotlib.pyplot as plt


def simulated_annealing_weights(net, temperature, best_metrics=None, switch_config=None,
                                alpha=0.3, beta=0.4, gamma=0.3):
    """
    Generate edge weights using simulated annealing principles without copying the network.

    Parameters:
    -----------
    net : pandapower.auxiliary.pandapowerNet
        Current network state
    temperature : float
        Current annealing temperature (0.0 to 1.0)
    best_metrics : dict, optional
        Dictionary with best metrics found so far
    switch_config : array-like, optional
        Best switch configuration found so far
    alpha, beta, gamma : float
        Weights for different criteria (cost, loading, power demand)

    Returns:
    --------
    numpy.ndarray
        Edge weights for graph
    """
    nr_elements = net.line.shape[0]

    # Base weights - if no previous solutions, use random weights
    if best_metrics is None:
        weights = np.random.random_sample(nr_elements)
        return weights

    # --- Calculate guided components ---

    # 1. Cost component - prefer cheaper lines
    cost_factor = 1.0 / (net.line.cost.values + 1e-6)
    norm_cost = cost_factor / np.max(cost_factor)

    # 2. Line loading component - prefer less loaded lines
    if 'loading_percent' in net.res_line.columns:
        loading = net.res_line.loading_percent.values.copy()
        # We want to avoid highly loaded lines
        loading_factor = 1.0 / (loading + 1e-6)
        norm_loading = loading_factor / np.max(loading_factor)
    else:
        norm_loading = np.ones(nr_elements)

    # 3. Usage component - prefer lines used in previous best solution
    if switch_config is not None:
        # Lines not in the switch_config are part of the tree (closed switches)
        all_switches = set(range(len(net.switch)))
        tree_switches = set(all_switches) - set(switch_config)

        # Lines corresponding to closed switches are preferred
        tree_lines = net.switch.loc[list(tree_switches), 'element'].values

        # Create an indicator for preferred lines
        usage_factor = np.ones(nr_elements) * 0.5
        usage_factor[np.isin(np.arange(nr_elements), tree_lines)] = 1.0
        norm_usage = usage_factor
    else:
        norm_usage = np.ones(nr_elements)

    # --- Combine components with weights ---
    guided_weights = (alpha * norm_cost +
                      beta * norm_loading +
                      gamma * norm_usage)

    # --- Add randomness based on temperature ---
    # High temperature = more random, low temperature = more guided
    random_weights = np.random.random_sample(nr_elements)
    final_weights = temperature * random_weights + (1 - temperature) * guided_weights

    # Add small random noise to prevent identical weights
    final_weights += np.random.random_sample(nr_elements) * 0.01

    return final_weights


def enhanced_simulated_annealing_weights(net, temperature, best_metrics=None, switch_config=None,
                                         alpha=0.3, beta=0.4, gamma=0.3):
    """
    Generate edge weights with enhanced diversity for simulated annealing.

    Parameters:
    -----------
    net : pandapower.auxiliary.pandapowerNet
        Current network state
    temperature : float
        Current annealing temperature (0.0 to 1.0)
    best_metrics : dict, optional
        Dictionary with best metrics found so far
    switch_config : array-like, optional
        Best switch configuration found so far

    Returns:
    --------
    numpy.ndarray
        Edge weights for graph
    """
    nr_elements = net.line.shape[0]

    # Base weights - if no previous solutions, use random weights
    if best_metrics is None:
        return np.random.random_sample(nr_elements)

    # --- Calculate guided components ---

    # 1. Cost component - prefer cheaper lines
    cost_factor = 1.0 / (net.line.cost.values + 1e-6)
    norm_cost = cost_factor / np.max(cost_factor)

    # 2. Line loading component - prefer less loaded lines
    if 'loading_percent' in net.res_line.columns:
        loading = net.res_line.loading_percent.values.copy()
        # We want to avoid highly loaded lines
        loading_factor = 1.0 / (loading + 1e-6)
        norm_loading = loading_factor / np.max(loading_factor)
    else:
        norm_loading = np.ones(nr_elements)

    # 3. Usage component - prefer lines used in previous best solution but with temperature-based exploration
    if switch_config is not None:
        # Lines not in switch_config are part of the tree (closed switches)
        all_switches = set(range(len(net.switch)))
        tree_switches = set(all_switches) - set(switch_config)

        # Lines corresponding to closed switches are preferred
        tree_lines = net.switch.loc[list(tree_switches), 'element'].values

        # Create an indicator for preferred lines with temperature-based exploration
        # Higher temperature -> more uniform weights (more exploration)
        # Lower temperature -> more biased towards best known solution (more exploitation)
        base_value = 0.5 + 0.5 * temperature  # Starts at 1.0 (all equal) and decreases to 0.5
        usage_factor = np.ones(nr_elements) * base_value

        # Adjust weights based on temperature
        boost = 1.0 + (1.0 - temperature)  # Starts at 1.0 and increases to 2.0
        usage_factor[np.isin(np.arange(nr_elements), tree_lines)] *= boost

        norm_usage = usage_factor / np.max(usage_factor)
    else:
        norm_usage = np.ones(nr_elements)

    # --- Inject some diversity based on line properties ---
    # Use line length and capacity to create diverse weights
    # Shorter lines with higher capacity are generally preferred
    if 'length_km' in net.line.columns and 'max_i_ka' in net.line.columns:
        length = net.line.length_km.values
        capacity = net.line.max_i_ka.values

        # Normalize
        norm_length = 1.0 - (length / (np.max(length) + 1e-6))  # Shorter is better
        norm_capacity = capacity / (np.max(capacity) + 1e-6)  # Higher is better

        # Combine
        diversity_factor = 0.5 * norm_length + 0.5 * norm_capacity

        # Scale by temperature (more important at higher temps for exploration)
        diversity_weight = temperature * 0.3  # Up to 30% impact
    else:
        diversity_factor = np.ones(nr_elements)
        diversity_weight = 0.0

    # --- Combine components with weights ---
    # Adjust alpha, beta, gamma based on temperature to favor exploration at high temps
    # and exploitation at low temps
    adjusted_alpha = alpha * (1 + 0.5 * (1 - temperature))  # Cost becomes more important at low temps
    adjusted_beta = beta
    adjusted_gamma = gamma * (1 + temperature)  # Usage patterns more important at high temps

    # Normalize the adjusted weights
    total = adjusted_alpha + adjusted_beta + adjusted_gamma + diversity_weight
    adjusted_alpha /= total
    adjusted_beta /= total
    adjusted_gamma /= total
    diversity_weight /= total

    guided_weights = (
            adjusted_alpha * norm_cost +
            adjusted_beta * norm_loading +
            adjusted_gamma * norm_usage +
            diversity_weight * diversity_factor
    )

    # --- Add randomness based on temperature ---
    # High temperature = more random, low temperature = more guided
    random_weights = np.random.random_sample(nr_elements)

    # Use exponential decay for temperature influence to focus on good solutions faster
    temp_factor = np.exp(-3 * (1 - temperature))  # Starts near 1.0, decays to ~0.05
    final_weights = temp_factor * random_weights + (1 - temp_factor) * guided_weights

    # Add small random noise to prevent identical weights
    final_weights += np.random.random_sample(nr_elements) * 0.01

    return final_weights


class SimulatedAnnealingTreeFinder(TreeTopologyGenerator):
    """
    Tree topology generator that uses simulated annealing to guide the search.

    Inherits from TreeTopologyGenerator to leverage existing tree generation logic
    while adding temperature-based exploration vs. exploitation control.
    """

    def __init__(self, net, evaluation_func, initial_temp=1.0, final_temp=0.01,
                 cooling_rate=0.95, iterations_per_temp=10, alpha=0.3, beta=0.4, gamma=0.3, **kwargs):
        """
        Initialize the simulated annealing tree finder.

        Parameters:
        -----------
        net : pandapower.auxiliary.pandapowerNet
            The pandapower network (will be modified in-place)
        evaluation_func : callable
            Function to evaluate the network quality
        initial_temp : float
            Initial temperature (0.0 to 1.0)
        final_temp : float
            Final temperature (0.0 to 1.0)
        cooling_rate : float
            Rate at which temperature decreases (0.0 to 1.0)
        iterations_per_temp : int
            Number of iterations at each temperature
        alpha, beta, gamma : float
            Weights for different criteria (cost, loading, power demand)
        **kwargs : dict
            Additional parameters for validation (e.g., min_vm_pu, max_line_loading_percent)
        """
        # Initialize parent class with a small n_trees since we'll control iterations ourselves
        super().__init__(net=net, evaluation_func=evaluation_func,
                         update_func=self._weight_update_func, n_trees=1, **kwargs)

        # Simulated annealing parameters
        self.initial_temp = initial_temp
        self.temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Track optimization progress
        self.best_metric = np.inf
        self.best_switch_config = None
        self.best_switch_states = None
        self.best_metrics = None

        # Store statistics
        self.iteration_history = []
        self.temp_history = []
        self.metric_history = []

        # Set of evaluated configurations to avoid duplicates
        self.evaluated_configs = set()

    def _weight_update_func(self, net):
        """
        Weight update function that uses simulated annealing principles.

        This method is passed to the parent class's initialization to be used
        during graph edge weight updates.
        """
        nr_elements = net.line.shape[0]

        # Base weights - if no previous solutions, use random weights
        if self.best_metrics is None:
            weights = np.random.random_sample(nr_elements)
            return weights

        # --- Calculate guided components ---

        # 1. Cost component - prefer cheaper lines
        cost_factor = 1.0 / (net.line.cost.values + 1e-6)
        norm_cost = cost_factor / np.max(cost_factor)

        # 2. Line loading component - prefer less loaded lines
        if 'loading_percent' in net.res_line.columns:
            loading = net.res_line.loading_percent.values.copy()
            # We want to avoid highly loaded lines
            loading_factor = 1.0 / (loading + 1e-6)
            norm_loading = loading_factor / np.max(loading_factor)
        else:
            norm_loading = np.ones(nr_elements)

        # 3. Usage component - prefer lines used in previous best solution but with temperature-based exploration
        if self.best_switch_config is not None:
            # Lines not in switch_config are part of the tree (closed switches)
            all_switches = set(range(len(net.switch)))
            tree_switches = set(all_switches) - set(self.best_switch_config)

            # Lines corresponding to closed switches are preferred
            tree_lines = net.switch.loc[list(tree_switches), 'element'].values

            # Create an indicator for preferred lines with temperature-based exploration
            # Higher temperature -> more uniform weights (more exploration)
            # Lower temperature -> more biased towards best known solution (more exploitation)
            base_value = 0.5 + 0.5 * self.temp  # Starts at 1.0 (all equal) and decreases to 0.5
            usage_factor = np.ones(nr_elements) * base_value

            # Adjust weights based on temperature
            boost = 1.0 + (1.0 - self.temp)  # Starts at 1.0 and increases to 2.0
            usage_factor[np.isin(np.arange(nr_elements), tree_lines)] *= boost

            norm_usage = usage_factor / np.max(usage_factor)
        else:
            norm_usage = np.ones(nr_elements)

        # --- Combine components with weights ---
        guided_weights = (self.alpha * norm_cost +
                          self.beta * norm_loading +
                          self.gamma * norm_usage)

        # --- Add randomness based on temperature ---
        # High temperature = more random, low temperature = more guided
        random_weights = np.random.random_sample(nr_elements)
        final_weights = self.temp * random_weights + (1 - self.temp) * guided_weights

        # Add small random noise to prevent identical weights
        final_weights += np.random.random_sample(nr_elements) * 0.01

        return final_weights

    def _update_graph(self, from_buses, to_buses):
        """
        Override parent's _update_graph method to properly use our weight update function.
        """
        # Clear edges and add with updated weights
        self.graph.clear_edges()

        # Get weights from our weight update function with current temperature
        weights = self._weight_update_func(self.net)

        # Make sure we have the correct number of weights
        nr_elements = len(from_buses)
        if len(weights) != nr_elements:
            print(f"Warning: update_func returned {len(weights)} weights, but {nr_elements} were expected.")
            weights = weights[:nr_elements] if len(weights) > nr_elements else np.concatenate(
                [weights, np.random.random_sample(nr_elements - len(weights))])

        # Create edge data with weights
        edge_data = list(zip(from_buses, to_buses, weights))
        self.graph.add_edges_from(edge_data)

        print(f"Graph updated with weights range: {np.min(weights):.4f} to {np.max(weights):.4f}")

    def save_switch_states(self):
        """Save the current switch states."""
        return self.net.switch.closed.copy()

    def restore_switch_states(self, switch_states):
        """Restore switch states from a saved configuration."""
        self.net.switch.closed = switch_states.copy()

    def _evaluate_tree_with_annealing(self, switch_config):
        """
        Evaluate a tree configuration using simulated annealing principles.

        This extends the regular evaluation with acceptance probability for worse solutions.
        """
        # Save current switch states
        current_switch_states = self.save_switch_states()

        # Set all switches closed, then open the specified ones
        self.net.switch.closed = True
        self.net.switch.loc[switch_config, 'closed'] = False

        # Run power flow
        try:
            pp.runpp(self.net)
        except:
            print("Power flow did not converge, skipping this configuration")
            self.restore_switch_states(current_switch_states)
            return False

        # Check if network is valid
        if not net_valid(self.net, **self.validation_params):
            print("Network constraints violated, skipping this configuration")
            self.restore_switch_states(current_switch_states)
            return False

        # Calculate metrics
        current_metric = self.evaluation_func(self.net)

        # Calculate cost for this configuration
        lines = self.net.switch.loc[switch_config, 'element']
        lines_cost = self.total_line_cost - self.net.line.loc[lines, 'cost'].sum()

        # Check if this is better than our best solution
        improved = False
        if current_metric < self.best_metric:
            print(f"New best solution: {current_metric:.4f} (previous: {self.best_metric:.4f})")
            self.best_metric = current_metric
            self.best_switch_config = switch_config.copy()
            self.best_switch_states = self.save_switch_states()
            improved = True

            # Store metrics for guiding future searches
            self.best_metrics = {
                'cost': lines_cost,
                'loading': self.net.res_line.loading_percent.max(),
                'power_demand': np.sqrt(
                    self.net.res_ext_grid.p_mw.sum() ** 2 + self.net.res_ext_grid.q_mvar.sum() ** 2)
            }
        else:
            # Accept worse solution with probability based on temperature and difference
            delta = current_metric - self.best_metric
            acceptance_prob = np.exp(-delta / self.temp)

            if np.random.random() < acceptance_prob:
                print(f"Accepting worse solution ({current_metric:.4f}) with probability {acceptance_prob:.4f}")
                self.best_metric = current_metric
                self.best_switch_config = switch_config.copy()
                self.best_switch_states = self.save_switch_states()
                improved = True

                # Store metrics for guiding future searches
                self.best_metrics = {
                    'cost': lines_cost,
                    'loading': self.net.res_line.loading_percent.max(),
                    'power_demand': np.sqrt(
                        self.net.res_ext_grid.p_mw.sum() ** 2 + self.net.res_ext_grid.q_mvar.sum() ** 2)
                }
            else:
                # Restore previous best switch states if we have one
                if self.best_switch_states is not None:
                    self.restore_switch_states(self.best_switch_states)
                    pp.runpp(self.net)

        return improved

    def run(self, max_iterations=1000):
        """
        Run the simulated annealing optimization.

        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations

        Returns:
        --------
        pandapower.auxiliary.pandapowerNet
            Best network configuration found (original network modified in-place)
        """
        print("Starting simulated annealing optimization...")
        iteration = 0

        # Initialize mappings and create graph - from parent class methods
        from_buses, to_buses = self._initialize_mappings()
        self._create_graph(from_buses, to_buses)
        self._find_cycles()

        print(f"Found {len(self.cycles)} cycles in the network")

        # Main simulated annealing loop
        try:
            while self.temp > self.final_temp and iteration < max_iterations:
                print(f"\nTemperature: {self.temp:.4f}, Iteration: {iteration}")

                # Generate trees at this temperature
                for t in range(self.iterations_per_temp):
                    if iteration >= max_iterations:
                        break

                    # Update graph with current temperature-based weights
                    self._update_graph(from_buses, to_buses)

                    # Generate minimum spanning tree
                    mse = rx.minimum_spanning_edges(self.graph, weight_fn=float)
                    arr = mse.__array__().T
                    el = list(zip(arr[0], arr[1]))

                    # Find co-tree edges (not in spanning tree)
                    try:
                        co_tree = np.array([
                            self.edge_switch[(u, v)]
                            for u, v in self.graph.edge_list()
                            if (u, v) not in el and (v, u) not in el and (u, v) in self.edge_switch
                        ])
                    except KeyError:
                        # Sometimes edges might not be in edge_switch mapping
                        print("Key error in edge_switch mapping, skipping this tree")
                        continue

                    # Skip if not valid
                    if co_tree.size != np.unique(co_tree).size or co_tree.size != self.nr_of_cycles:
                        print(f"Invalid tree: found {co_tree.size} co-tree edges, need {self.nr_of_cycles}")
                        continue

                    # Order tree switches using independent function
                    ordered_tree_switches = self._reorder_flat_list(self.cycle_switches, co_tree)

                    # Check if all elements are assigned
                    no_none = all(si is not None for si in ordered_tree_switches)
                    if not no_none:
                        print("Could not assign all cycle switches")
                        continue

                    # Convert to tuple for set membership check
                    config_tuple = tuple(sorted(ordered_tree_switches))

                    # Skip if we've already evaluated this configuration
                    if config_tuple in self.evaluated_configs:
                        print("Skipping duplicate configuration")
                        continue

                    # Add to evaluated configurations
                    self.evaluated_configs.add(config_tuple)

                    # Evaluate using simulated annealing principles
                    self._evaluate_tree_with_annealing(ordered_tree_switches)

                    # Record history
                    self.iteration_history.append(iteration)
                    self.temp_history.append(self.temp)
                    self.metric_history.append(self.best_metric)

                    iteration += 1

                # Cool down after processing iterations_per_temp configurations
                self.temp *= self.cooling_rate
                print(f"Cooling down to temperature: {self.temp:.4f}")

            print(f"\nOptimization completed after {iteration} iterations")
            print(f"Best metric value: {self.best_metric:.4f}")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")

        # Apply best configuration to the network if not already applied
        if self.best_switch_states is not None:
            print("Applying best switch configuration...")
            self.restore_switch_states(self.best_switch_states)
            pp.runpp(self.net)
        else:
            # Restore original switch states if no valid solution was found
            print("No valid solution found, restoring original switch states")
            self.restore_switch_states(self.original_switch_states)
            pp.runpp(self.net)

        return self.net

    def plot_optimization_progress(self, save_path=None):
        """
        Plot the optimization progress.

        Parameters:
        -----------
        save_path : str, optional
            If provided, save the plot to this file
        """
        try:

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Plot metric history
            ax1.plot(self.iteration_history, self.metric_history, 'b-')
            ax1.set_ylabel('Metric Value')
            ax1.set_title('Optimization Progress')
            ax1.grid(True)

            # Plot temperature history
            ax2.plot(self.iteration_history, self.temp_history, 'r-')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Temperature')
            ax2.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")

            plt.show()

        except ImportError:
            print("Matplotlib not available, cannot plot optimization progress")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    np.random.seed(1234)
    # Load the network
    net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\substation_planning_pp_grid.json")


    # Define evaluation function
    def s_ext_grid(net):
        return np.sqrt(net.res_ext_grid.p_mw.sum() ** 2 + net.res_ext_grid.q_mvar.sum() ** 2)

    def vd_sum(net):
        vm = net.res_bus.vm_pu.values
        return np.sum(np.abs(1 - vm))

    def vd(net):
        vm = net.res_bus.vm_pu.values
        return np.max(np.abs(1 - vm))


    def line_losses_s_mva(net):
        pl_mw = net.res_line.pl_mw.values
        ql_mvar = net.res_line.ql_mvar.values
        s_mva = np.sqrt(np.power(pl_mw, 2) + np.power(ql_mvar, 2))
        return np.sum(s_mva)


    # Create simulated annealing optimizer using inheritance
    sa_optimizer = SimulatedAnnealingTreeFinder(
        net=net,
        evaluation_func=vd,  # Primary metric
        initial_temp=1.0,
        final_temp=0.0001,
        cooling_rate=0.995,
        iterations_per_temp=150,
        # Criteria weights
        alpha=0.4,  # Cost weight
        beta=0.2,    # Line loading weight
        gamma=0.4,  # Power demand weight
        # Validation parameters:
        min_vm_pu=0.95,
        max_vm_pu=1.05,
        max_line_loading=50.,
    )

    # Run the optimization - this modifies the network in-place
    sa_optimizer.run(max_iterations=100_000)

    # Print results - network is already modified with the best configuration
    print_result_summary(net)

    # Optional: Save the optimized network
    pp.to_json(net, 'optimized_network_line_losses.json')
    net.line.loc[net.switch.loc[~net.switch.closed].element, 'in_service'] = False
    pp.runpp(net)
    lines = get_line_gdf(net)
    lines.to_file("simulated_annealing_line_losses.geojson")

    sa_optimizer.plot_optimization_progress()

    lines.plot(column="loading_percent", legend=True)
    plt.show()

