import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

def minimum_cut(graph):
    """
    Finds the minimum edge cut of a graph using the Ford-Fulkerson method (max-flow/min-cut).
    """
    if not nx.is_connected(graph):
        raise ValueError("The graph must be connected.")

    # Convert to a directed graph with uniform capacities
    directed_graph = nx.DiGraph()
    for u, v in graph.edges():
        directed_graph.add_edge(u, v, capacity=1)
        directed_graph.add_edge(v, u, capacity=1)

    # Use the first and last nodes as source and sink
    source = list(directed_graph.nodes)[0]
    sink = list(directed_graph.nodes)[-1]

    #  The minimum cut using Ford-Fulkerson from scratch
    residual = {edge: directed_graph.edges[edge]['capacity'] for edge in directed_graph.edges()}
    max_flow = 0

    def bfs(source, sink, parent):
        visited = {node: False for node in directed_graph.nodes()}
        queue = [source]
        visited[source] = True

        while queue:
            u = queue.pop(0)

            for v in directed_graph.neighbors(u):
                if not visited[v] and residual.get((u, v), 0) > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    parent = {}
    while bfs(source, sink, parent):
        # Find the maximum flow through the path found by BFS
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual[(parent[s], s)])
            s = parent[s]

        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            residual[(u, v)] -= path_flow
            residual[(v, u)] = residual.get((v, u), 0) + path_flow
            v = parent[v]

        max_flow += path_flow

    # Find reachable nodes in the residual graph
    visited = {node: False for node in directed_graph.nodes()}
    queue = [source]
    visited[source] = True
    while queue:
        u = queue.pop(0)
        for v in directed_graph.neighbors(u):
            if not visited[v] and residual.get((u, v), 0) > 0:
                queue.append(v)
                visited[v] = True

    # Find the edges that cross the cut
    reachable = [node for node in visited if visited[node]]
    non_reachable = [node for node in directed_graph.nodes() if not visited[node]]
    cut_edges = [(u, v) for u in reachable for v in directed_graph.neighbors(u) if v in non_reachable]

    return max_flow, cut_edges


def visualize_graph(graph, title):
    """
    Visualizes the graph.

    Args:
        graph (networkx.Graph): The graph to visualize.
        title (str): Title for the plot.
    """
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='black'
    )
    plt.title(title)
    plt.show()


def visualize_subgraphs(graph, cut_edges, title):
    """
    Visualizes the two resulting subgraphs after removing the cut edges.
    """
    graph_copy = graph.copy()
    graph_copy.remove_edges_from(cut_edges)
    components = list(nx.connected_components(graph_copy))

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))

    for component in components:
        subgraph = graph.subgraph(component)
        nx.draw(
            subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='black'
        )

    plt.title(title)
    plt.show()



def compute_degree_centrality(graph):
    """
    Computes the degree centrality for all nodes in a NetworkX graph.
    """
    n = len(graph)  # Total number of nodes
    return {node: graph.degree[node] / (n - 1) for node in graph.nodes}

from collections import deque
from collections import deque

def bfs(graph, start_node, return_paths=False):
    """
    Perform BFS to compute shortest path distances, and optionally path counts and predecessors(for the Betweenes Centrality).
    """
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    queue = deque([start_node])

    if return_paths:
        paths = {node: 0 for node in graph.nodes}
        paths[start_node] = 1
        predecessors = {node: [] for node in graph.nodes}

    while queue:
        current_node = queue.popleft()
        current_distance = distances[current_node]

        for neighbor in graph.neighbors(current_node):
            if distances[neighbor] == float('inf'):  # First visit
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

            if return_paths and distances[neighbor] == current_distance + 1:
                paths[neighbor] += paths[current_node]
                predecessors[neighbor].append(current_node)

    if return_paths:
        return distances, paths, predecessors
    else:
        return distances

def compute_closeness_centrality(graph):
    """
    Computes closeness centrality for all nodes in a graph using the modular BFS.
    """
    centrality = {}
    num_nodes = len(graph)

    for node in graph.nodes:
        shortest_paths = bfs(graph, node)  # Only distances are needed
        reachable_nodes = sum(1 for dist in shortest_paths.values() if dist < float('inf')) - 1

        if reachable_nodes > 0:
            total_distance = sum(dist for dist in shortest_paths.values() if dist < float('inf'))
            centrality[node] = reachable_nodes / total_distance
        else:
            centrality[node] = 0.0

    return centrality

def compute_betweenness_centrality(graph):
    """
    Computes betweenness centrality for all nodes in a graph using the modular BFS.
    """
    centrality = {node: 0.0 for node in graph.nodes}

    for source in graph.nodes:
        distances, paths, predecessors = bfs(graph, source, return_paths=True)
        dependency = {node: 0.0 for node in graph.nodes}
        sorted_nodes = sorted(distances, key=lambda x: -distances[x])

        for node in sorted_nodes:
            for pred in predecessors[node]:
                ratio = paths[pred] / paths[node]
                dependency[pred] += ratio * (1 + dependency[node])
            if node != source:
                centrality[node] += dependency[node]

    num_nodes = len(graph)
    normalization = (num_nodes - 1) * (num_nodes - 2)
    if normalization > 0:
        for node in centrality:
            centrality[node] /= normalization

    return centrality

def compute_pagerank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Computes the PageRank of nodes in a graph.
    """
    # Initialize variables
    num_nodes = len(graph)
    pagerank = {node: 1 / num_nodes for node in graph}  # Initial uniform rank
    new_pagerank = pagerank.copy()

    for iteration in range(max_iterations):
        for node in graph:
            rank_sum = 0
            for neighbor in graph.predecessors(node):  # Nodes pointing to the current node
                rank_sum += pagerank[neighbor] / graph.out_degree(neighbor)

            # PageRank formula with damping factor
            new_pagerank[node] = (1 - damping_factor) / num_nodes + damping_factor * rank_sum

        # Check for convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in graph)
        if diff < tolerance:
            break

        pagerank = new_pagerank.copy()  # Update for the next iteration

    return pagerank


def compute_single_source_dijkstra(graph, source, target):
    """
    Computes the shortest path using Dijkstra's algorithm for a graph.
    """
    import heapq

    # Priority queue to select the next node with the smallest distance
    priority_queue = [(0, source, [])]
    visited = set()

    while priority_queue:
        current_distance, current_node, path = heapq.heappop(priority_queue)

        # Skip if the node was already visited
        if current_node in visited:
            continue
        visited.add(current_node)

        # Update the path
        path = path + [current_node]

        # If we reach the target, return the distance and path
        if current_node == target:
            return current_distance, path

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                edge_weight = graph[current_node][neighbor].get('weight', 1)  # Default weight is 1 if not provided
                heapq.heappush(priority_queue, (current_distance + edge_weight, neighbor, path))

    # If no path is found
    return float('inf'), []


def create_graph_with_distance(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        origin = row["Origin_city"]
        destination = row["Destination_city"]
        distance = row["Distance"]

        # Add edge with distance as the weight
        if not pd.isna(distance):
            G.add_edge(origin, destination, weight=distance)
    return G


def louvain_algorithm_numpy(graph):
    """Louvain community detection using NumPy for performance."""
    # Convert graph to adjacency matrix
    nodes = list(graph.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")

    # Initialize communities: each node starts in its own community
    num_nodes = len(nodes)
    communities = np.arange(num_nodes)

    def modularity_gain(i, j, adjacency_matrix, communities, m):
        """Compute the modularity gain for moving node i to community of node j."""
        k_i = adjacency_matrix[i].sum()
        k_j = adjacency_matrix[j].sum()
        delta_q = adjacency_matrix[i, j] - (k_i * k_j) / (2 * m)
        return delta_q

    # Total weight of edges in the graph
    m = adjacency_matrix.sum() / 2

    while True:
        improvement = False

        for i in range(num_nodes):
            current_community = communities[i]
            best_community = current_community
            max_gain = 0

            # Check modularity gain for moving to neighboring communities
            for j in range(num_nodes):
                if i != j and adjacency_matrix[i, j] > 0:
                    gain = modularity_gain(i, j, adjacency_matrix, communities, m)
                    if gain > max_gain:
                        max_gain = gain
                        best_community = communities[j]

            # Move to the best community if it improves modularity
            if best_community != current_community:
                communities[i] = best_community
                improvement = True

        if not improvement:
            break

    # Normalize community IDs and group nodes by community
    unique_communities = {community: idx for idx, community in enumerate(np.unique(communities))}
    community_dict = {unique_communities[community]: [] for community in unique_communities}

    for node, community in zip(nodes, communities):
        community_dict[unique_communities[community]].append(node)

    return community_dict


def analyze_flight_network(graph, city1, city2):
    """Analyze the flight network and provide community insights."""
    # Detect communities
    communities = louvain_algorithm_numpy(graph)

    # Output total number of communities
    print(f"Total number of communities: {len(communities)}")
    for community_id, cities in communities.items():
        print(f"Community {community_id}: {cities}")

    # Check if city1 and city2 are in the same community
    same_community = any(city1 in cities and city2 in cities for cities in communities.values())
    print(f"Cities {city1} and {city2} are in the same community: {same_community}")

    # Visualize the graph with communities
    visualize_communities(graph, communities)


def visualize_communities(graph, communities):
    """Visualize the graph with different colors for each community."""
    pos = nx.spring_layout(graph)
    color_map = []
    node_to_community = {node: community_id for community_id, nodes in communities.items() for node in nodes}

    for node in graph.nodes():
        color_map.append(node_to_community[node])

    nx.draw(
        graph, pos, node_color=color_map, with_labels=True, cmap=plt.cm.rainbow, node_size=500
    )
    plt.show()
