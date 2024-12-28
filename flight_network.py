import pandas as pd
from typing import Set, Tuple, Dict, List
from collections import defaultdict

class FlightNetwork:
    def __init__(self):
        """
        Initializes a flight network with:
        - Nodes (airports)
        - Edges (connections between airports)
        - Dictionaries for incoming and outgoing edges
        """
        self.nodes: Set[str] = set()  # Set of airports
        self.edges: Set[Tuple[str, str]] = set()  # Set of connections
        self.in_edges: Dict[str, List[str]] = defaultdict(list)  # Incoming edges
        self.out_edges: Dict[str, List[str]] = defaultdict(list)  # Outgoing edges

    def add_nodes_and_edges(self, origin_airports: pd.Series, destination_airports: pd.Series) -> None:
        """
        Adds nodes (airports) and edges (flights) to the network.
        :param origin_airports: Pandas Series containing origin airports.
        :param destination_airports: Pandas Series containing destination airports.
        """
        # Add unique nodes
        self.nodes.update(origin_airports)
        self.nodes.update(destination_airports)
        
        # Add edges between pairs of airports
        self.edges.update(zip(origin_airports, destination_airports))
        
        # Update dictionaries for incoming and outgoing edges
        for origin, destination in zip(origin_airports, destination_airports):
            self.in_edges[destination].append(origin)
            self.out_edges[origin].append(destination)

    def in_degree(self, node: str) -> int:
        """
        Calculates the number of incoming edges for a node.
        :param node: Name of the node (airport)
        :return: Number of incoming edges
        """
        return len(self.in_edges[node])

    def out_degree(self, node: str) -> int:
        """
        Calculates the number of outgoing edges for a node.
        :param node: Name of the node (airport)
        :return: Number of outgoing edges
        """
        return len(self.out_edges[node])




    def create_subgraph(self, sampled_nodes: List[str]) -> 'FlightNetwork':
        """
        Creates a subgraph from the sampled nodes.
        :param sampled_nodes: List of sampled node (airport) names.
        :return: A new FlightNetwork instance representing the subgraph.
        """
        subgraph = FlightNetwork()
        
        # Add the sampled nodes to the subgraph
        subgraph.nodes.update(sampled_nodes)
        
        # Filter edges to include only those that have both ends in the sampled nodes
        for origin, destination in self.edges:
            if origin in sampled_nodes and destination in sampled_nodes:
                subgraph.edges.add((origin, destination))
                subgraph.out_edges[origin].append(destination)
                subgraph.in_edges[destination].append(origin)
        
        return subgraph