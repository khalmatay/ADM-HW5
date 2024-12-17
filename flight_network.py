import pandas as pd
from typing import Set, Tuple, Dict, List
from collections import defaultdict

class FlightNetwork:
    def __init__(self):
        """
        Inizializza una rete di voli con:
        - Nodi (aeroporti)
        - Archi (connessioni tra aeroporti)
        - Dizionari per archi entranti e uscenti
        """
        self.nodes: Set[str] = set()  # Insieme degli aeroporti
        self.edges: Set[Tuple[str, str]] = set()  # Insieme delle connessioni
        self.in_edges: Dict[str, List[str]] = defaultdict(list)  # Archi entranti
        self.out_edges: Dict[str, List[str]] = defaultdict(list)  # Archi uscenti

    def add_nodes_and_edges(self, origin_airports: pd.Series, destination_airports: pd.Series) -> None:
        """
        Aggiunge nodi (aeroporti) e archi (voli) alla rete.
        :param origin_airports: Serie Pandas contenente gli aeroporti di origine.
        :param destination_airports: Serie Pandas contenente gli aeroporti di destinazione.
        """
        # Aggiunge nodi unici
        self.nodes.update(origin_airports)
        self.nodes.update(destination_airports)
        
        # Aggiunge archi tra coppie di aeroporti
        self.edges.update(zip(origin_airports, destination_airports))
        
        # Aggiorna i dizionari di archi entranti e uscenti
        for origin, destination in zip(origin_airports, destination_airports):
            self.in_edges[destination].append(origin)
            self.out_edges[origin].append(destination)

    def in_degree(self, node: str) -> int:
        """
        Calcola il numero di archi entranti per un nodo.
        :param node: Nome del nodo (aeroporto)
        :return: Numero di archi entranti
        """
        return len(self.in_edges[node])

    def out_degree(self, node: str) -> int:
        """
        Calcola il numero di archi uscenti per un nodo.
        :param node: Nome del nodo (aeroporto)
        :return: Numero di archi uscenti
        """
        return len(self.out_edges[node])


