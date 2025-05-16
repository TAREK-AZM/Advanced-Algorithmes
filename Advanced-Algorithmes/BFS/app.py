import json
import heapq

class BestFirstSearch:
    def __init__(self, graph, heuristics):
        self.graph = graph
        self.heuristics = heuristics
    
    def search(self, start, goal):
        # Priority queue for nodes to explore
        frontier = []
        heapq.heappush(frontier, (self.heuristics[start], start))
        
        # Dictionary to keep track of path
        came_from = {start: None}
        # Dictionary to keep track of actual cost from start
        cost_so_far = {start: 0}
        
        while frontier:
            current_heuristic, current_node = heapq.heappop(frontier)
            
            # Check if we've reached the goal
            if current_node == goal:
                break
            
            # Explore neighbors
            for neighbor, distance in self.graph[current_node].items():
                new_cost = cost_so_far[current_node] + distance
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (self.heuristics[neighbor], neighbor))
                    came_from[neighbor] = current_node
        
        # Reconstruct path and calculate total cost
        path = []
        path_cost = 0
        current = goal
        while current != start:
            path.append(current)
            next_node = came_from[current]
            path_cost += self.graph[next_node][current]
            current = next_node
        path.append(start)
        path.reverse()
        
        return path, path_cost, self.heuristics[start]

def load_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def main():
    # Load graph and heuristics from JSON files
    try:
        graph = load_json_file('graph.json')
        heuristics = load_json_file('heuristics.json')
    except FileNotFoundError:
        print("Erreur: Les fichiers graph.json et heuristics.json doivent être présents.")
        return
    except json.JSONDecodeError:
        print("Erreur: Fichiers JSON mal formés.")
        return
    
    # Create BFS instance
    bfs = BestFirstSearch(graph, heuristics)
    
    # User interface
    print("Villes disponibles:", ", ".join(graph.keys()))
    start = input("Ville de départ: ").strip()
    goal = input("Ville d'arrivée: ").strip()
    
    if start not in graph or goal not in graph:
        print("Erreur: Ville non reconnue.")
        return
    
    # Perform search
    path, total_cost, heuristic_value = bfs.search(start, goal)
    
    # Display results
    print("\nChemin trouvé:", " -> ".join(path))
    print("Coût total du chemin (distance réelle):", total_cost)
    print("Valeur heuristique initiale (estimation):", heuristic_value)
    print("Nombre d'étapes:", len(path) - 1)

if __name__ == "__main__":
    main()