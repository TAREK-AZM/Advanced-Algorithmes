import heapq
import pygame
import sys
import time
import random
from typing import List, Tuple, Dict, Set

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 40
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Best-First Search Pathfinding")

class Node:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.g_cost = 0  # Cost from start to current node
        self.h_cost = 0  # Heuristic cost (estimated cost to goal)
        self.f_cost = 0  # f = g + h
        self.neighbors = []
        self.parent = None
        self.is_wall = False
        
    def __lt__(self, other):
        # For priority queue comparison - sort by f_cost
        return self.f_cost < other.f_cost
    
    def get_pos(self) -> Tuple[int, int]:
        return self.row, self.col
    
    def make_wall(self):
        self.is_wall = True
    
    def reset(self):
        self.is_wall = False
        self.parent = None
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0
    
    def update_neighbors(self, grid: List[List['Node']]):
        self.neighbors = []
        
        # Check all four adjacent cells
        directions = [
            (0, 1),   # right
            (1, 0),   # down
            (0, -1),  # left
            (-1, 0),  # up
            # Uncomment to allow diagonal movement
            # (1, 1),   # down-right
            # (-1, 1),  # up-right
            # (1, -1),  # down-left
            # (-1, -1)  # up-left
        ]
        
        for dr, dc in directions:
            new_row, new_col = self.row + dr, self.col + dc
            
            # Check if the new position is valid
            if (0 <= new_row < ROWS and 
                0 <= new_col < COLS and 
                not grid[new_row][new_col].is_wall):
                self.neighbors.append(grid[new_row][new_col])

def create_grid() -> List[List[Node]]:
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(COLS):
            node = Node(i, j)
            grid[i].append(node)
    return grid

def draw_grid_lines():
    # Draw horizontal lines
    for i in range(ROWS + 1):
        pygame.draw.line(screen, GRAY, (0, i * GRID_SIZE), (WIDTH, i * GRID_SIZE))
    
    # Draw vertical lines
    for j in range(COLS + 1):
        pygame.draw.line(screen, GRAY, (j * GRID_SIZE, 0), (j * GRID_SIZE, HEIGHT))

def draw_node(node: Node, color: Tuple[int, int, int]):
    x = node.col * GRID_SIZE
    y = node.row * GRID_SIZE
    pygame.draw.rect(screen, color, (x, y, GRID_SIZE, GRID_SIZE))

def draw(grid: List[List[Node]], closed_set: Set[Node], open_set: List[Node], 
         start: Node, end: Node, path: List[Node]):
    screen.fill(WHITE)
    
    # Draw all nodes
    for row in grid:
        for node in row:
            if node.is_wall:
                draw_node(node, BLACK)
            else:
                draw_node(node, WHITE)
    
    # Draw closed set (explored nodes)
    for node in closed_set:
        if node != start and node != end:
            draw_node(node, RED)
    
    # Draw open set (frontier nodes)
    for node in open_set:
        if node != start and node != end:
            draw_node(node, GREEN)
    
    # Draw path
    for node in path:
        if node != start and node != end:
            draw_node(node, BLUE)
    
    # Draw start and end nodes if they exist
    if start:
        draw_node(start, ORANGE)
    if end:
        draw_node(end, PURPLE)
    
    # Draw grid lines
    draw_grid_lines()
    
    pygame.display.update()

def heuristic(node1: Node, node2: Node) -> float:
    """Manhattan distance heuristic"""
    x1, y1 = node1.get_pos()
    x2, y2 = node2.get_pos()
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(current: Node) -> List[Node]:
    """Reconstructs path from end node to start node using parent pointers"""
    path = []
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)  # Add the start node
    return path[::-1]  # Reverse to get path from start to end

def best_first_search(grid: List[List[Node]], start: Node, end: Node) -> Tuple[bool, List[Node]]:
    """
    Implements Best-First Search algorithm
    Returns a tuple (found_path, path_nodes)
    """
    # Initialize open set (priority queue) and closed set
    open_set = []
    open_set_hash = {start}  # For O(1) lookup
    closed_set = set()
    
    # Calculate heuristic for start node
    start.h_cost = heuristic(start, end)
    start.f_cost = start.h_cost  # In pure Best-First, f = h
    
    # Add start node to open set
    heapq.heappush(open_set, (start.f_cost, start))
    
    # Update neighbors for all nodes
    for row in grid:
        for node in row:
            node.update_neighbors(grid)
    
    path = []
    
    while open_set:
        # Get node with lowest f_cost from open set
        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)
        
        # Add to closed set
        closed_set.add(current)
        
        # Found the end
        if current == end:
            path = reconstruct_path(end)
            return True, path
        
        # Explore neighbors
        for neighbor in current.neighbors:
            if neighbor in closed_set:
                continue
            
            # Calculate heuristic for neighbor
            neighbor.h_cost = heuristic(neighbor, end)
            neighbor.f_cost = neighbor.h_cost  # In pure Best-First, f = h
            
            if neighbor not in open_set_hash:
                # Set parent for path reconstruction
                neighbor.parent = current
                
                # Add to open set
                heapq.heappush(open_set, (neighbor.f_cost, neighbor))
                open_set_hash.add(neighbor)
        
        # Draw current state
        draw(grid, closed_set, [n for n in open_set_hash], start, end, path)
        time.sleep(0.02)  # Animation delay
        
    # No path found
    return False, []

def main():
    grid = create_grid()
    start = None
    end = None
    
    # Main game loop
    running = True
    searching = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle mouse events (only when not searching)
            if not searching:
                if pygame.mouse.get_pressed()[0]:  # Left click
                    pos = pygame.mouse.get_pos()
                    col, row = pos[0] // GRID_SIZE, pos[1] // GRID_SIZE
                    node = grid[row][col]
                    
                    # Set start node if not set
                    if not start and node != end:
                        start = node
                    
                    # Set end node if not set and not start
                    elif not end and node != start:
                        end = node
                    
                    # Set wall if not start or end
                    elif node != start and node != end:
                        node.make_wall()
                
                elif pygame.mouse.get_pressed()[2]:  # Right click
                    # Erase nodes/walls
                    pos = pygame.mouse.get_pos()
                    col, row = pos[0] // GRID_SIZE, pos[1] // GRID_SIZE
                    node = grid[row][col]
                    node.reset()
                    
                    if node == start:
                        start = None
                    elif node == end:
                        end = None
            
            # Handle keyboard events
            if event.type == pygame.KEYDOWN:
                # Start search with Space key (if start and end are set)
                if event.key == pygame.K_SPACE and start and end and not searching:
                    searching = True
                    
                    # Run Best-First Search algorithm
                    found, path = best_first_search(grid, start, end)
                    
                    if found:
                        print(f"Path found! Length: {len(path)}")
                    else:
                        print("No path found!")
                    
                    searching = False
                
                # Reset grid with 'r' key
                if event.key == pygame.K_r:
                    grid = create_grid()
                    start = None
                    end = None
                    searching = False
                
                # Generate random walls with 'w' key
                if event.key == pygame.K_w and not searching:
                    # Clear existing walls
                    for row in grid:
                        for node in row:
                            if node != start and node != end and node.is_wall:
                                node.reset()
                    
                    # Generate new walls randomly (about 30% of the grid)
                    for _ in range(int(ROWS * COLS * 0.3)):
                        row = random.randint(0, ROWS - 1)
                        col = random.randint(0, COLS - 1)
                        node = grid[row][col]
                        if node != start and node != end and not node.is_wall:
                            node.make_wall()
        
        # Draw grid
        if not searching:
            open_set = []
            closed_set = set()
            path = []
            draw(grid, closed_set, open_set, start, end, path)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()