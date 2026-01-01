import copy
import re
import heapq
import tkinter as tk
from tkinter import ttk

# Requirement 4: Tiles are labeled R, G, B and move in that order
TILE_LABELS = ['R', 'G', 'B']  # Tile order: R first, then G, then B


class Node:
    """
    Represents a node in the A* search tree.
    Each node contains a state, path cost, heuristic value, and parent reference.
    """

    def __init__(self, parent_node, state, path_cost, heuristic, current_played_tile):
        self.parent_node = parent_node
        self.state = state
        self.path_cost = path_cost
        # Requirement 3: Total cost = path cost + Hamming distance heuristic
        self.total_cost = path_cost + heuristic
        self.current_played_tile = current_played_tile

    def __lt__(self, other):
        """Comparison for priority queue - lower total cost has higher priority"""
        return self.total_cost < other.total_cost

    # Requirement 4: Tiles move in R, G, B order
    def next_tile_to_move(self, turn):
        """
        Determines which tile moves next based on turn number.
        Turn 1, 4, 7, 10... -> R        % 1
        Turn 2, 5, 8... -> G            % 2
        Turn 3, 6, 9... -> B            % 0
        """
        turn_mod = turn % 3
        if turn_mod == 1:
            return 'R'
        elif turn_mod == 2:
            return 'G'
        else:  # turn_mod == 0
            return 'B'

    def print_node(self, turn):
        """Prints the node information including state, costs, and current tile to move"""
        print(f"\n======= Turn {turn} (Tile: {self.current_played_tile}) =======")
        self.state.print_board()
        print(f"Next Tile to Move: {self.next_tile_to_move(turn + 1)}")         # Turn+1 is used to indicate the next move for a piece.
        print(f"Path Cost (g): {self.path_cost}")
        print(f"Hamming Distance (h): {int(self.total_cost) - int(self.path_cost)}")        # Requirement 3: Show Hamming distance heuristic
        print(f"Total Cost (f = g + h): {self.total_cost}")
        print("=" * 25)


class Game:
    """
    Main game class that handles initialization and A* search algorithm.
    """

    def __init__(self):
        init_coords, goal_coords = self.get_init_and_goal_state()                   # Requirement 1: Initial and goal states given by user
        self.state, self.goal_state = self.init_states(init_coords, goal_coords)
        print("\n" + "=" * 40)
        print("INITIAL STATE:")
        self.state.print_board()
        print("\nGOAL STATE:")
        self.goal_state.print_board()
        print("=" * 40)

    def init_states(self, init_coordinates, goal_coordinates):
        """
        Creates initial and goal State objects from user-provided coordinates.
        """
        tiles = list()
        for i in range(3):                  # Requirement 4: Use R, G, B labels instead of numbers
            tiles.append(Tile(TILE_LABELS[i],
                              int(init_coordinates[i][0]) - 1,
                              int(init_coordinates[i][1]) - 1,
                              int(goal_coordinates[i][0]) - 1,
                              int(goal_coordinates[i][1]) - 1))

        initial_state = [[None, None, None], [None, None, None], [None, None, None]]
        goal_state = [[None, None, None], [None, None, None], [None, None, None]]

        for tile in tiles:
            initial_state[int(tile.x_position)][int(tile.y_position)] = tile
            goal_state[int(tile.goal_x_position)][int(tile.goal_y_position)] = tile

        return State(initial_state), State(goal_state)


    def get_init_and_goal_state(self):                      # Requirement 1: Initial and goal states given by user
        """
        Gets initial and goal state coordinates from user input.
        Format: (row, column)
        """
        print("""
The form of the 3x3 board is:
         Col 1  Col 2  Col 3
Row 1      _      _      _
Row 2      _      _      _
Row 3      _      _      _

Tiles: R (Red), G (Green), B (Blue)
""")
        # Requirement 1: Get initial state from user
        print("----------- Initial State -----------")
        print("Enter coordinates for each tile as (row,col) e.g., (1,2)")
        initial_coordinates = list()
        for tile_label in TILE_LABELS:
            while True:
                print(f"Tile {tile_label} ->  ", end="")
                temp_coordinate = str(input())
                if re.match(r"^\(([123]),([123])\)$", temp_coordinate):
                    coordinate = temp_coordinate.strip("()").split(",")
                    if coordinate in initial_coordinates:
                        print("Error: Position already occupied. Enter a different position.")
                        continue
                    initial_coordinates.append(coordinate)
                    break
                else:
                    print("Invalid format. Please use (row,col) format, e.g., (1,2)")

        # Requirement 1: Get goal state from user
        print("\n----------- Goal State -----------")
        print("Enter goal coordinates for each tile as (row,col) e.g., (1,2)")
        goal_coordinates = list()
        for tile_label in TILE_LABELS:
            while True:
                print(f"Tile {tile_label} ->  ", end="")
                temp_coordinate = str(input())
                if re.match(r"^\(([123]),([123])\)$", temp_coordinate):
                    coordinate = temp_coordinate.strip("()").split(",")
                    if coordinate in goal_coordinates:
                        print("Error: Position already occupied. Enter a different position.")
                        continue
                    goal_coordinates.append(coordinate)
                    break
                else:
                    print("Invalid format. Please use (row,col) format, e.g., (1,2)")

        return initial_coordinates, goal_coordinates

    def is_goal(self, state):
        """Checks if a given state matches the goal state"""
        return state == self.goal_state

    def set_new_state(self, label, move, state):
        """
        Creates a new state by moving the specified tile to the new position.
        """
        new_state = copy.deepcopy(state)
        prev_tile = None
        for i in range(3):
            for j in range(3):
                if new_state.state[i][j] is not None:
                    if new_state.state[i][j].label == label:
                        prev_tile = copy.deepcopy(new_state.state[i][j])
                        new_state.state[i][j] = None
                        break  # We found the stone and deleted its previous position, exit the loop.
            if prev_tile:
                break

        # Goal positions are preserved when creating a new tile object.
        new_state.state[move[0]][move[1]] = Tile(label, move[0], move[1],
                                                 prev_tile.goal_x_position,
                                                 prev_tile.goal_y_position)
        return new_state

    # Requirement 3: A* search with Hamming distance heuristic
    # Requirement 5: Expansion till 10th turn
    # Requirement 6: Show chosen state, cost, and alternative states
    def solve_a_star(self, ui_mode=True):
        """
        Implements A* search algorithm with:
        - Hamming distance as heuristic
        - Turn limit of 10
        - Prints chosen state and alternative states with costs
        """
        pq = []  # Priority queue for A* search
        visited_states = {} # We keep the places we've visited so we don't have to go back again.

        max_turns = 10          # Requirement 5: Maximum 10 turns

        # Requirement 3: Use Hamming distance as heuristic / Initialize with start state
        initial_heuristic = self.state.state_hamming_distance()
        # Since the first step is not the result of the previous move, path_cost is 0, current_played_tile is None.
        initial_node = Node(None, self.state, 0, initial_heuristic, None)
        heapq.heappush(pq, initial_node)

        turn = 0  # First Node Turn:0
        goal_found = False
        final_node = None

        # Add initial state to visited_states
        visited_states[self.state] = 0

        if not ui_mode:
            print("\n" + "=" * 50)
            print("STARTING A* SEARCH")
            print("=" * 50)

        while pq:
            expanded_node = heapq.heappop(pq)
            current_turn = expanded_node.path_cost

            # Goal Control
            if self.is_goal(expanded_node.state):
                goal_found = True
                final_node = expanded_node
                if not ui_mode:
                    print("\n" + "*" * 50)
                    print(f"GOAL REACHED at Path Cost {expanded_node.path_cost}!")
                    print(f"Total Cost: {expanded_node.total_cost}")
                    print("*" * 50)
                break

            # 10 Turn Control
            if current_turn >= max_turns:
                if not ui_mode:
                    print(f"\nMax turns limit ({max_turns}) reached for current path. Pruning.")
                continue

            # Visited situation check (Skip unless arrived via a less costly route)
            if expanded_node.path_cost > visited_states.get(expanded_node.state, float('inf')):
                continue

            # Create New Steps

            # Turn Number for New Steps(0   1   2   3...)
            next_turn = current_turn + 1

            # Requirement 4: Determine which tile moves (R, G, B order)
            current_tile = expanded_node.next_tile_to_move(next_turn)

            # Requirement 2: Get all possible moves (including diagonal)
            possible_moves = expanded_node.state.get_possible_moves(current_tile)

            if not possible_moves:
                if not ui_mode:
                    print(f"\n--- Turn {next_turn}: Tile {current_tile} is blocked. Pruning current path. ---")
                continue

            # Requirement 6: Generate and display all alternative states with costs
            if not ui_mode:
                expanded_node.print_node(current_turn)  # Show last situation
                print(f"\n--- Turn {next_turn}: Possible Moves for Tile {current_tile} ---")

            alternative_states = []

            for move in possible_moves:
                new_state = self.set_new_state(current_tile, move, expanded_node.state)
                # Each move's cost is 1 so path_cost+1
                path_cost = expanded_node.path_cost + move[2]

                # Requirement 3: Use Hamming distance as heuristic
                heuristic = new_state.state_hamming_distance()
                total_cost = path_cost + heuristic

                move_direction = self.get_move_direction(move)
                alternative_states.append({
                    'state': new_state,
                    'path_cost': path_cost,
                    'heuristic': heuristic,
                    'total_cost': total_cost,
                    'move': move,
                    'direction': move_direction
                })

            # Requirement 6: Sort by total cost and show all alternatives
            alternative_states.sort(key=lambda x: x['total_cost'])

            if not ui_mode:
                for idx, alt in enumerate(alternative_states):
                    chosen_mark = " ** CHOSEN **" if idx == 0 else ""
                    print(f"\nAlternative {idx + 1}: Move {alt['direction']}{chosen_mark}")
                    alt['state'].print_board()
                    print(f"  Move Cost: {alt['move'][2]}")
                    print(f"  Path Cost (g): {alt['path_cost']}")
                    print(f"  Hamming Distance (h): {alt['heuristic']}")
                    print(f"  Total Cost (f): {alt['total_cost']}")

            # Add all alternatives to priority queue
            for alt in alternative_states:
                # Add/update the new status to the visited list.
                if alt['path_cost'] < visited_states.get(alt['state'], float('inf')):
                    visited_states[alt['state']] = alt['path_cost']

                    # Requirement 3: Use Hamming distance heuristic
                    heapq.heappush(pq, Node(expanded_node, alt['state'], alt['path_cost'],
                                            alt['heuristic'], current_tile))

        # Requirement 5: Inform user if solution not found within 10 turns
        if not goal_found and not ui_mode:
            print("\n" + "!" * 50)
            print("SOLUTION NOT FOUND WITHIN MAX PATH COST LIMIT!")
            print("The goal state could not be reached or path cost limit exceeded.")
            print("!" * 50)

        return final_node

    # Requirement 2: Helper function to identify move direction
    def get_move_direction(self, move):
        """
        Returns a string describing the move direction.
        """
        direction_map = {
            (-1, 0): "UP",
            (1, 0): "DOWN",
            (0, -1): "LEFT",
            (0, 1): "RIGHT",
            (-1, -1): "DIAGONAL UP-LEFT",
            (-1, 1): "DIAGONAL UP-RIGHT",
            (1, -1): "DIAGONAL DOWN-LEFT",
            (1, 1): "DIAGONAL DOWN-RIGHT"
        }
        # move list: [new_x, new_y, cost, dx, dy]
        if len(move) >= 5:
            dx, dy = move[3], move[4]
            return direction_map.get((dx, dy), "UNKNOWN")
        return "MOVE"

    def show_solution_path(self, node):
        """Displays the complete path from initial state to goal state"""
        temp_node = node
        path_list = []
        path_list.append(temp_node)
        while temp_node.parent_node is not None:
            temp_node = temp_node.parent_node
            path_list.append(temp_node)
        path_list.reverse()

        print("\n" + "=" * 40)
        print("SOLUTION PATH")
        print("=" * 40)
        for idx, node in enumerate(path_list):
            if idx == 0:
                print(f"\nStep {idx}: INITIAL STATE")
            else:
                # We get which stone moved from the node object.
                print(
                    f"\nStep {idx}: Tile {node.current_played_tile} moved (g={node.path_cost}, h={node.total_cost - node.path_cost}, f={node.total_cost})")

            node.state.print_board()
            if idx < len(path_list) - 1:
                print("    ↓")
        print("\n" + "=" * 40)


class State:
    """
    Represents a state of the 3x3 board.
    Contains methods for displaying, comparing, and calculating heuristics.
    """

    def __init__(self, matrix):
        self.state = matrix

    def print_board(self):
        """Prints the current board state in a formatted way"""
        print("  1 2 3")
        for i in range(3):
            print(f"{i + 1}", end=" ")
            for j in range(3):
                if self.state[i][j] is None:
                    print("_", end=" ")
                else:
                    print(self.state[i][j].label, end=" ")
            print()

    # Requirement 3: Use Hamming distance as heuristic
    def state_hamming_distance(self):
        """
        Calculates Hamming distance: number of tiles not in their goal position.
        Requirement 3: Hamming distance is used as the heuristic for A* search
        """
        distance = 0
        for row in self.state:
            for element in row:
                if element is not None:
                    # Hamming: 1 if not in goal position, 0 if in goal position
                    if not element.is_goal():
                        distance += 1
        return distance

    # Requirement 2: Tiles can move up, down, right, left, or diagonal
    def get_possible_moves(self, label):
        """
        Gets all possible moves for a tile with the given label.
        Requirement 2: Includes up, down, left, right, AND diagonal moves
        All moves have cost 1 (uniform cost for this implementation)
        """
        moves = list()
        x, y = -1, -1
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is not None and self.state[i][j].label == label:
                    x = i
                    y = j
                    break
            if x != -1:
                break

        if x == -1:
            return moves

        # Direction offsets: (dx, dy, direction_name)
        directions = [
            (-1, 0, "UP"),
            (1, 0, "DOWN"),
            (0, -1, "LEFT"),
            (0, 1, "RIGHT"),
            (-1, -1, "DIAG_UL"),
            (-1, 1, "DIAG_UR"),
            (1, -1, "DIAG_DL"),
            (1, 1, "DIAG_DR")
        ]

        for dx, dy, direction_name in directions:
            new_x = x + dx
            new_y = y + dy
            # Check if new position is within bounds and empty
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                if self.state[new_x][new_y] is None:
                    # Cost is 1 for all moves.
                    moves.append([new_x, new_y, 1, dx, dy])
        return moves

    def __eq__(self, other):
        """Checks if two states are equal (all tiles in same positions)"""
        if not isinstance(other, State):
            return False

        positions_self = {}
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is not None:
                    positions_self[self.state[i][j].label] = (i, j)

        positions_other = {}
        for i in range(3):
            for j in range(3):
                if other.state[i][j] is not None:
                    positions_other[other.state[i][j].label] = (i, j)

        return positions_self == positions_other

    # The `__hash__` method is necessary for State objects to be used in dicts or sets.
    def __hash__(self):
        # Create a tuple to represent the state (it can be hashed)
        # Example: (('R', 1, 1), ('G', 2, 2), ('B', 3, 3))
        tile_positions = []
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is not None:
                    tile_positions.append((self.state[i][j].label, i, j))
        return hash(tuple(sorted(tile_positions)))


class Tile:
    """
    Represents a single tile on the board.
    Requirement 4: Tiles are labeled R, G, B
    """

    def __init__(self, label, x, y, goal_x, goal_y):
        self.label = label  # R, G, or B
        self.x_position = x
        self.y_position = y
        self.goal_x_position = goal_x
        self.goal_y_position = goal_y

    def __str__(self):
        return (f"Tile {self.label}: "
                f"Current ({self.x_position + 1}, {self.y_position + 1}), "
                f"Goal ({self.goal_x_position + 1}, {self.goal_y_position + 1})")

    # Requirement 3: Hamming distance helper - checks if tile is at goal
    def is_goal(self):
        """Returns True if tile is at its goal position"""
        return self.x_position == self.goal_x_position and self.y_position == self.goal_y_position

    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.label == other.label and self.x_position == other.x_position and self.y_position == other.y_position


class ResultsUI:
    """
    Tkinter UI to display the optimal path results after A* search.
    """

    def __init__(self, solution_node, goal_state):
        self.root = tk.Tk()
        self.root.title("A* Search Results - Optimal Path")
        self.root.geometry("1200x600")
        self.root.configure(bg="#f0f0f0")

        if solution_node:
            self.display_optimal_path(solution_node, goal_state)
        else:
            self.display_no_solution()


    def get_tile_color(self, tile):
        """Get color for tile display"""
        colors = {'R': '#ff9999', 'G': '#99ff99', 'B': '#9999ff'}
        return colors.get(tile, 'white')

    def display_no_solution(self):
        """Display message when no solution found"""
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        label = ttk.Label(frame,
                          text="SOLUTION NOT FOUND WITHIN MAX PATH COST LIMIT!\n\n"
                               "The goal state could not be reached in the allowed steps.",
                          font=("Arial", 14, "bold"),
                          foreground="red",
                          justify=tk.CENTER)
        label.pack(expand=True)

    def display_optimal_path(self, final_node, goal_state):
        """Display the optimal path"""
        path = []
        current = final_node
        while current is not None:
            path.append(current)
            current = current.parent_node
        path.reverse()

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title = ttk.Label(main_frame, text="OPTIMAL PATH - A* Search Solution",
                          font=("Arial", 16, "bold"))
        title.pack(side=tk.TOP, pady=10)

        # Canvas with scrollbar for horizontal scrolling
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set)

        # Add each state in the path
        for idx, node in enumerate(path):
            step_frame = ttk.LabelFrame(scrollable_frame, text=f"Step {idx}", padding=10)
            step_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

            # Board display
            board_frame = ttk.Frame(step_frame)
            board_frame.pack(padx=5, pady=5)

            self.draw_board_in_frame(board_frame, node.state)

            # Cost information
            heuristic = int(node.total_cost) - int(node.path_cost)
            info_text = f"Path Cost (g): {node.path_cost}\n"
            info_text += f"Heuristic (h): {heuristic}\n"
            info_text += f"Total Cost (f): {node.total_cost}"

            if idx > 0:
                info_text = f"Tile: {node.current_played_tile} moved\n" + info_text

            info_label = ttk.Label(step_frame, text=info_text,
                                   font=("Arial", 9), justify=tk.LEFT)
            info_label.pack(padx=5, pady=5)

            # Arrow between steps (except last)
            if idx < len(path) - 1:
                arrow_label = ttk.Label(scrollable_frame, text="→", font=("Arial", 24, "bold"))
                arrow_label.pack(side=tk.LEFT, padx=5)

        # Goal state display
        goal_frame = ttk.LabelFrame(scrollable_frame, text="GOAL", padding=10)
        goal_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        goal_board_frame = ttk.Frame(goal_frame)
        goal_board_frame.pack(padx=5, pady=5)
        self.draw_board_in_frame(goal_board_frame, goal_state)
        goal_label = ttk.Label(goal_frame, text="Target State", font=("Arial", 9, "bold"))
        goal_label.pack(pady=5)

        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Summary at bottom
        summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)

        summary_text = f"✓ Solution found in {len(path) - 1} moves    |    "
        summary_text += f"Final Path Cost: {final_node.path_cost}    |    "
        summary_text += f"Total Cost (f): {final_node.total_cost}"

        summary_label = ttk.Label(summary_frame, text=summary_text,
                                  font=("Arial", 11, "bold"), foreground="green")
        summary_label.pack()

    def draw_board_in_frame(self, parent, state):
        """Draw a board state in the given frame"""
        board_frame = ttk.Frame(parent)
        board_frame.pack()

        # Column labels
        ttk.Label(board_frame, text="", width=3).grid(row=0, column=0)
        for col in range(1, 4):
            ttk.Label(board_frame, text=str(col), font=("Arial", 8, "bold"),
                      width=3).grid(row=0, column=col)

        # Rows
        for i in range(3):
            ttk.Label(board_frame, text=str(i + 1), font=("Arial", 8, "bold"),
                      width=3).grid(row=i + 1, column=0)
            for j in range(3):
                cell = state.state[i][j]
                if cell is None:
                    text = "_"
                    bg_color = "#e0e0e0"
                else:
                    text = cell.label
                    bg_color = self.get_tile_color(cell.label)

                label = tk.Label(board_frame, text=text, width=4, height=2,
                                 bg=bg_color, font=("Arial", 12, "bold"),
                                 relief=tk.RIDGE, borderwidth=2)
                label.grid(row=i + 1, column=j + 1, padx=1, pady=1)


def main():
    """
    Main function - creates game and runs A* search.
    """
    print("=" * 50)
    print("AI PROJECT 2025-2026: 3x3 Board Game with A* Search ")
    print("=" * 50)

    try:
        game = Game()
    except EOFError:
        print("\n[Error]: User input could not be received. Run the program from the command line or a suitable environment..")
        return
    except Exception as e:
        print(f"\n[Error]: An error occurred while starting the game: {e}")
        return

    solution_node = game.solve_a_star(ui_mode=False)

    # Display results in tkinter UI
    if solution_node:
        game.show_solution_path(solution_node)

    if solution_node:
        ui = ResultsUI(solution_node, game.goal_state)
        ui.root.mainloop()
    else:
        ui = ResultsUI(None, game.goal_state)
        ui.root.mainloop()


if __name__ == "__main__":
    main()