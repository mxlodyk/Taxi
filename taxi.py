import gymnasium as gym
import heapq
import queue
import multiprocessing
import sys

# Environment Seed
current_seed = 100

# ==================
# Taxi Problem Class
# ==================
class TaxiProblem:

    # ===========
    # Initialiser
    # ===========
    def __init__(self, seed):
        self.env = gym.make("Taxi-v3", render_mode="human")
        self.start_state, _ = self.env.reset(seed=current_seed)

    # ==================
    # Heuristic Function
    # ==================
    def heuristic(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)

        # Passenger is at a location (0 to 3)
        if passenger_location < 4:
            # Convert passenger location to coordinates
            passenger_coords = self.env.unwrapped.locs[passenger_location]
            # Return heuristic for picking up passenger (Manhattan distance between the taxi and the passenger)
            distance_to_passenger = abs(taxi_row - passenger_coords[0]) + abs(taxi_col - passenger_coords[1])

            # Provide strong negative heuristic to prioritise picking up passenger if taxi is at passenger location
            if distance_to_passenger == 0:
                return -100
            return distance_to_passenger

        # Passenger is in the taxi
        elif passenger_location == 4:
            # Convert destination location to coordinates
            destination_coords = self.env.unwrapped.locs[destination]
            # Return heuristic for dropping off customer (Manhattan distance between the taxi and the destination)
            distance_to_destination = abs(taxi_row - destination_coords[0]) + abs(taxi_col - destination_coords[1])

            # Encourage moving toward the destination with an estimate of the reward
            expected_reward = 20 - distance_to_destination
            # Negative to prioritize higher rewards
            return -expected_reward

        else:
            raise ValueError(f"Unexpected passenger_location value: {passenger_location}")

    # ==========
    # Goal State
    # ==========
    def goal_state(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        return passenger_location == destination

    # ==========
    # Successors
    # ==========
    # Generate successors for a given state
    def successors(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        successors = []
        # For each available action
        for action in range(self.env.action_space.n):
            # Set the environment state to the current state
            self.env.unwrapped.s = state

            new_state, reward, terminated, truncated, _ = self.env.step(action)

            if action == 5:
                if passenger_location == 4 and (taxi_row, taxi_col) == self.env.unwrapped.locs[destination]:
                    successors.append((new_state, action, reward))
                    # Skip rest of loop since drop off must be done if passenger is at destination
                    continue
                    
            # Avoid adding terminated states back into exploration
            if not terminated and not truncated:
                successors.append((new_state, action, reward))
        return successors

    # =========================
    # Reconstruct Solution Path
    # =========================
    def reconstruct_path(self, came_from, current_state):
        path = []
        while current_state in came_from:
            # Update current_state and retrieve action
            current_state, action = came_from[current_state]
            # Append action that was taken to reach current_state
            path.append(action)
        # Reverse to get the path in correct order
        path.reverse()
        return path

    # ===============
    # Render Solution
    # ===============
    def render_solution(self, solution):
        # Reset environment with the same seed as the environment used for searching
        self.env.reset(seed=current_seed)
        # For each action in the solution path
        for action in solution:
            # Take the action
            self.env.step(action)
            # Render each step
            self.env.render()
        # Close environment
        self.env.close() 


# =========
# A* Search
# =========
def a_star_search(problem):
    # Initialise start state
    start_state = problem.start_state
    # Initialise priority queue with tuples containing priority and state
    frontier = [(-0, start_state)]
    # Convert queue to heap for retrieving highest reward actions
    heapq.heapify(frontier)

    # Initialise dictionary to store each (previous) state:action key pair that led to current state
    came_from = {}
    # Initialise dictionary to track the cumulative 'cost' from the start state to the current state
    rewards_so_far = {start_state: 0}
    explored = set()

    while frontier:
        # Pop state with the highest priority
        priority, current_state = heapq.heappop(frontier)

        if problem.goal_state(current_state):
            return problem.reconstruct_path(came_from, current_state)

        explored.add(current_state)

        for next_state, action, reward in problem.successors(current_state):
            new_rewards = rewards_so_far[current_state] + reward

            if next_state not in explored and (next_state not in rewards_so_far or new_rewards < rewards_so_far[next_state]):
                rewards_so_far[next_state] = new_rewards
                priority = new_rewards + problem.heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = (current_state, action)

    # No solution found
    return None

 # ====================
# Dijkstra's Algorithm
# ====================
def dijkstras_algorithm(problem):
    # Initialise start state
    start_state = problem.start_state
    # Initialise priority queue with tuples containing priority and state
    frontier = [(0, start_state)]
    heapq.heapify(frontier)
    came_from = {}
    rewards_so_far = {start_state: 0}
    explored = set()

    while frontier:
        # Pop state with smallest cumulative reward
        current_reward, current_state = heapq.heappop(frontier)
        if problem.goal_state(current_state):
            return problem.reconstruct_path(came_from, current_state)
        
        explored.add(current_state)

        for next_state, action, reward in problem.successors(current_state):
            new_reward = rewards_so_far[current_state] + reward

            if next_state not in explored and (next_state not in rewards_so_far or new_reward < rewards_so_far[next_state]):
                rewards_so_far[next_state] = new_reward
                heapq.heappush(frontier, (new_reward, next_state))
                came_from[next_state] = (current_state, action)

    # No solution found
    return None

# ===============================================
# Function to Run A* Search in a Separate Process
# ===============================================
def run_algorithm(algorithm, seed, event, solution_queue):
    problem = TaxiProblem(seed)
    if algorithm == "a_star_search":
        solution = a_star_search(problem)
        if solution:
            print("A* Solution actions:", solution)
            solution_queue.put(("A*", solution))
        else:
            print("A* No solution found.")
            solution_queue.put(("A*", None))
    elif algorithm == "dijkstras_algorithm":
        solution = dijkstras_algorithm(problem)
        if solution:
            print("Dijkstra Solution actions:", solution)
            solution_queue.put(("Dijkstra", solution))
        else:
            print("A* No solution found.")
            solution_queue.put(("Dijkstra", None))
    event.set() 

# ==========================================================
# Function to Run Dijkstra's Algorithm in a Separate Process
# ==========================================================
# def run_dijkstra(seed, event, solution_queue):
#     problem = TaxiProblem(seed)
#     solution = dijkstras_algorithm(problem)
#     if solution:
#         print("Dijkstra's Solution actions:", solution)
#         solution_queue.put(("Dijkstra", solution))
#     else:
#         print("Dijkstra's No solution found.")
#     solution_queue.put(("Dijkstra", None))
#     # Event: Dijkstra's algorithm complete
#     event.set()

# ====================
# Main Processing Loop
# ====================
def main():
    global current_seed
    selection = -1

    while selection != 5:
        selection = input("Please select a search algorithm:\n"
                        "1. A* Search\n"
                        "2. Dijkstra's Algorithm\n"
                        "3. Compare\n"
                        "4. Change Seed\n"
                        "5. Quit\n")
        
        # Selection 1: A*
        if selection == "1":
            problem = TaxiProblem(current_seed)
            solution = a_star_search(problem)
            if solution:
                problem.render_solution(solution)

        # Selection 2: Dijkstra's
        elif selection == "2":
            problem = TaxiProblem(current_seed)
            solution = dijkstras_algorithm(problem)
            if solution:
                problem.render_solution(solution)

        # Selection 3: Compare
        elif selection == "3":

            event1 = multiprocessing.Event()
            event2 = multiprocessing.Event()

            solution_queue = multiprocessing.Queue()

            process1 = multiprocessing.Process(target=run_algorithm, args=("a_star_search", current_seed, event1, solution_queue))
            process2 = multiprocessing.Process(target=run_algorithm, args=("dijkstras_algorithm", current_seed, event2, solution_queue))

            process1.start()
            process2.start()

            event1.wait()
            event2.wait()

            process1.join()
            process2.join()

            while not solution_queue.empty():
                algorithm_name, solution = solution_queue.get()
                if solution:
                    print(f"{algorithm_name} rendering...")
                    problem = TaxiProblem(current_seed if algorithm_name == "A*" else current_seed + 1)
                    problem.render_solution(solution)
                else:
                    pass

        # Selection 4: Change Seed
        elif selection == "4":
            current_seed = int(input("Enter seed:\n"))
            main()

        # Selection 5: Quit
        elif selection == "5":
            sys.exit(0)

if __name__ == "__main__":
    main()