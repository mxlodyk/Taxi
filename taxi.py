import gymnasium as gym
import heapq
import queue
import multiprocessing

# Environment Seed
SEED = 100

# ==================
# Taxi Problem Class
# ==================
class TaxiProblem:

    # ===========
    # Initialiser
    # ===========
    def __init__(self, seed):
        self.env = gym.make("Taxi-v3", render_mode="human")
        self.start_state, _ = self.env.reset(seed=SEED)

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

            if not terminated and not truncated:  # Avoid adding terminal states back into the exploration
                successors.append((new_state, action, reward))
        return successors

    # =========
    # A* Search
    # =========
    def a_star_search(self):
        # Initialise start state
        start_state = self.start_state
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

            if self.goal_state(current_state):
                return self.reconstruct_path(came_from, current_state)

            explored.add(current_state)

            for next_state, action, reward in self.successors(current_state):
                new_rewards = rewards_so_far[current_state] + reward

                if next_state not in explored and (next_state not in rewards_so_far or new_rewards < rewards_so_far[next_state]):
                    rewards_so_far[next_state] = new_rewards
                    priority = new_rewards + self.heuristic(next_state)
                    heapq.heappush(frontier, (priority, next_state))
                    came_from[next_state] = (current_state, action)

        # No solution found
        return None
    
    # ====================
    # Dijkstra's Algorithm
    # ====================
    def dijkstras_algorithm(self):
        # Initialise start state
        start_state = self.start_state
        # Initialise priority queue with tuples containing priority and state
        frontier = [(0, start_state)]
        heapq.heapify(frontier)
        came_from = {}
        rewards_so_far = {start_state: 0}
        explored = set()

        while frontier:
            # Pop state with smallest cumulative reward
            current_reward, current_state = heapq.heappop(frontier)
            if self.goal_state(current_state):
                return self.reconstruct_path(came_from, current_state)
            
            explored.add(current_state)

            for next_state, action, reward in self.successors(current_state):
                new_reward = rewards_so_far[current_state] + reward

                if next_state not in explored and (next_state not in rewards_so_far or new_reward < rewards_so_far[next_state]):
                    rewards_so_far[next_state] = new_reward
                    heapq.heappush(frontier, (new_reward, next_state))
                    came_from[next_state] = (current_state, action)

        # No solution found
        return None

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
        self.env.reset(seed=SEED)
        # For each action in the solution path
        for action in solution:
            # Take the action
            self.env.step(action)
            # Render each step
            self.env.render()
            # Close environment
        self.env.close() 

# ====================
# Main Processing Loop
# ====================
# Get user input to determine which search algorithm to use
selection = input("Please select a search algorithm:\n"
                  "1. A* Search\n"
                  "2. Dijkstra's Algorithm\n"
                  "3. Compare\n")
# Create problem instance and run A* search
problem = TaxiProblem(seed=SEED)
# Menu logic
if selection == "1":
    solution = problem.a_star_search()
    # If search found a solution
    if solution:
        # Print solution path
        print("Solution actions:", solution)
        # Render solution path
        problem.render_solution(solution)
    else:
        print("No solution found.")
elif selection == "2":
    solution = problem.dijkstras_algorithm()
    # If search found a solution
    if solution:
        # Print solution path
        print("Solution actions:", solution)
        # Render solution path
        problem.render_solution(solution)
    else:
        print("No solution found.")