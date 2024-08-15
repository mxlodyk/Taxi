import heapq
import taxi_problem as taxi

# =========
# A* Search
# =========
def a_star_search(problem):
    # Initialise start state
    start_state = problem.start_state
    # Initialise list with tuples containing reward and state
    frontier = [(0, start_state)]
    # Convert list to heap
    heapq.heapify(frontier)

    # Initialise dictionary to store state:action key pair values
    # This is used for reconstructing the solution path when a solution is found
    came_from = {}
    # Initialise dictionary to store state:cumulative_reward key pair values
    # This stores g(n)
    rewards_so_far = {start_state: 0}
    # Initialise set to store states that have been explored
    explored = set()

    while frontier:
        # Pop state with the highest priority
        # The priority is g(n) + h(n)
        priority, state = heapq.heappop(frontier)
        # Check if the state is the goal state
        if problem.check_goal_state(state):
            return problem.reconstruct_path(came_from, state)
        # Mark state as explored
        explored.add(state)
        # For each successor of this state
        for next_state, action, reward in problem.successors(state):
            # Calculate g(n) of successor
            new_rewards = rewards_so_far[state] + reward
            # If the successor has not been explored and the successor g(n) has not been calculated 
            # or the successor g(n) is greater than the current g(n) stored for that successor
            if next_state not in explored and (next_state not in rewards_so_far or new_rewards < rewards_so_far[next_state]):
                rewards_so_far[next_state] = new_rewards
                # Calculate g(n) + h(n) of successor
                # This is the evaluation function of the algorithm
                # Priority is negated to simulate max_heap
                priority = -(new_rewards + problem.heuristic(next_state))
                # Push this successor state and reward to heap
                heapq.heappush(frontier, (priority, next_state))
                # Store the state and action that led to this successor state
                came_from[next_state] = (state, action)

    # No solution found
    return None

# ============================
# Iterative Deepening A* Search
# ============================
def iterative_deepening_a_star_search(problem):
    def dfs(state, g, threshold, path, visited):
        f = -(g + problem.heuristic(state))
        if f > threshold:
            return None, f
        if problem.check_goal_state(state):
            return path, f

        min_threshold = float('inf')
        visited.add(state)

        for next_state, action, reward in problem.successors(state):
            if next_state in visited:
                continue  # Avoid cycles

            new_g = g + reward  # Use actual reward for each action
            new_path = path + [action]
            result, temp_threshold = dfs(next_state, new_g, threshold, new_path, visited)

            if result is not None:
                return result, temp_threshold

            if temp_threshold < min_threshold:
                min_threshold = temp_threshold

        visited.remove(state)  # Backtrack to explore other paths
        return None, min_threshold

    # Initial threshold is the heuristic estimate from the start state
    threshold = problem.heuristic(problem.start_state)

    while True:
        visited = set()
        path = []
        result, temp_threshold = dfs(problem.start_state, 0, threshold, path, visited)

        if result is not None:
            return result  # Return the sequence of actions

        if temp_threshold == float('inf'):
            return None  # No solution found

        threshold = temp_threshold  # Increase the threshold for the next iteration



# ====================
# Dijkstra's Algorithm
# ====================
def dijkstras_algorithm(problem):
    # Initialise start state
    start_state = problem.start_state
    # Initialise list with tuples containing reward and state
    frontier = [(0, start_state)]
    # Convert list to heap
    heapq.heapify(frontier)
    # Initialise dictionary to store state:action key pair values
    # This is used for reconstructing the solution path when a solution is found
    came_from = {}
    # Initialise dictionary to store state:cumulative_reward key pair values
    # This stores g(n)
    rewards_so_far = {start_state: 0}
    # Initialise set to store states that have been explored
    explored = set()

    while frontier:
        # Pop state with highest reward
        reward, state = heapq.heappop(frontier)
        # Store negated values to simulate max_heap
        reward = -reward
        # Check if the state is the goal state
        if problem.check_goal_state(state):
            return problem.reconstruct_path(came_from, state)
        # Mark state as explored
        explored.add(state)
        # For each successor of this state
        for next_state, action, reward in problem.successors(state):
            # Calculate g(n) of successor
            # This is the evaluation function of the algorithm
            new_reward = rewards_so_far[state] + reward
            # If the successor has not been explored and the successor g(n) has not been calculated 
            # or the successor g(n) is less than (since it has been negated) the current g(n) stored for that successor
            if next_state not in explored and (next_state not in rewards_so_far or new_reward < rewards_so_far[next_state]):
                # Store g(n) of this successor
                rewards_so_far[next_state] = -new_reward
                # Push this successor state and reward to heap
                # Reward is negated to simulate max_heap
                heapq.heappush(frontier, (-new_reward, next_state))
                # Store the state and action that led to this successor state
                came_from[next_state] = (state, action)

    # No solution found
    return None

# ===============================================
# Function to Run Algorithm in Separate Process
# ===============================================
def run_algorithm(algorithm, event, solution_queue):
    # Instantiate problem
    problem = taxi.TaxiProblem()
    # Run A* search function
    if algorithm == "a_star_search":
        solution = a_star_search(problem)
        if solution:
            print("A* Solution actions:", solution)
            # Store A* solution in multiprocessing queue
            solution_queue.put(("A*", solution))
        else:
            print("A* No solution found.")
            solution_queue.put(("A*", None))
    # Run Iterative Deepening A* search function
    elif algorithm == "iterative_deepening_a_star_search":
        solution = dijkstras_algorithm(problem)
        if solution:
            # Store IDA*'s solution in multiprocessing queue
            print("Iterative Deepening A* Solution actions:", solution)
            solution_queue.put(("Iterative Deepening A*", solution))
        else:
            print("Dijkstra No solution found.")
            solution_queue.put(("Iterative Deepening A*", None))
    # Set each algorithm event as complete
    event.set() 