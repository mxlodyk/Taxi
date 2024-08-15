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

# =============================
# Iterative Deepening A* Search
# =============================
def iterative_deepening_a_star_search(problem):
    # Recursive DFS function
    def dfs(state, g, threshold, path, visited):
        # Evaluation function
        f = g + problem.heuristic(state)

        # If 'f' exceeds threshold and no solution has been found, return 'f' value to use as 
        # threshold for next iteration
        if f > threshold:
            return None, f
        
        # Check if state is goal state
        if problem.check_goal_state(state):
            return path, f
        # Track smallest 'f' value that exceeds current threshold
        min_threshold = float('inf')
        # Mark state as visited
        visited.add(state)

        # Iterate over successor states
        for next_state, action, reward in problem.successors(state):
            # Cycle detection
            if next_state in visited:
                # Skip state
                continue
            
            # Calculate new 'g' value
            new_g = g + reward
            # Update path
            new_path = path + [action]
            # Recursively call DFS for each successor
            result, temp_threshold = dfs(next_state, new_g, threshold, new_path, visited)
            # Check if solution is found within recursive call
            if result is not None:
                return result, temp_threshold
            # Update 'min_threshold' if 'f' value from recursive call is smaller
            if temp_threshold < min_threshold:
                min_threshold = temp_threshold
        # Unmark state as visited so algorithm can explore other paths while backtracking
        visited.remove(state) 
        return None, min_threshold

    # Initialise threshold as heuristic of start state
    threshold = problem.heuristic(problem.start_state)

    # Infinite loop
    while True:
        # Initialise set to store visited states
        visited = set()
        # Initialise list to build solution path
        path = []
        # Call DFS to explore state space within current threshold
        result, temp_threshold = dfs(problem.start_state, 0, threshold, path, visited)

        # Check if solution is found
        if result is not None:
            return result
        # Check if updated threshold is infinity
        if temp_threshold == float('inf'):
            # No solution found
            return None
        
        # Increase the threshold for the next iteration
        threshold = temp_threshold


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