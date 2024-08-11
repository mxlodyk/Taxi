import heapq

import gymnasium as gym
import queue

class TaxiProblem:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode="human")
        self.start_state, _ = self.env.reset(seed=102)

    def heuristic(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        # Passenger is at a location (0 to 3)
        if passenger_location < 4:
            # Convert passenger location to coordinates
            passenger_coords = self.env.unwrapped.locs[passenger_location]
            # Return Manhattan distance between the taxi and the passenger
            distance_to_passenger = abs(taxi_row - passenger_coords[0]) + abs(taxi_col - passenger_coords[1])

            # Provide strong negative heuristic to prioritise picking up customer if reached
            if distance_to_passenger == 0:
                return -100
            else:
                return distance_to_passenger

        # Passenger is in the taxi
        elif passenger_location == 4:
            # Convert destination location to coordinates
            destination_coords = self.env.unwrapped.locs[destination]
            # Return Manhattan distance between the taxi and the destination
            return abs(taxi_row - destination_coords[0]) + abs(taxi_col - destination_coords[1])
        else:
            raise ValueError(f"Unexpected passenger_location value: {passenger_location}")

    # Check if state is goal state
    def is_goal_state(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        return passenger_location == destination

    # Generate successors of given state
    def get_successors(self, state):
        successors = []
        # For each available action
        for action in range(self.env.action_space.n):
            # Set the environment state to the current state
            self.env.unwrapped.s = state

            new_state, reward, terminated, truncated, _ = self.env.step(action)
            if not terminated and not truncated:  # Avoid adding terminal states back into the exploration
                successors.append((new_state, action, reward))
        return successors

    def a_star_search(self):
        # Initialise start state
        start_state = self.start_state
        # Initialise priority queue with tuples containing priority and state
        frontier = [(-0, start_state)]
        # Convert queue to heap for retrieving highest reward actions
        heapq.heapify(frontier)

        # Initialise dictionary to store each state and action that led to current state
        path_taken = {}
        # Initialise dictionary to track the cost from the start state to current state
        rewards_so_far = {start_state: 0}
        explored = set()

        while frontier:
            priority, current_state = heapq.heappop(frontier)

            if self.is_goal_state(current_state):
                return self.reconstruct_path(path_taken, current_state)

            explored.add(current_state)

            for next_state, action, reward in self.get_successors(current_state):
                new_rewards = rewards_so_far[current_state] + reward

                if next_state not in explored and (next_state not in rewards_so_far or new_rewards < rewards_so_far[next_state]):
                    rewards_so_far[next_state] = new_rewards
                    priority = new_rewards + self.heuristic(next_state)
                    heapq.heappush(frontier, (priority, next_state))
                    path_taken[next_state] = (current_state, action)

        return None  # No solution found

    def reconstruct_path(self, came_from, current_state):
        path = []
        while current_state in came_from:
            current_state, action = came_from[current_state]
            path.append(action)
        path.reverse()  # Reverse to get the path from start to goal
        return path

    def render_solution(self, solution):
        self.env.reset()
        for action in solution:
            self.env.step(action)
            self.env.render()  # Render each step
        self.env.close()  # Close the environment when done

# Create problem instance and run A* search
problem = TaxiProblem()
solution = problem.a_star_search()
if solution:
    print("Solution actions:", solution)
    problem.render_solution(solution)  # Render the solution actions
else:
    print("No solution found.")
