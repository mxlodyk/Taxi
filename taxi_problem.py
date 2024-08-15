import gymnasium as gym

# ==================
# Taxi Problem Class
# ==================
class TaxiProblem:

    # Environment Seed
    current_seed = 100

    # ===========
    # Initialiser
    # ===========
    def __init__(self, seed=None):
        # Set seed if seed is passed through the parameter
        if seed is not None:
            self.current_seed = seed
        # Create Taxi-v3 environment
        self.env = gym.make("Taxi-v3", render_mode="human")
        # Initialise start state
        self.start_state, _ = self.env.reset(seed=self.current_seed)

    # ==================
    # Heuristic Function
    # ==================
    def heuristic(self, state):
        # Decode state
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)

        # Passenger is at a location
        if passenger_location < 4:
            # Convert passenger location to coordinates
            passenger_coords = self.env.unwrapped.locs[passenger_location]
            distance_to_passenger = abs(taxi_row - passenger_coords[0]) + abs(taxi_col - passenger_coords[1])

            # Return strong positive heuristic if taxi is at passenger location to prioritise pick up action
            if distance_to_passenger == 0:
                return 100
            # Return heuristic for picking up passenger 
            # The heuristic is the Manhattan distance between the taxi and the passenger
            return -(distance_to_passenger * 10)

        # Passenger is in the taxi
        elif passenger_location == 4:
            # Convert destination location to coordinates
            destination_coords = self.env.unwrapped.locs[destination]
            distance_to_destination = abs(taxi_row - destination_coords[0]) + abs(taxi_col - destination_coords[1])
            # Return heuristic for dropping off customer
            # The heuristic is the Manhattan distance between the taxi and the passenger
            # passenger subtracted from the reward for dropping the customer off (20)
            return 20 - distance_to_destination

        else:
            raise ValueError(f"Unexpected passenger_location value: {passenger_location}")

    # ==========
    # Goal State
    # ==========
    def check_goal_state(self, state):
        # Decode state
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        # Return true if passenger is at destination
        return passenger_location == destination

    # ==========
    # Successors
    # ==========
    def successors(self, state):
        # Decode state
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        # Create list to store successors
        successors = []
        # For each available action
        for action in range(self.env.action_space.n):
            # Reset the environment state to the given state
            self.env.unwrapped.s = state
            # Take action
            new_state, reward, terminated, truncated, _ = self.env.step(action)

            # If drop off action is taken and the passenger is at the destination
            # Add successor to list and skip rest of loop since drop off must be done
            if action == 5:
                if passenger_location == 4 and (taxi_row, taxi_col) == self.env.unwrapped.locs[destination]:
                    successors.append((new_state, action, reward))
                    continue
                    
            # Avoid adding terminated states back into exploration
            if not terminated and not truncated:
                # Add succesoot to list
                successors.append((new_state, action, reward))
        return successors

    # =========================
    # Reconstruct Solution Path
    # =========================
    def reconstruct_path(self, came_from, state):
        path = []
        while state in came_from:
            # Update state and retrieve action
            state, action = came_from[state]
            # Append action that was taken to reach state
            path.append(action)
        # Reverse to get the path in correct order
        path.reverse()
        return path

    # ===============
    # Render Solution
    # ===============
    def render_solution(self, solution):
        # Reset environment with the same seed as the environment used for searching
        self.env.reset(seed=self.current_seed)
        # For each action in the solution path
        for action in solution:
            # Take the action
            self.env.step(action)
            # Render each step
            self.env.render()
        # Close environment
        self.env.close() 


