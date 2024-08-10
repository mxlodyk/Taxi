import gymnasium as gym
import queue

class TaxiProblem:

    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode="human")
        self.destination = -1
        self.passenger = -1
        self.taxi = [-1, -1]
        if self.goal_test:
            self.env.close()

    def start_state(self):
        state, info = self.env.reset()
        print(state) # Print start state
        return state

    def goal_state(self, state):
        # return observation
        pass

    def goal_test(self, state):
        # return self.env == self.goal_state()
        return False
        pass

    def decode_state(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        print(taxi_row, taxi_col, passenger_location, destination) # Print decoded state
        return taxi_row, taxi_col, passenger_location, destination

    def take_action(self, state):
        result = []
        action = self.env.action_space.sample()
        taxi_row, taxi_col, passenger_location, destination = self.decode_state(state)

        # Map passenger location to coordinates
        passenger_coords = [(0, 0), (0, 4), (4, 0), (4, 3)]

        if passenger_location < 4:
            passenger_row, passenger_col = passenger_coords[passenger_location]

            if passenger_location != 4 and (taxi_row, taxi_col) == (passenger_row, passenger_col):
                action = 4  # Pickup action
            else:
                action = self.env.action_space.sample()

        observation, reward, terminated, truncated, info = self.env.step(action)  # Take action

        print(f"Action taken: {action}, Reward: {reward}")
        result.append((action, observation, reward, terminated, truncated, info))
        return result

    def successors_and_costs(self, state):
        result = []

def uniform_cost_search(problem):
    frontier = queue.PriorityQueue()
    frontier.put((0, problem.start_state(), list()))
    while frontier:
        past_reward, state, solution = frontier.get()
        if problem.goal_test(state):
            return (past_reward, solution)
        for action, new_state, reward, terminated, truncated, info in problem.take_action(state):
            frontier.put((past_reward + reward, new_state, solution + action))

def print_solution(solution):
    total_cost, history = solution
    print(f"Total cost: {total_cost}")
    for item in history:
        print(item)

problem = TaxiProblem()
#start_state = problem.start_state()
#problem.decode_state(start_state)
#for _ in range(50):
#    problem.take_action(start_state)
#print(start_state)
print_solution(uniform_cost_search(problem))