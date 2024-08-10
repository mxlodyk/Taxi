import gymnasium as gym
import pygame

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

    def goal_state(self):
        # return observation
        pass

    def goal_test(self):
        # return self.env == self.goal_state()
        return False
        pass

    def decode_state(self, state):
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
        print(taxi_row, taxi_col, passenger_location, destination) # Print decoded state

    def take_action(self):
        action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(action)  # Take action
        print(f"Action taken: {action}, Reward: {reward}")

    #def

problem = TaxiProblem()
start_state = problem.start_state()
problem.decode_state(start_state)
for _ in range(50):
    problem.take_action()
print(start_state)