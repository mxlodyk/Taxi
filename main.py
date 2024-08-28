import multiprocessing
import sys
from time import time
import taxi_problem as taxi
import algorithms

# ====================
# Main Processing Loop
# ====================
def main():

    # Environment seed.
    current_seed = 100

    # Display menu.
    selection = -1
    while selection != 5:
        selection = input("Please select a search algorithm:\n"
                        "1. A* search\n"
                        "2. Iterative deepening A* search\n"
                        "3. Compare algorithms\n"
                        "4. Change seed\n"
                        "5. Quit\n")
        
        # Selection 1: A* search.
        if selection == "1":
            problem = taxi.TaxiProblem(seed=current_seed)
            # Record time.
            t0 = time()
            # Perform A* search.
            solution = algorithms.a_star_search(problem)
            t1 = time() - t0
            if solution:
                # Print results.
                print("A* Solution actions:", solution)
                print(f"Speed: {t1:.2f}")
                # Render solution.
                problem.render_solution(solution)

        # Selection 2: IDA* search.
        elif selection == "2":
            # Instantiate problem.
            problem = taxi.TaxiProblem(seed=current_seed)
            # Record time.
            t0 = time()
            # Perform IDA* search.
            solution = algorithms.iterative_deepening_a_star_search(problem)
            t1 = time() - t0
            if solution:
                # Print results.
                print("IDA*'s Solution actions:", solution)
                print(f"Speed: {t1:.2f}")
                # Render solution.
                problem.render_solution(solution)

        # Selection 3: Compare algorithms.
        elif selection == "3":
            # Instantiate events for running each algorithm.
            event1 = multiprocessing.Event()
            event2 = multiprocessing.Event()
            # Create multiprocessing queue for storing solutions.
            solution_queue = multiprocessing.Queue()
            # Create processes for running algorithms
            process1 = multiprocessing.Process(target=algorithms.run_algorithm, args=("a_star_search", event1, solution_queue, current_seed))
            process2 = multiprocessing.Process(target=algorithms.run_algorithm, args=("iterative_deepening_a_star_search", event2, solution_queue, current_seed))
            # Start processes.
            process1.start()
            process2.start()
            # Wait for both algorithms to finish.
            event1.wait()
            event2.wait()
            # Join processes.
            process1.join()
            process2.join()

            while not solution_queue.empty():
                # Get solution from multiprocessing queue.
                algorithm_name, solution = solution_queue.get()
                if solution:
                    print(f"{algorithm_name} rendering...")
                    # Instantiate problem for rendering.
                    problem = taxi.TaxiProblem(seed=current_seed)
                    # Render solutions.
                    problem.render_solution(solution)
                else:
                    raise ValueError(f"Unexpected solution from {algorithm_name}: {solution}")

        # Selection 4: Change seed.
        elif selection == "4":
            # Check input is integer.
            try:
                # Get input.
                new_seed = int(input("Enter seed:\n"))
                # Check input is positive integer.
                if new_seed > 0:
                    # Assign new seed.
                    current_seed = new_seed
                else:
                    print("The seed must be a positive integer.")
            except ValueError:
                print("The seed must be a positive integer.")

        # Selection 5: Quit.
        elif selection == "5":
            problem.env.close()
            sys.exit(0)

if __name__ == "__main__":
    main()