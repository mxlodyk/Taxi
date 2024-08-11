# Taxi-v3 Search Algorithms

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Design](#design)
- [Algorithm Descriptions](#algorithm-descriptions)

## Introduction
This project demonstrates the implementation of two search algorithms, **A\*** and **Dijkstra's Algorithm**, within the `Taxi-v3` environment from OpenAI's Gym library. The objective is to solve the Taxi problem by finding an optimal path for the taxi to pick up and drop off passengers.

## Features
- **A\* Search**: An implementation that uses a heuristic to guide the search for the optimal path.
- **Dijkstra's Algorithm**: A non-heuristic approach that finds the shortest path based on accumulated cost.
- **Environment Rendering**: Visualizes the solution path within the Taxi-v3 environment.
- **Comparison Mode**: Runs both algorithms in parallel and compares their solutions.

## Installation

### Prerequisites
- Python 3.8 or later
- OpenAI Gymnasium library
- Multiprocessing module (standard with Python)

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/taxi-v3-search-algorithms.git
    ```
2. Navigate to the project directory:
    ```bash
    cd taxi-v3-search-algorithms
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the program, execute the `main.py` script. You can do this either from the command line or by running the script in your preferred Integrated Development Environment (IDE).

### Command Line
Run the program by executing the `main()` function. You will be prompted to select one of the following options:
    

```bash
python main.py
```

## Design

![TaxiProblem Class Diagram](https://github.com/mxlodyk/Taxi/assets/class_diagram.png)

