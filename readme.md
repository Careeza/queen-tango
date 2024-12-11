# Queen & Tango Puzzle Solver

This project provides Python-based automation to solve two puzzle types: **Queen** and **Tango**.  
It captures your screen, detects the puzzle, computes a solution, and then simulates mouse clicks to fill in the correct answers.

## Features

- **Screen Capture & Detection**: Identifies puzzle grids on the screen using image processing (OpenCV).
- **Automated Solving**:
  - **Queen Puzzle**: Places a queen in each row, column, and color region, ensuring no two queens touch (even diagonally).
  - **Tango Puzzle**: Fills a 6x6 grid with suns and moons, ensuring:
    - Each row/column has exactly 3 suns and 3 moons.
    - No three identical symbols appear consecutively.
    - All adjacency constraints (equal '=' or opposite '×') are respected.
- **Automated Input**: Uses PyAutoGUI to simulate mouse clicks and solve the puzzle in your browser or application.

## Requirements

- Python 3.x
- All required Python packages are listed in the `requirements.txt` file.

Install them via pip:

```bash
pip install -r requirements.txt
```

## Before You Start

You must pre-generate the solution files for both puzzles before running the main solver scripts. To do this, run:

```bash
python generate_solutions.py
```

This will create:
- `tango_solution.npy` containing all valid Tango solutions.
- `queen_solution_n.npy` files for n from 5 to 10, containing precomputed queen solutions.

## Usage

1. **Open the Puzzle**: Make sure the puzzle is visible on your screen.
2. **Run the Solver**:
	- For the Queen puzzle: python `queen_solver.py`
	- For the Tango puzzle: python `tango_solver.py`
3. **Toggle Detection**: Press `c` to start/stop puzzle detection and solving.
4. **Exit**: Press `ESC` at any time to stop the program.

## Notes
- The solver uses image processing heuristics, so it may need adjustments for different screen sizes or puzzle variants.
- Ensure the puzzle remains stable on the screen for accurate detection.
