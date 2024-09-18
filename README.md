
Done by:
Islam Mahfouz Al-Huda 202073257
Abdulmalik Farea Masoud 202174067
Catch The Ball AI Game
=======================

This project implements a simple game using **Python** where a basket catches a falling ball. The basket is controlled using Deep Q-Learning (DQN) from the `Stable Baselines3` library.

Game Features
-------------

- The game is built using `Pygame` for rendering.
- A custom environment is created using `Gymnasium` to manage the game's logic.
- Deep Q-Learning (DQN) is used to train the AI model that controls the basket.
- The AI learns to move the basket left, right, or stay still to catch the falling ball.

Installation
------------

To run this project, you'll need to install the following dependencies:

.. code-block:: bash

    pip install pygame gymnasium numpy stable-baselines3

Usage
-----

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/Islam776-prog/AI-Catch-The-Ball-Game

2. Run the Python script to start training the AI model:



3. The AI will learn to catch the ball after a few thousand training steps. You can also load a pre-trained model to see it in action.

File Structure
--------------

- `catch_the_ball_env.py`: Defines the custom Gym environment and game logic.
- `catch_the_ball_dqn.zip`: The saved AI model after training.



Developed by: [Islam and Abdulmalik]
