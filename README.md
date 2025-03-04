# ES-sim2real Optimization

This project explores the effectiveness of Evolution Strategies (ES) and Meta-Learning, combined with Domain Randomization (DR), to optimize a neural network controlling a robot in a simulated environment. The ultimate goal is transferring these optimized parameters to a real robot, thereby bridging the gap between simulation and reality (sim2real gap).

Conducted by Cesare Maria Dalbagno under the supervision of Dr. Giacomo Spigler at the AI for Robotics Lab (AIR-Lab), Department of Cognitive Science and Artificial Intelligence, Tilburg University (Netherlands), the work showed that while performance improved in simulation, it was not significantly better than ES alone. Further research with more complex controllers, environments, and real-world experiments is needed to fully realize the benefits of ES and meta-deep reinforcement learning (mDRL).

## Research Question
Can an agent trained in simulation using meta-deep reinforcement learning with evolutionary strategies and domain randomization generalize effectively to the real world?

## Methodology
1. **Initial Tests:**  
   - Four MuJoCo environments (Ant, HalfCheetah, Hopper, and Humanoid) were used with different randomization strategies.  
   - While our mDRL framework worked as expected, it did not substantially outperform ES alone, and had higher computational costs.
   ![Alt text](Paper_and_presentation\hopper.gif)

2. **Franka FR3 Robot:**  
   - We aimed to implement our approach on a Franka FR3 robot.  
   - Because no existing FR3 MuJoCo environment was available, we attempted to create one.  
   - Challenges arose in defining a suitable reward function and ensuring the simulation controller closely mirrored real-world conditions.
   

## Results
The real-world experiment was not completed, so definitive conclusions cannot be drawn. Simulation results suggest the framework is viable (see picture), but further experiments are required to uncover the full potential of ES and mDRL in real-world settings.

![Alt text](Paper_and_presentation\es_sim2real_preliminary.png)

## Code Structure
The project is developed in Python using MuJoCo and the Gymnasium library. Key directories include:

### _tmp_full_dr
- **create_rand_json_mujoco.py**  
  Generates JSON files specifying randomized environments.
- **train_test_mujoco.py**  
  Trains and tests the randomized environments (use the `train` variable in the code to switch modes).
- **dr_wrapper.py**  
  Wraps the randomized environments.
- **Folders and JSON Files**  
  Contain per-environment settings and scripts.
- **Plots**  
  Stores simulation result graphs.
- **Results**  
  Contains baseline parameter sweep outputs.
- **HP_sweep**  
  Houses scripts for hyperparameter tuning.

### Meta_RL
- **2024_sim2real_via_metalearn**  
  Code to run and test ES-mDRL in MuJoCo using MPI.
- **Scripts**  
  Houses experiment scripts.  
  - **dr_fetch**: Experiments for the Fetch environment.  
  - **FR3**:
    - **fr3_env_mujoco**:  
      - **fr3_robot**: Assets, XML files, and Python environment setup for the robot.  
      - **fr3_tasks**: Task definitions for the robot.  
    - **robotics_scripts**: Utility functions to handle potential library conflicts.

### Paper_and_presentation
- Draft versions of the paper and presentation slides.
