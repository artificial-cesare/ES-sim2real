# ES-sim2real optimization

This project was part of a research internship conducted by Cesare Maria Dalbagno under the supervision of Dr. Giacomo Spigler at the AI for Robotics Lab (AIR-Lab), Department of Cognitive Science and Artificial Intelligence Tilburg University, Tilburg, The Netherlands. 

The project aimed to investigate the effectiveness of Evolution Strategies (ES) and Meta-Learning in combination with Domain Randomization in optimizing the parameters of a neural network that controls a robot in a simulated environment. The goal was to transfer the optimized parameters to the real robot and test its performance in the real world, therefore bridging the gap between simulation and reality.

Even though our learning framework yielded an improve in performance in the simulations we ran, it was not significantly better than ES alone. More work has to be done in the direction of a more complex controller and a more complex environment to see the real benefits of ES and mDRL, eventually transfering to a real world environment.

## Research question:
Can an agent trained in simulation using meta-deep reinforcement learning (mDRL) with evolutionary strategies (ES) and domain randomization (DR) generalize to the real world? 

## Methodology:
The project started with a through test of our framework on 4 MuJoCo environments (Ant, Half_Cheetah, Hopper and Humanoid) with different randomization techniques. The results hinted that the framework was working as expected, even though the results did not show a significant improvement over ES alone, even though the computational cost was higher.
We then wanted to proceed with a more complex task, by controlling the Franka FR3 robot in a real world environment. Back when we started though, there was no FR3 environment available on MuJoCo, therefore we spent some time trying to create our own environment. We found some issues defining the reward function and making the controller in simulation more similar to the real world one. 

## Results:
No definitive conclusions can be drawn from the results we obtained, as the real world experiment was not conducted. The results from the simulations suggest that the framework should work, but in order to uncover the real benefits of ES and mDRL more experiments have to be conducted.

## Code structure:
The project was developed in Python using the MuJoCo physics engine and the Gymnasium library. The code is structured in the following way:

There are four main folders: 

    _tmp_full_dr: contains the files to create, train and test the randomized environments in MuJoCo. The files are structured in the following way:
    - `create_rand_json_mujoco.py`: creates the randomized environments specs in a json file
    - `train_test_mujoco.py`: trains and tests the randomized environments (swithcing between training and testing is done by changing the `train` variable in the code)
    - `dr_wrapper.py`: contains the class that wraps the randomized environment
    - Any folder contains the JSON and the file to run the environment
    - Plots: contains the plots of the results obtained in the simulations
    - Results: contains the results of the parameter sweep to set a baseline for the experiments
    - HP_sweep: contains the files to run the hyperparameter sweep to set the best parameters for the experiments

    Meta_RL: 
    - 2024_sim2real_via_metalearn: contains files to run and test ES-mDRL in MuJoCo using MPI
    - Scripts: contains the scripts to run the experiments
        - dr_fetch : contains the files to run the experiments on the Fetch environment
        - FR3:
            - fr3_env_mujoco:
                - fr3_robot: assets, xml and .py env of the robot
                - fr3_tasks: class to specify tasks for the robot
            - robotics_scripts: utils fns that might be handy if libraries conflicts occur

    Paper_and_presentation: contains the temporary paper and the presentation of the project
