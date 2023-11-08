# Webots-Supervised-Learning
## Introduction
In this project we explore the world of robotics and machine learning through the lens of Webots, an open-source realistic 3D robot development environment used to design and model, program and simulate, as well as visualize robotics, equipped with an extensive library of robot models, sensors, actuators, environments, and so on.

Our project revolves around a robot operating within the Webots simulation environment. The robot is trained using supervised machine learning, where it is initially controlled through simple user (keyboard) inputs 'W' (forward), 'A' (turn left), 'D' (turn right), 'S' (reverse). During training, these keys are used to control the robot's movement such that it is able to traverse the different environments, in this stage it collects data consisting of 128 lidar sensor readings (distance measurements) and the corresponding user inputs at each timestep (set as 32ms). These lidar sensor readings measure depth information (in meters) and help provide detailed information about the robot's surroundings, for example they can used to nearby detect obstacles. The data collected (consists of an array taken at each timestep) is combined into a single dataset and is used to train machine learning models, allowing the robot to learn from the operator's actions.

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/assets/128000630/7be1de8e-3f23-4c35-bdf6-8d08a12ab728" alt="Lidar Image">
</div>
<p align="center"><i>Figure 1: <a href="https://cyberbotics.com/doc/reference/lidar">Lidar image</a></i></p>

Once the supervised learning phase is complete, the robot transitions to autonomous navigation. Now equipped with trained models, the robot leverages its knowledge to make decisions based on real-time lidar data (by making predictions on which actions to issue). It can autonomously explore its environment, detect obstacles, and respond to them intelligently, mirroring the behaviours it learned during training.

## <br>Supervised Machine Learning
