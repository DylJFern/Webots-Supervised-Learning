# Webots-Supervised-Learning
## Introduction
In this project we explore the world of robotics and machine learning through the lens of Webots, an open-source realistic 3D robot development environment used to design and model, program and simulate, as well as visualize robotics, equipped with an extensive library of robot models, sensors, actuators, environments, and so on.

## Supervised Learning
Supervised learning is a fundamental concept in machine learning, a subfield of artificial intelligence. It involves training a machine learning algorithm on a labeled dataset, where each data point is associated with a known outcome. The algorithm learns to map input data to the correct output by generalizing from the training examples. This enables the algorithm to make predictions or decisions on new, unseen (test) data. In other words, supervised learning algorithms learn from historical data to make future predictions or decisions.

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/assets/128000630/899000c5-e11e-4e30-b555-a97e874eda63" alt="Supervised Learning">
</div>
<p align="center"><i>Figure 1: <a href="https://www.kdnuggets.com/understanding-supervised-learning-theory-and-overview">Supervised learning</a></i></p>

### Components
* Input data: features or attributes used as input to the algorithm for prediction.
* Output (target): the algorithm aims to predict this value, it can be either continuous (regression) or categorical (classification).
* Training: the algorithm is trained using the labeled dataset, adjusting its internal parameters to minimize the difference between its predictions and the actual outcomes.
* Model: the result of training is a model, which is a mathematical function that captures the patterns and relationships in the data, it can be used to make predictions on unseen data.

### Example Applications
* Image and speech recognition
* Natural language processing
* Predictive analytics
* Risk assessment
* Fraud detection

## Project Overview
The project is centered around a robot operating within the Webots simulation environment. The robot is trained using supervised machine learning, where it is initially controlled through simple user (keyboard) inputs:
* `'W'` (forward)
* `'A'` (turn left)
* `'D'` (turn right)
* `'S'` (reverse)

During training, these keys are used to control the robot's movement such that it is able to traverse the different environments, in this stage it collects data consisting of 128 lidar sensor readings (distance measurements) and the corresponding user inputs at each timestep (set as 32ms). These lidar sensor readings measure depth information (in meters) and help provide detailed information about the robot's surroundings, for example they can used to nearby detect obstacles. The data collected (consists of an array taken at each timestep) is combined into a single dataset and is used to train machine learning models, allowing the robot to learn from the operator's actions.

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/lidar_image.jpg" alt="Lidar Image">
</div>
<p align="center"><i>Figure 2: <a href="https://cyberbotics.com/doc/reference/lidar">Lidar image</a></i></p>

Once the supervised learning phase is complete, the robot transitions to autonomous navigation. Now equipped with trained models, the robot leverages its knowledge to make decisions based on real-time lidar data (by making predictions on which actions to issue). It can autonomously explore its environment, detect obstacles, and respond to them intelligently, mirroring the behaviours it learned during training.

## Training
### Controller Design
We need to design a controller to control the robot via keyboard inputs and collect lidar data.
```python
# control loop
while robot.step(TIME_STEP) != -1:
    # read keyboard input
    key = keyboard.getKey()

    # control the robot based on user input
    for i in range(0,4):
        wheels[i].setPosition(float('inf'))
        # if 'W' (caps-lock active) or 'w' key is pressed
        if key == 87:
            wheels[i].setVelocity(max_speed)
        # if 'S' (caps-lock active) or 's' key is pressed
        elif key == 83:
            wheels[i].setVelocity(-max_speed)
        # if 'D' (caps-lock active) or 'd' key is pressed
        elif key == 68:
            if i == 0 or i == 2:
                wheels[i].setVelocity(max_speed)
            else:
                wheels[i].setVelocity(-max_speed)
        # if 'A' (caps-lock active) or 'A' key is pressed
        elif key == 65:
            if i == 0 or i == 2:
                wheels[i].setVelocity(-max_speed)
            else:
                wheels[i].setVelocity(max_speed)
        else:
            wheels[i].setVelocity(0)

    # if any key is not pressed (value is -1), replace it with NaN
    if key == -1:
        key = np.nan
    
    # collect LIDAR data and keyboard input
    lidar_data_row = lidar.getRangeImage() # get lidar data for each 'sensor point' as an array of distance measurements (at each TIME_STEP = 32)
    data_row = [key] + lidar_data_row
    data.append(data_row)
```
In this simple control loop we are controlling the robot's movement, where only single key input is read at a time. For instance, if `'W'` is pressed and held the robot will continue to move forward, if we then press `'A'` (while still holding `'W'`), `'A'` will override `'W'` and the robot will now turn left until `'A'` is released, at which point the robot will continue moving forward (assuming `'W'` is still being held). 

At each timestep during the simulation, the keyboard input and the 128 lidar data readings are recorded as a single row.

### Data Collection




