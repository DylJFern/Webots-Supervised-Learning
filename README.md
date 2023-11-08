# Webots-Supervised-Learning
## Introduction
In this project we explore the world of robotics and machine learning through the lens of Webots, an open-source realistic 3D robot development environment used to design and model, program and simulate, as well as visualize robotics, equipped with an extensive library of robot models, sensors, actuators, environments, and so on.

## Supervised Learning
Supervised learning is a fundamental concept in machine learning, a subfield of artificial intelligence. It involves training a machine learning algorithm on a labeled dataset, where each data point is associated with a known outcome. The algorithm learns to map input data to the correct output by generalizing from the training examples. This enables the algorithm to make predictions or decisions on new, unseen (test) data. In other words, supervised learning algorithms learn from historical data to make future predictions or decisions.

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/supervised_learning_example.webp" alt="Supervised Learning">
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

During training, these keys are used to control the robot's movement such that it is able to traverse the different environments, in this stage it collects data consisting of 128 LIDAR sensor readings (distance measurements) and the corresponding user inputs at each timestep (set as 32ms). These LIDAR sensor readings measure depth information (in meters) and help provide detailed information about the robot's surroundings, for example they can used to nearby detect obstacles. The data collected (consists of an array taken at each timestep) is combined into a single dataset and is used to train machine learning models, allowing the robot to learn from the operator's actions.

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/lidar_image.jpg" alt="LIDAR Image">
</div>
<p align="center"><i>Figure 2: <a href="https://cyberbotics.com/doc/reference/lidar">LIDAR Image</a></i></p>

Once the supervised learning phase is complete, the robot transitions to autonomous navigation. Now equipped with trained models, the robot leverages its knowledge to make decisions based on real-time LIDAR data (by making predictions on which actions to issue). It can autonomously explore its environment, detect obstacles, and respond to them intelligently, mirroring the behaviours it learned during training.

## Training
### Controller Design
We need to design a controller to control the robot via keyboard inputs and collect LIDAR data.
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

At each timestep during the simulation, the keyboard input and the 128 LIDAR data readings are recorded as a single row.

### Data Collection
LIDAR (light detection and ranging) is a method for determining ranges by targeting an object or a surface with a laser and measuring the time for the reflected light to return to the receiver.

In Webots, we utilize LIDAR sensors to collect data for training our robot. The robot is controlled through user inputs, specifically `'W'` (forward), `'A'` (turn left), `'D'` (turn right), and `'S'` (reverse) using the keyboard. At each time step, the robot captures data from its LIDAR sensor and this data consists of 128 distance measurements, providing a detailed view of the surroundings.

Within each of the training environments, we would follow a relative path to end up at a predefined location (that we decided upon in advance) to train the robot. While traveling the path we would consider important factors to help simplify the learning process, these consisted of:

1) Travel forwards (as often as possible), and in the direction with the largest distance measurements based on LIDAR sensor data.
2) When an obstacle interferes with our path (in other words, as the distance readings approach zero), turn in a direction to not only avoid it but also repeat the process in the previous step (maximize the distance to the next obstacle(s)) at that instance of time.

Essentially, we would travel forward (ideally, in a direction with the largest distance measurements) and attempt to reduce them (to values close to zero) at which point an obstacle obscures our path, then in an attempt to avoid the obstacle we turn in a direction that would again maximize these readings (last seen, at the instance just before the turn is performed). In other words, for this learning approach, the robot would more or less "hug" the obstacle as it travels alongside it, and only made turns when an obstacle blocked the front of the robot.

### Example Training Paths
<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/training_paths1.jpg" alt="Training Environment 1 - Example Paths">
</div>
<p align="center"><i>Figure 3: Training Environment 1 - Example Paths</i></p>

<br>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/training_paths2.jpg" alt="Training Environment 2 - Example Paths">
</div>
<p align="center"><i>Figure 4: Training Environment 2 - Example Paths</i></p>

<br>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/training_paths3.jpg" alt="Training Environment 3 - Example Paths">
</div>
<p align="center"><i>Figure 5: Training Environment 3 - Example Paths</i></p>

Above are the three training environments used to train the robot with example paths we (as the user) would take to arrive at the predefined destination, and throughout the training process we can see that these paths may vary. For example, in [Figure 3](Example-Training-Paths) we may take sharp turns (hold `'A'` or `'D'` until the robot has rotated 90 degrees - defined by 'red' and 'blue' paths), gradual turns (defined by the 'green' path), or incremental turns (defined by the 'orange' path).

Although it is cruical we stick to those 'important factors' previously listed for training, it is just as beneficial to include some variation in how it is trained such that the robot is exposed to different scenarios and can adapt accordingly. Alone, this variation does not produce successful results, but when combined into a single DataFrame that is used to train the model(s), it would be able to make effective predictions. An example seen [here](https://drive.google.com/drive/folders/1ECIexPmrRcgrG8U1BK7BgpV55TZMp_gV), where the robot was trained on the same environment in different ways and for different durations (`train_lidar_data{i}`) with each trained dataset fit to a Random Forest Classifier model (using default parameters) and tested in two different environments (`test{j}_lidar_data{i}`). 

Here we see that:
* `train_lidar_data3`: (00:02:19 or ~140s run time * 32ms timesteps ≈ 4480 rows of data)
  * Effective in the testing environments due to the amount of available data it was exposed to. 
  * It implemented those 'important factors', e.g. "hugging" the wall and then performed turns in the direction that maximized LIDAR sensor readings (when an obstacle obscured it from moving forwards).
* `train_lidar_data2` (≈ 2720 rows of data)
  * Did not have much data to successfully navigate the testing environments.
  * Performed less than ideal turns.
* `train_lidar_data1` (≈ 2912 rows of data)
  * Did not have much data to successfully navigate the testing environments.
  * Performed better than the previously trained `train_lidar_data2` model due to more refined turns with sufficient space.

> Note: These example videos of train (user control) and test (autonomous) provided were initially performed to showcase how the robot would respond and get a idea for what the results may look like on simple environments (which have now all been used strictly for training as seen in the [example training paths section](Example-Training-Paths). New test environments (not showcased yet) were created to provide more of a challenge.

