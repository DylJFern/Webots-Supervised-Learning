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

### Example Training Paths 1
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

Above are the three training environments used to train the robot with example paths we (as the user) would take to arrive at the predefined destination, and throughout the training process we can see that these paths may vary. For example, in [Figure 3](README.md#example-training-paths) we may take sharp turns (hold `'A'` or `'D'` until the robot has rotated 90 degrees - defined by 'red' and 'blue' paths), gradual turns (defined by the 'green' path), or incremental turns (defined by the 'orange' path).

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

> Note 1: These example videos of train (user control) and test (autonomous) provided were initially performed to showcase how the robot would respond and get a idea for what the results may look like on simple environments (which have now all been used strictly for training as seen in the [example training paths section](README.md#example-training-paths). New test environments (not showcased yet) were created to provide more of a challenge.
> 
> Note 2: The LIDAR sensor type used in the videos was later replaced another type to correct problems associated with the original one, but the training concepts are the same.

### Example Training Paths 2
With the models trained on the data obtained by controlling the robot in the training environments, we had tested the robot in the testing environments. What we noticed was the robot still required additional training on specific situations that it may encounter in the testing environments.

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/additional_training_paths.jpg" alt="Training Environment 3 - Additional Example Paths">
</div>
<p align="center"><i>Figure 6: Training Environment 3 - Additional Example Paths</i></p>

Additional training was performed on specific situations such as the robot's ability in turning out of a corner, performing left- and right-hand as well as 180 degree/U-turns, and avoiding obstacles along the path (not just boundary ones, e.g. the border).

## Testing
The test environments were ideated and modeled over the duration of the project to help encapsulate complex environments but at the same time were reasonable such that the models fit to the training data should be able to make predictions on unseen data (or an unseen environment with familiar features).

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/webots/worlds/.mars_test1.jpg" alt="Testing Environment 1">
</div>
<p align="center"><i>Figure 7: Testing Environment 1</i></p>

<br>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/webots/worlds/.mars_test2.jpg" alt="Testing Environment 2">
</div>
<p align="center"><i>Figure 8: Testing Environment 2</i></p>

The Random Forest Classifier model was the initial algorithm selected when doing initial testing and was predominantly used as it was able to make accurate decisions in most cases when predicting inputs based on new unseen LIDAR data. However, it was still not without its faults (which will be later on with current issues - which could mainly be due to how the model is trained or even undertrained in additional situations). With the test environments fine tuned, we noticed that Random Forest Classifier was still able to successfully navigate the obstacles. However, we wanted to see how other models would compare, as such we introduced Logistic Regression, MLP Classifier, and XGBoost using "default" parameters and introduced hyperparameter tuning with grid search for all four models (for a total of 8 models).

### Issues
While the Random Forest Classifier is able to successfully navigate the test environment, it can still to fail to make correct decisions in certain situations as seen in these [videos](https://drive.google.com/drive/folders/1LCVFectN9OZ4fpbs6eDZ-ISFXEkipZut).

* in `rfc_model_issue1.mp4`, we can see the robot will turn in the direction of the largest LIDAR sensor readings, but it fails to generalize. The robot does not know what an open area is, it has been trained on environments where it is surrounded by obstacles and can make simple decisions where readings on one side may be larger or smaller than readings on the other side. While it is trained to move in the direction with the largest distances and attempt to reduce it, it is 'centered' where readings on both sides are approximately equal and so it "jitters" left and right trying to make a decision.
* in `rfc_model_issue2.mp4`, we can see that even slight variations can effect the robot's behaviour, but this issue could most likely be fixed just by training the robot to perform tighter turns. At the instance just before the right turn, the robot knows the furthest LIDAR readings are to the right; however, it overturns and is unable to fully correct itself (due to a lack of training data in such a situation), as such its wheel gets stuck on the edge of the wall. Had we extended the vertical or even horizontal just a bit further out or in, the robot should have no trouble avoiding the obstacles and getting stuck.
* in `rfc_model_issue3.mp4`, we can see somewhat a similar situation to the previous. In this case, the robot initially detects the furthest distance away coming out of the left turn as the corner and continues "hugging" the wall; however, when it gets close it realizes this is indeed not the case and attempts to perform a right turn but ends up getting stuck in a similar situation as the issue seen in `rfc_model_issue1.mp4` where it cannot make the decision on what direction to turn. In addition, the robot was never trained to reverse (`'S'`), had it been it may have been able to correct itself, but the reason it was not trained to reverse is because it contradicts how it has been trained (in other words, it wants to travel to the furthest distance way to reduce it and perform a turn to avoid colliding with the obstacle, if it were to reverse do the opposite by increasing this distance from the wall).
* in `rfc_model_issue4.mp4`, while it may seem obvious based on how it was trained or what the user would do, even though the robot wants to go to the right side of the environment world, it still fails to determine what direction to turn as it does what to reduce that distance, but at the same time it wants to avoid the obstacle, and so on.

So, when we mention the testing environments were ideated and modeled over the duration of the project, it was more they were designed and slightly modified such that it would create a challenge but would not introduce untrained or undertrained elements. The issues with the Random Forest Classifier model would also be consistent with the other models regardless of how "good" it is or how "well" tuned it is.

## Evaluation Metrics
To get a sense of model performance, we can look at measures such as time-based (how long was the robot able to travel for without getting stuck), distance to obstacle (how efficient the robot was at avoiding obstacles), and decision-making (how accurate was the robot at predicting the correct actions), as accuracy alone does not 'paint the full picture'.

### Time-Based
If it was not already evident, Random Forest Classifier with "default" parameters was able to successfully navigate both environments. We noticed that logitistic regression and its hyperparameter tuned variant were not able to properly distinguish between the majority class (`'W'`) and underrepresented classes (`'A'` and `'D'`), as such it was not even able to perform simple turns. In terms of performance Random Forest Classifier, XGBoost, and MLP Classifier have the potential to outperform Logistic Regression (which they did) on complex, non-linear problems. 

However, when comparing XGBoost and MLP Classifier to Random Forest Classifier, the "default" and "best" parameter variations failed to correctly distinguish between the classes during testing. This could be due to the fact that Random Forest Classifier typically has fewer hyperparameters to tune compared to MLP and XGBoost, it is known for its robustness to noisy data and outliers by filtering out irrelevant information, and is well-suited for imbalanced datasets (which is the case in this problem). XGBoost and MLP Classifier have a higher tuning complexitiy, but they were not fully utilized (instead typical/common parameters and values were used due to hyperparameter tuning being a computation expensive and time consuming process) which might be why these "best" parameter models failed to perform compared to their "default" parameter models or failed to compete with Random Forest Classifier.

### Distance-to-Obstacle
In '[test environment 2](https://drive.google.com/drive/folders/1GsqW1eqElkXM3ZpYG7s-LWtDq16zsEEa)', we can see that all models were effective at maintaining a distance of approximately 0.2 units from the obstacle (e.g. the wall). However, in '[test environment 1](https://drive.google.com/drive/folders/1bceqO_et1zOKtDafcC8bbPMSAqjvGEcN)', we notice that the Random Forest Classifier was consistently better at maintaining a sufficient distance from the obstacle, making it more successful at avoidance, thereby allowing it to travel further. For instance, the more distance from the obstacle, the easier it would be for the robot to make the distances as well as perform better turns (as previously mentioned, we never really training the robot with tight turns - which could be an undesired result of why models like XGBoost and MLP Classifier were unable to travel further).

However, this information should also be taken 'with a grain of salt', as the amount of data available for Random Forest Classifier is significantly larger compared to that of other models (as it was able to run longer, so it was stopped later). An example of this can be seen in 'test environment 2' when we simulate Logistic Regression even after we know it is unable to turn and just continues to drive forwards (in other words, as we introduce more data - simulate Logistic Regression longer, we notice that its average distance-to-obstacles is significantly lower).

### Decision-Making
#### Training Data (Input)
<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_training.PNG" alt="Training: Mean Distance Measurement Across LIDAR Range Index ">
</div>
<p align="center"><i>Figure 9: Training - Mean Distance Measurements Across LIDAR Range Index</i></p>

This graph depicts the mean distance measurement (or LIDAR range) across all of its 128 indices using a 95% confidence interval. For the training data, this says that we were driving forward with almost equal distance to obstacles on both sides and lots of room in front, we were turning left when seeing a obstacle close on our left but more open space on the right, and when we were turning right we would see an obstacle more closely on our right than left.

#### Testing Data (Action) - Test Environment 2
<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_best_log_reg.PNG" alt="Testing: best_log_reg - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_log_reg.PNG" alt="Testing: log_reg - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_best_mplc.PNG" alt="Testing: best_mlpc - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_mplc.PNG" alt="Testing: mlpc - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_best_rfc.PNG" alt="Testing: best_rfc - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_rfc.PNG" alt="Testing: rfc - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_best_xgboost.PNG" alt="Testing: best_xgboost - Mean Distance Measurements Across LIDAR Range Index ">
</div>

<div align="center">
  <img src="https://github.com/DylJFern/Webots-Supervised-Learning/blob/master/images/128_mean_dist_value_testing_xgboost.PNG" alt="Testing: xgboost - Mean Distance Measurements Across LIDAR Range Index ">
</div>
<p align="center"><i>Figure 10: Testing - Mean Distance Measurements Across LIDAR Range Index</i></p>

For the testing data (for 'test environment 2'), we notice that Random Forest Classifier is able to most closely replicate the behaviour seen in training (user control) within a testing environment (through predicted actions). This is indicated by its ability to make decisions in real-time more effectively than the other models examined. One may notice that "best_log_reg" and "log_reg" do not have any plots for `'A'` and `'D'` control, and once again (as seen in the video) this is due to its inability to make complex non-linear decisions leading to inproper classification (in other words, Logistic Regression was unable to perform any turns as it was not able to distinguish `'A'` and `'D'` classes from `'W'` due to them being underrepresented).
