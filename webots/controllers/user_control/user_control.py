from controller import Robot, Keyboard
import numpy as np
import pandas as pd
import os # allow us to interact with the operating system
# # provides tools for easy and efficient serialization of Python objects
# import pickle 
# import joblib # designed to work well with numerical data (e.g. large NumPy arrays) and scientific data structures

TIME_STEP = 32

# initialize the robot instance
robot = Robot()

# initialize the keyboard device instance
keyboard = Keyboard()
keyboard.enable(TIME_STEP) # enable keyboard sampling

# initialize the LIDAR instance
lidar = robot.getDevice("lidar")
lidar.enable(TIME_STEP) # enable lidar sampling
lidar.enablePointCloud() # enable PointCloud to get distances from sensor to objects in environment

# initialize the motors
wheel_names = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
wheels = [] # empty list to store initialized motors of the wheels
for name in wheel_names:
    wheels.append(robot.getDevice(name))

max_speed = 2.00 # set the maximum motor speed (radians/second)

# initialize empty list of range measurement names
columns = [] 
# initialize empty list to store the range measurements and keyboard inputs
data = []

# add a column for keyboard input
columns.append("Keyboard Input")
# add columns for range measurement (resolution number = number of columns = unique number of 'sensor points')
for i in range(lidar.getHorizontalResolution()):
    columns.append(f"Range{i}")

# control loop
while robot.step(TIME_STEP) != -1:
    # read keyboard input
    key = keyboard.getKey()

    # control the robot based on user input
    for i in range(0,4):
        wheels[i].setPosition(float('inf'))
        # if 'W' key is pressed (key code for 'W' is 87, but also accepts lowercase 'w' - when caps-lock is inactive)
        if key == 87: # note: Webots characters 'W' and 'w' did not work, as such the ASCII code was used
            wheels[i].setVelocity(max_speed)
        # if 'S' key is pressed (key code for 'S' is 83, but also accepts lowercase 's' - when caps-lock is inactive)
        elif key == 83:
            wheels[i].setVelocity(-max_speed)
        # if 'D' key is pressed (key code for 'D' is 68, but also accepts lowercase 'd' - when caps-lock is inactive)
        elif key == 68:
            if i == 0 or i == 2:
                wheels[i].setVelocity(max_speed)
            else:
                wheels[i].setVelocity(-max_speed)
        # if 'A' key is pressed (key code for 'A' is 65, but also accepts lowercase 'a' - when caps-lock is inactive)
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
    lidar_data_row = lidar.getRangeImage() # get the data reading for each 'sensor point' as an array of distance measurements (at each TIME_STEP)
    data_row = [key] + lidar_data_row
    data.append(data_row)

# find the next available CSV file name
i = 1
while True:
    csv_filename = f"../../../data/training/lidar_data{i}.csv"
    if not os.path.exists(csv_filename):
        break
    i += 1

# create a DataFrame with specified column names
lidar_data = pd.DataFrame(data, columns=columns)
# save the collected keyboard inputs and LIDAR data to the CSV file
lidar_data.to_csv(csv_filename, index=False)

# note: object serialization (methods below) did not work with Webots due to file size and compression time (resulting in forced termination) and was not necessary
# # serialize the collected data using pickle
# pickle_filepath = f"../../../data/training/lidar_data{i}.pkl"
# with open(pickle_filepath, 'wb') as pkl_file:
#     pickle.dump(data, pkl_file)

# # serialize the collected data using joblib without compression
# joblib_filepath1 = f"../../../data/training/lidar_data{i}.joblib1"
# with open(joblib_filepath1, 'wb') as jlib_file:
#     joblib.dump(data, jlib_file)

# # serialize the collected data using joblib with compression
# joblib_filepath2 = f"../../../data/training/lidar_data{i}.joblib2.gz"
# with open(joblib_filepath2, 'wb') as jlib_file_cmp:
#     joblib.dump(data, jlib_file_cmp, compress=('gzip', 3))