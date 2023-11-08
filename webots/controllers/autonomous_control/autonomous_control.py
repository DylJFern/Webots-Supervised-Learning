from controller import Supervisor, Keyboard
import numpy as np
import joblib # allows us to load the trained model
import pandas as pd
import os

TIME_STEP = 32

# initialize the supervisor (special type of robot) instance
robot = Supervisor()

# initialize the keyboard device instance
keyboard = Keyboard()
keyboard.enable(TIME_STEP) # enable keyboard sampling

# extract the name of the currently running world file
world_namepath = robot.getWorldPath()
world_filename = os.path.basename(world_namepath)

# extract the world number from the world file name  
world_number = int(world_filename.lstrip("mars_test").rstrip('.wbt'))

# determine the directory path based on the world file name
if "mars_test1.wbt" in world_filename:
    directory_path = "../../../data/testing/env1"
elif "mars_test2.wbt" in world_filename:
    directory_path = "../../../data/testing/env2"
else:
    # create a new directory path with extracted world number
    directory_path = f"../../../data/testing/env{world_number}" 

# create the directory folder if it does not exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# define a dictionary to load existing completion times
completion_times = {}
# load existing completion times from the file
completion_time_file = os.path.join(directory_path, f"completion_time_env{world_number}.txt")
if os.path.isfile(completion_time_file):
    with open(completion_time_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            model, time_value = line.strip().split(": ")
            time_value = time_value.replace(" seconds", "")
            completion_times[model] = float(time_value)

# set active model (i.e. comment out unused models and uncomment model to be used)
# model_name = "log_reg_model"
# model_name = "best_log_reg_model"
model_name = "rfc_model" 
# model_name = "best_rfc_model" 
# model_name = "mlpc_model"
# model_name = "best_mlpc_model"
# model_name = "xgboost_model"
# model_name = "best_xgboost_model"
model = joblib.load(f"../../../data/model/{model_name}.joblib") # load the trained model

# initialize the LIDAR instance
lidar = robot.getDevice("lidar")
lidar.enable(TIME_STEP) # enable lidar sampling
lidar.enablePointCloud() # enable PointCLoud to get distances from sensor to objects in environment

# initialize the motors
wheel_names = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
wheels = [] # empty list to store initialized motors of the wheels
for name in wheel_names:
    wheels.append(robot.getDevice(name))

max_speed = 2.00 # set the motor maximum speed (radians/second)

# initialize empty list of range measurement names
columns = [] 
# initialize empty list to store the range measurements and predicted actions
data = []

# add a column for predicted action
columns.append("Predicted Action")
# add columns for range measurement (resolution number = number of columns = unique number of 'sensor points')
for i in range(lidar.getHorizontalResolution()):
    columns.append(f"Range{i}")

# control loop
while robot.step(TIME_STEP) != -1:
    # read keyboard input
    key = keyboard.getKey()

    # get LIDAR (distance) measurements
    lidar_data_row = lidar.getRangeImage()

    # make the predictions based on LIDAR measurements
    prediction = model.predict([lidar_data_row])

    # control the robot based on the predicted action of model
    for i in range(0,4):
        wheels[i].setPosition(float('inf'))
        # if 'W' key is pressed (key code for 'W' is 87, but accepts 'w' without caps-lock) - this corresponds to xgboost mapping of "2"
        if prediction == 87 or prediction == 2:
            wheels[i].setVelocity(max_speed)
        # if 'S' key is pressed (key code for 'S' is 83, but accepts 's' without caps-lock) - this corresponds to xgboost mapping of "3"
        elif prediction == 83 or prediction == 3: # note: in the training, no reversing/backward movement was performed as it contradicted what we were trying to achieve (in training), as a result "S" is obsolete
            wheels[i].setVelocity(-max_speed)
        # if 'D' key is pressed (key code for 'D' is 68, but accepts 'd' without caps-lock) - this corresponds to xgboost mapping of "1"
        elif prediction == 68 or prediction == 1:
            if i == 0 or i == 2:
                wheels[i].setVelocity(max_speed)
            else:
                wheels[i].setVelocity(-max_speed)
        # if 'A' key is pressed (key code for 'A' is 65, but accepts 'a' without caps-lock) - this corresponds to xgboost mapping of "0"
        elif prediction == 65 or prediction == 0:
            if i == 0 or i == 2:
                wheels[i].setVelocity(-max_speed)
            else:
                wheels[i].setVelocity(max_speed)
        else:
            wheels[i].setVelocity(0)
    
    print("Predicted Action:", prediction[0])

    # store the predicted action and LIDAR data
    data_row = [prediction[0]] + lidar_data_row
    data.append(data_row)
    
    # if 'P' key is pressed (key code for 'P' is 87, but accepts 'p' without caps-lock)
    if key == 80:
        # stop the robot's motion
        for i in range(0,4):
            wheels[i].setVelocity(0)
        
        # find the next available CSV file name
        i = 1
        while True:
            csv_filename = os.path.join(directory_path, f"lidar_data_{model_name}{i}_env{world_number}.csv")
            if not os.path.exists(csv_filename):
                break
            i += 1

        # create a DataFrame with specified column names
        lidar_data = pd.DataFrame(data, columns=columns)
        # save the collected keyboard inputs and LIDAR data to the CSV file
        lidar_data.to_csv(csv_filename, index=False)

        # retrieve the simulation time (in seconds), and in Webots this time is adjusted according to how fast the simulation is run
        total_time_seconds = robot.getTime() 

        # update the completion time for (only) the current model
        completion_times[model_name] = total_time_seconds

        # write (save) the completion times to the file
        with open(completion_time_file, "w") as file:
            for model, time_value in completion_times.items():
                file.write(f"{model}: {time_value:.3f} seconds\n")
        
        break # exit the while-loop

# message to inform user how to proceed
print("Pause and reload the simulation.")