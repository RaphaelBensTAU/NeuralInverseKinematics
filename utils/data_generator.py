import numpy as np
import ikpy.chain
import pickle
import h5py

NUM_JOINTS = 7
NUM_SAMPLES_TO_GENERATE = 1000000
URDF_FILE_PATH = "/home/raphael/home/raphael/PycharmProjects/hyperikArxiv/assets/franka/urdf/franka_ik.urdf"
OUTPUT_FILE_PATH = f'franka_train_{NUM_SAMPLES_TO_GENERATE}'

r_arm = ikpy.chain.Chain.from_urdf_file(URDF_FILE_PATH)

upper = []
lower = []
for i in range(1, len(r_arm.links) - 1):
    lower.append(r_arm.links[i].bounds[0])
    upper.append(r_arm.links[i].bounds[1])

upper = np.array(upper)
lower = np.array(lower)

results = []
inputs = []
for i in range(NUM_SAMPLES_TO_GENERATE):
    random_joint_angles = (upper - lower) * np.random.rand(len(upper)) + lower
    joint_angles = [0] + [random_joint_angles[i] for i in range(NUM_JOINTS)] + [0]
    real_frame = r_arm.forward_kinematics(joint_angles)
    results.append(real_frame[:3, 3])
    inputs.append(random_joint_angles)

results = np.array(results)
inputs = np.array(inputs)

f_train = h5py.File(OUTPUT_FILE_PATH, 'w')
f_train.create_dataset('results', data=results)
f_train.create_dataset('inputs', data=inputs)
f_train.close()
