# Neural Inverse Kinematics

This repository is the official implementation of Neural Inverse Kinematics (ICML 2022) by Raphael Bensadoun et al.

**[arXiv](https://arxiv.org/pdf/2205.10837.pdf) | [Code](https://github.com/RaphaelBensTAU/NeuralInverseKinematics)**

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate neural_inverse_kinematics
```
Models, plots and metrics are stored in NeuralInverseKinematics/runs.

## Training on custom kinematic chain
Please use utils/data_generator.py script to generate data and run the next command -

```
python train.py --chain-path PATH_TO_KINEMATIC_CHAIN.urdf --train-data-path PATH_TO_TRAINING_DATA.hdf5 --test-data-path PATH_TO_VALIDATION_DATA.hdf5 --num-joints NUM_OF_JOINTS 
```

## Training on 2D 'snake' arms
## 2 joints
```
python train.py --chain-path assets/snakes_2D/urdf/snake_2D_2j.urdf --train-data-path data/snakes_2D/2j/snake_train_20k.hdf5 --test-data-path data/snakes_2D/2j/snake_val_1k.hdf5 --num-joints 2 --num-gaussians 2
```
## 4 joints
```
python train.py --chain-path assets/snakes_2D/urdf/snake_2D_4j.urdf --train-data-path data/snakes_2D/4j/snaka_2d_4j_1000k_train.hdf5 --test-data-path data/snakes_2D/4j/snaka_2d_4j_10k_val.hdf5 --num-joints 4 --num-gaussians 2
```

## Training on digit right arm

```
python train.py --chain-path assets/digit/urdf/digit_r_arm.urdf --train-data-path data/digit/train_20000.hdf5 --test-data-path data/digit/val_1000.hdf5 --num-joints 4
```

## Training on UR5 

```
python train.py --chain-path assets/UR5/urdf/ur5_robot.urdf --train-data-path data/ur5/ur5_train_1m.hdf5 --test-data-path data/ur5/ur5_val_1k.hdf5 --num-joints 6
```

## Training on Franka
```
python train.py --chain-path assets/franka/urdf/franka_ik.urdf --train-data-path data/franka/franka_train_1000k.hdf5 --test-data-path data/franka/franka_1k_val.hdf5 --num-joints 7
```
