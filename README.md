# Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning

## Getting Strated
CoWorld is implemented and tested on Ubuntu 20.04 with python == 3.7, PyTorch == 1.13.1:

1) Create an environment
```bash
conda create -n coworld python=3.7
conda activate coworld
```
2) Install dependencies
```bash
pip install -r requirements.txt
```

3) a) Copy all files in `./modified_dmc_xml` to the DMC directory in your conda environment, such as `/home/.conda/envs/your_env_name/lib/python3.7/site-packages/dm_control/suite/`. 
b) Put Offline training data and evaluation data into `./offline_train_data/` and `./offline_eval_data/`. The complete offline datasets will be released upon acceptance.

## Meta-World/RoboDesk/DMC
1. Training command on Meta-World:  
```bash
python3 co_training.py --source_task metaworld_drawer-close --target_task metaworld_door-close --configs defaults metaworld
```
2. Training command on RoboDesk:  
```bash
python3 co_training.py --source_task metaworld_button-press --target_task robodesk_push_green --configs defaults robodesk
```
3. Training command on DMC:  
```bash
python3 co_training.py --source_task walker_walk --target_task walker_downhill --configs defaults dmc
```


## Acknowledgement
The codes refer to the implemention of [dreamer-torch](https://github.com/jsikyoon/dreamer-torch). Thanks for the authors！



