# Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning
#### Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning [NeurIPS, 2024]
Qi Wang*, Junming Yang*, Yunbo Wang, Xin Jin, Wenjun Zeng, Xiaokang Yang

[[arXiv]](https://arxiv.org/pdf/2305.15260)  [[Project Page]](https://qiwang067.github.io/coworld)

<img src="assets/overview.png" alt="Teaser image" height="400" />

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

3) Copy all files in `./modified_dmc_xml` to the DMC directory in your conda environment, such as `/home/.conda/envs/your_env_name/lib/python3.7/site-packages/dm_control/suite/`.
4) Download the offline dataset [here](https://drive.google.com/file/d/1XXXLyx8oX40EXl8GNVgU_PpoPJHolZGJ/view?usp=sharing). 

## Meta-World/RoboDesk/DMC
1. Training command on Meta-World:  
```bash
python3 co_training.py --source_task metaworld_drawer-close --target_task metaworld_door-close \
--offline_traindir 'offline_metaworld_data_path' \
--configs defaults metaworld 
```
2. Training command on RoboDesk:  
```bash
python3 co_training.py --source_task metaworld_button-press --target_task robodesk_push_green \
--offline_traindir 'offline_robodesk_data_path' \
--configs defaults robodesk
```
3. Training command on DMC:  
```bash
python3 co_training.py --source_task walker_walk --target_task walker_downhill \
--offline_traindir 'offline_dmc_data_path' \
--configs defaults dmc
```
## Citation
```bib
@inproceedings{wang2024making,
  title={Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning}, 
  author={Qi Wang and Junming Yang and Yunbo Wang and Xin Jin and Wenjun Zeng and Xiaokang Yang},
  booktitle={NeurIPS},
  year={2024}
}
```

## Acknowledgement
The codes refer to the implemention of [dreamer-torch](https://github.com/jsikyoon/dreamer-torch). Thanks for the authors！



