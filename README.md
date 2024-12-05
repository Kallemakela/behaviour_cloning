# Imitation learning experiment

- td3 bc

4
Pure BC: Mean Reward: 718.22 +/- 32.85, med: 762.34
Pure BC (Step 4): Mean Reward: 742.81 +/- 39.45, med: 866.65
Pure BC (Step 10): Mean Reward: 652.69 +/- 31.86, med: 645.82



## Speed



## Input

- 4 stacked B/W frames of size 48x48
- frame step is 4, so e.g. observation 16 includes frames 16, 12, 8, 4

![Input](car_racing_v3_obs.png)

## Usage

```bash

python record.py # record expert data
python process_recorded_data.py # process expert data

python imi.py # BC
python imi_ac.py # BC + Critic pretrain
python finetune.py # finetune BC model
python finetune_ac.py # finetune BC + Critic pretrain model
python baseline.py # PPO baseline

python compare.py
```