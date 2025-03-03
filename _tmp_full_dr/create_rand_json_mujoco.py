#this file creates a JSON file for each randomized environment

import json
import torch.nn as nn

def get_activation_fn_name(fn):
    return fn.__name__ if not isinstance(fn, str) else fn

# Common settings
common_policy_kwargs = dict(
                log_std_init=-2,
                activation_fn=nn.Tanh, #nn.ReLU
                net_arch=dict(pi=[256], vf=[256]),
                lstm_hidden_size=256,
                )

# Environment settings
environments = {
    "Hopper": {
        "env_name": 'Hopper-v4',
        "rand_class": "AutoRand", # "SetParams", "Rand"
        "train_steps": 30_000_000,
        "n_envs": 16,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 3,
        "gamma": 0.995,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
        "sde_sample_freq": 4,
        #"activation_fn": get_activation_fn_name(nn.Tanh), #nn.ReLU   ; hopper dr breaks through the suboptimal policy with relu but gets stuck with tanh; however, relu is much slower to learn and much more unstable
        "policy_kwargs": common_policy_kwargs,
        "dr_specs": {
            "body('torso').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('thigh').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('leg').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('foot').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('foot_joint').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('leg_joint').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('thigh_joint').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "geom('foot_geom').friction[0]": {"uniform": [0.8, 1.2], "type": "*"}
        }
    },
    "HalfCheetah": {
        "env_name": "HalfCheetah-v4",
        "train_steps": 30_000_000,
        "rand_class": "AutoRand",
        "n_envs": 16,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 3,
        "gamma": 0.995,
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        "sde_sample_freq": 4,
        #"activation_fn": get_activation_fn_name(nn.Tanh),
        "policy_kwargs": common_policy_kwargs,
        "dr_specs": {
            "body('torso').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('bthigh').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('bshin').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('bfoot').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('fthigh').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('fshin').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('ffoot').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "geom('floor').friction[0]": {"uniform": [0.8, 1.2], "type": "*"}
        }
    },
    "Ant": {
        "env_name": "Ant-v4",
        "train_steps": 30_000_000,
        "rand_class": "AutoRand",
        "n_envs": 16,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 3,
        "gamma": 0.995,
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        "sde_sample_freq": 4,
        #"activation_fn": get_activation_fn_name(nn.Tanh),
        "policy_kwargs": common_policy_kwargs,
        "dr_specs": {
            "body('torso').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('front_left_leg').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('front_right_leg').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('back_leg').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('right_back_leg').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('hip_1').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('hip_2').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('hip_3').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('hip_4').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "geom('floor').friction[0]": {"uniform": [0.8, 1.2], "type": "*"}
        }
    },
    "Humanoid": {
        "env_name": "Humanoid-v4",
        "train_steps": 30_000_000,
        "rand_class": "AutoRand",
        "n_envs": 16,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 3,
        "gamma": 0.995,
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        "sde_sample_freq": 4,
        #"activation_fn": get_activation_fn_name(nn.Tanh),
        "policy_kwargs": common_policy_kwargs,
        "dr_specs": {
            "body('torso').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "moodel.body('right_lower_arm').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('left_lower_arm').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('left_foot').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "body('right_foot').mass": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('left_knee').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('right_knee').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('left_elbow').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "joint('right_elbow').damping": {"uniform": [0.8, 1.2], "type": "*"},
            "geom('floor').friction[0]": {"uniform": [0.8, 1.2], "type": "*"}
        }
    }
}

# Write each environment's settings to a separate JSON file
for env_name, settings in environments.items():
    filename = f"{env_name.lower()}_rand_settings.json"
    with open(filename, 'w') as json_file:
        json.dump(settings, json_file, indent=4)
    print(f"JSON file '{filename}' created successfully.")