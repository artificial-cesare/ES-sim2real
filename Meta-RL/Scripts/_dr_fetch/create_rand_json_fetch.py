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
    "Push": {
        "env_name": "FetchPushDense-v2",
        "rand_class": "AutoRand", # "SetParams", "Rand"
        "train_steps": 50_000_000,
        "n_envs": 16,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 3,
        "gamma": 0.995,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
        #"sde_sample_freq": 4,
        #"activation_fn": get_activation_fn_name(nn.Tanh), #nn.ReLU   ; hopper dr breaks through the suboptimal policy with relu but gets stuck with tanh; however, relu is much slower to learn and much more unstable
        "policy_kwargs": common_policy_kwargs,
        "dr_specs": {
            "body('robot0:base_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "body('robot0:torso_lift_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "body('robot0:head_pan_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "body('robot0:head_tilt_link').mass": {"uniform": [0.95, 1.05], "type": "*"},

            "joint('robot0:torso_lift_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('robot0:head_pan_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('robot0:head_tilt_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('robot0:shoulder_pan_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},

            "geom('robot0:r_gripper_finger_link').friction[0]": {"uniform": [0.95, 1.05], "type": "*"},
            "geom('robot0:l_gripper_finger_link').friction[0]": {"uniform": [0.95, 1.05], "type": "*"}, 

            "body('object0').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('object0:joint').size[0]": {"uniform": [0.95, 1.05], "type": "*"}, 
            "joint('object0:joint').size[1]": {"uniform": [0.95, 1.05], "type": "*"}
            }
    },
    "Slide": {
        "env_name": "FetchSlideDense-v2",
        "rand_class": "AutoRand",
        "train_steps": 50_000_000,
        "n_envs": 16,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 3,
        "gamma": 0.995,
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        #"sde_sample_freq": 4,
        #"activation_fn": get_activation_fn_name(nn.Tanh),
        "policy_kwargs": common_policy_kwargs,
        "dr_specs": {
            "body('robot0:base_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "body('robot0:torso_lift_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "body('robot0:head_pan_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "body('robot0:head_tilt_link').mass": {"uniform": [0.95, 1.05], "type": "*"},

            "joint('robot0:torso_lift_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('robot0:head_pan_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('robot0:head_tilt_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('robot0:shoulder_pan_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},

            "geom('robot0:r_gripper_finger_link').friction[0]": {"uniform": [0.95, 1.05], "type": "*"},
            "geom('robot0:l_gripper_finger_link').friction[0]": {"uniform": [0.95, 1.05], "type": "*"}, 

            "body('object0').mass": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('object0:joint').size[0]": {"uniform": [0.95, 1.05], "type": "*"},
            "joint('object0:joint').size[1]": {"uniform": [0.95, 1.05], "type": "*"}
            }
    }
}

# Write each environment's settings to a separate JSON file
for env_name, settings in environments.items():
    filename = f"{env_name.lower()}_rand_settings.json"
    with open(filename, 'w') as json_file:
        json.dump(settings, json_file, indent=4)
    print(f"JSON file '{filename}' created successfully.")