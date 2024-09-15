dr_specs_robot0 = {
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
    "joint('object0:joint').size[0]": {"uniform": [0.95, 1.05], "type": "*"}
}

"""
"body('robot0:shoulder_pan_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:shoulder_lift_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:upperarm_roll_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:elbow_flex_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:forearm_roll_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:wrist_flex_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:wrist_roll_link').mass": {"uniform": [0.95, 1.05], "type": "*"},
"body('robot0:gripper_link').mass": {"uniform": [0.95, 1.05], "type": "*"},


"joint('robot0:shoulder_lift_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
"joint('robot0:elbow_flex_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
"joint('robot0:forearm_roll_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},
"joint('robot0:wrist_flex_joint').damping": {"uniform": [0.95, 1.05], "type": "*"},


"""