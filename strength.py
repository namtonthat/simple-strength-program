"""
Create strength program

Returns: outputs/program.txt
"""

import numpy as np
import yaml
import pandas as pd
from dataclasses import dataclass

import logging
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def round_weights(weight, microplates=False):
    """
    Round the weights to the nearest 2.5 kg (or kg if microplates available)
    """
    if microplates:
        pl_weight = int(np.floor(weight))
    else:
        pl_weight = round(np.ceil(weight / 2.5)) * 2.5

    return pl_weight


if __name__ == "__main__":
    LOGGER.info('Reading source data and configs')
    df_rpe = pd.read_csv('source/rpe-calculator.csv').set_index("RPE")
    rpe_schema = yaml.load(open('source/rpe.yaml', "r"), Loader=yaml.FullLoader)
    exercises = yaml.load(open('config/exercise.json', "r"), Loader = yaml.FullLoader)

    LOGGER.info('Reading user inputs')
    user_lifts = yaml.load(open('config/user_lifts.yaml', "r"), Loader = yaml.FullLoader)
    user_gym = yaml.load(open('config/user_gym.yaml', "r"), Loader = yaml.FullLoader)

    for lift in config.keys():
        if lift in ("squat", "bench", "deadlift", "overhead-press"):
            program_goal, program_type, reps, weight, rpe, rpe_pref = get_user_config(
                lift
            )
            one_rm = calculate_1rm(reps, weight, rpe)
            print(f"\n{str.capitalize(lift)} (1RM): {one_rm} kg")

            if "strength" in program_goal:
                (
                    top_sets,
                    top_reps,
                    rpe_top,
                    backoff_sets,
                    backoff_reps,
                    rpe_backoff,
                ) = get_sets_reps(config, program_goal)
                # strength section
                top_training_range = calculate_training_range(one_rm, top_reps, rpe_top)
                # backoff section
                backoff_training_range = calculate_training_range(
                    one_rm, backoff_reps, rpe_backoff
                )

                compound_program = output_training_range(
                    backoff_training_range, strength_training_range=top_training_range
                )
            else:
                # strength section
                max_sets, max_reps, rpe_max = get_sets_reps(config, program_goal)
                training_range = calculate_training_range(one_rm, max_reps, rpe_max)
                compound_program = output_training_range(
                    backoff_training_range=training_range
                )
                full_program.append(compound_program)

            accessory_program = get_accessory_lifts(one_rm, lift, config)
            full_program.append(accessory_program)

    with open('outputs/program.txt', 'w+') as f:
    f.write('\n'.join(full_program))
