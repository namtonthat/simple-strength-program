"""
Create strength program

Returns: outputs/program.txt
"""
import numpy as np
import yaml
import pandas as pd
import itertools
from typing import List, Optional
from dataclasses import dataclass

import logging

# configs
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class LiftWeight:
    """A dataclass to store the weight for a lift"""

    weight: float
    reps: int
    rpe: float

    @property
    def one_rep_max(self) -> float:
        """
        Use Brzycki's formula to calculate the theortical 1RM
        """
        one_rm_pct = df_rpe.get(f"{self.reps}").to_dict().get(self.rpe)
        one_rm = int(np.round(np.floor(self.weight / one_rm_pct), 0))
        return one_rm


@dataclass
class CompoundLift:
    """A compound exercise with sets and reps defined"""

    backoff_sets: List[LiftWeight]
    lift_name: str
    calculated_one_rm: float
    top_sets: Optional[List[LiftWeight]] = None


def calculate_1rm(reps, weight, rpe):
    """
    Use Brzycki's formula to calculate the theortical 1RM
    """
    one_rm_pct = df_rpe[f"{reps}"].to_dict().get(rpe)
    one_rm = int(np.round(np.floor(weight / one_rm_pct), 0))
    return one_rm


def round_weights(weight, microplates=False):
    """
    Round the weights to the nearest 2.5 kg (or kg if microplates available)
    """
    if microplates:
        pl_weight = int(np.floor(weight))
    else:
        pl_weight = round(np.ceil(weight / 2.5)) * 2.5

    return pl_weight


def calculate_training_range(one_rm, reps, rpe_schema) -> List[LiftWeight]:
    """
    Calculate the training day progression for a given 1RM
    """
    df_ref = df_rpe[f"{reps}"].to_dict()
    week_weights = []
    week_rpes = []

    for rpe in rpe_schema:
        week_rpe = df_ref.get(rpe)
        week_weight = week_rpe * one_rm

        week_rpes.append(week_rpe)
        week_weights.append(week_weight)

    # round the weights to the nearest 2.5kg
    week_weights = [round_weights(i, microplates) for i in week_weights]

    # iterate through training range to make sure there is enough spacing between the numbers
    training_range = seperate_training_range(week_weights, microplates)
    weights_lifted = list(zip(training_range, itertools.repeat(reps), week_rpes))
    weights_lifted = [LiftWeight(*i) for i in weights_lifted]

    return weights_lifted


def seperate_training_range(training_range, microplates):
    """
    Round the training range to the next 5kg
    """
    increment = 2.5
    if microplates:
        increment = 1

    # Define limits
    min_weight = np.floor(training_range[0] / 5) * 5
    max_weight = np.ceil(training_range[-1] / 2.5) * 2.5
    # Create even split but then increase RPE
    training_range = np.linspace(min_weight, max_weight, 5).tolist()

    # Update the values in between to ensure that the spacing is a little more accurate
    for week, weight in enumerate(training_range, 0):
        if weight != min_weight and weight != max_weight:
            training_range[week] = np.ceil(weight / increment) * increment

    return training_range


def check_weekly_difference(training_range):
    weekly_difference = np.diff(training_range).tolist()
    for weekly_diff in weekly_difference:
        if weekly_diff <= 2.5:
            return True
    return False


def get_accessory_lifts(one_rm, lift, config):
    """
    Get the accessory lifts for the user
    """
    accessory_lifts = config["accessory_lifts"][lift]
    for accessory_lift in accessory_lifts.keys():
        lift_stats = accessory_lifts[accessory_lift]
        max_pct = lift_stats["max"]
        [min_reps, max_reps] = lift_stats["reps"]
        rpe_acc = config["rpe_pref"]["accessory_lifts"]["backoff_sets"]
        acc_one_rm = one_rm * max_pct

        min_accessory_weight = calculate_training_range(acc_one_rm, max_reps, rpe_acc)
        max_accessory_weight = calculate_training_range(acc_one_rm, min_reps, rpe_acc)

        output_accessory_range(
            min_accessory_weight,
            max_accessory_weight,
            accessory_lift,
            min_reps,
            max_reps,
        )


def output_accessory_range(min_tr, max_tr, lift, min_reps, max_reps):
    """
    Output the accessory range to the user
    """
    accessory_header = f"{lift} - 2 - 3 sets @ {min_reps} - {max_reps} reps"

    accessory_program = [accessory_header]
    for week, (min_weight, max_weight) in enumerate(zip(min_tr, max_tr), 1):
        accessory = f"W{week}: {min_weight} - {max_weight} kg"
        accessory_program.append(accessory)
    print(accessory_program)
    return "\n".join(accessory_program)


def output_training_range(backoff_training_range, strength_training_range=None):
    """
    Output the training range to the console
    """
    program = []
    if strength_training_range:
        for week, (top_weight, backoff_weight) in enumerate(
            zip(strength_training_range, backoff_training_range), 1
        ):
            week_program = (
                f"Week {week} \n"
                f"{top_sets} x {top_reps}: {top_weight}\n"
                f"{backoff_sets} x {backoff_reps}: {backoff_weight}"
            )
            program.append(week_program)
    else:
        for week, weight in enumerate(backoff_training_range, 1):
            week_program = f"Week {week} \n" f"{max_sets} x {max_reps}: {weight}"
            program.append(week_program)

    print(program)
    return "\n".join(program)


if __name__ == "__main__":
    LOGGER.info("Reading source data and configs")
    df_rpe = pd.read_csv("source/rpe-calculator.csv").set_index("RPE")
    defined_rpes = yaml.load(open("source/rpe.yaml", "r"), Loader=yaml.FullLoader)
    exercises = yaml.load(open("config/exercise.json", "r"), Loader=yaml.FullLoader)

    LOGGER.info("Reading user inputs")
    user_lifts = yaml.load(open("config/user_lifts.yaml", "r"), Loader=yaml.FullLoader)
    user_gym = yaml.load(open("config/user_gym.yaml", "r"), Loader=yaml.FullLoader)

    # manual loading values
    microplates = user_gym.get("microplates")

    for lift, stats in user_lifts.items():
        # clear previous stats
        top_training_range = []
        backoff_training_range = []

        one_rm = LiftWeight(
            weight=stats.get("weight"), reps=stats.get("reps"), rpe=stats.get("rpe")
        ).one_rep_max

        goal = stats.get("program-goal")
        program_type = goal.split("-")[0]
        rpes = defined_rpes.get(program_type)
        exercise_volume = exercises.get("program-goals").get(f"{goal}")

        # print(f"\n{str.capitalize(lift)} (1RM): {one_rm} kg")
        if program_type == "strength":
            top_training_range = calculate_training_range(
                one_rm, reps=exercise_volume.get("top-reps"), rpe_schema=rpes.get("top")
            )

        backoff_training_range = calculate_training_range(
            one_rm,
            reps=exercise_volume.get("backoff-reps"),
            rpe_schema=rpes.get("backoff"),
        )

        print(
            CompoundLift(
                backoff_sets=backoff_training_range,
                lift_name=lift,
                calculated_one_rm=one_rm,
                top_sets=top_training_range,
            )
        )
        # compound_program = output_training_range(
        #     backoff_training_range
        # )
        # else:
        #     # strength section
        #     max_sets, max_reps, rpe_max = get_sets_reps(config, program_goal)
        #     training_range = calculate_training_range(one_rm, max_reps, rpe_max)
        #     compound_program = output_training_range(
        #         backoff_training_range=training_range
        #     )
        #     full_program.append(compound_program)

        # accessory_program = get_accessory_lifts(one_rm, lift, config)
        # full_program.append(accessory_program)
