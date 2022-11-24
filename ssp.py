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

@dataclass
class OneRepMax:
    """A dataclass to store the one rep max for a lift"""

    lift: str
    weight: float


@dataclass
class Exercise:
    """A generic class for an exercise"""

    backoff_sets: List[LiftWeight]
    name: str
    predicted_one_rm: float
    top_sets: Optional[List[LiftWeight]] = None

    @property
    def weekly_sets(self) -> List[LiftWeight]:
        """
        Return the weekly training set for the exercise
        """
        if self.top_sets:
            weekly_tupled_sets = list(
                itertools.zip_longest(self.top_sets, self.backoff_sets)
            )
            weekly_sets = [item for sublist in weekly_tupled_sets for item in sublist]
            return weekly_sets
        else:
            return self.backoff_sets


def calculate_1rm(reps, rpe, weight):
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
    df_ref = DF_RPE[f"{reps}"].to_dict()
    week_weights = []
    week_rpes = []
    # print(df_ref)
    for rpe in rpe_schema:
        week_rpe = df_ref.get(rpe)
        week_weight = week_rpe * one_rm
        # print(week_rpe)
        # print(week_weight)
        week_rpes.append(week_rpe)
        week_weights.append(week_weight)

    # round the weights to the nearest 2.5kg
    training_range = [round_weights(i) for i in week_weights]

    # iterate through training range to make sure there is enough spacing between the numbers
    training_range = seperate_training_range(training_range)

    weights_lifted = list(zip(training_range, itertools.repeat(reps), week_rpes))
    weights_lifted = [LiftWeight(*i) for i in weights_lifted]

    return weights_lifted


def seperate_training_range(training_range):
    """
    Update the training values to ensure
    that there is enough separation between each week
    """
    increment = 2.5
    if MICROPLATES:
        increment = 1

    # Define limits
    min_weight = np.floor(training_range[0] / 10) * 10
    max_weight = np.ceil(training_range[-1] / 5) * 5

    # Create even split but then increase RPE
    training_range = np.linspace(min_weight, max_weight, len(training_range)).tolist()

    for week, weight in enumerate(training_range, 0):
        if weight != min_weight and weight != max_weight:
            training_range[week] = np.ceil(weight / increment) * increment

    return training_range


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


if __name__ == "__main__":
    LOGGER.info("Reading source data and configs")
    defined_rpes = yaml.load(open("source/rpe.yaml", "r"), Loader=yaml.FullLoader)
    exercises = yaml.load(open("config/exercise.json", "r"), Loader=yaml.FullLoader)

    LOGGER.info("Reading user inputs")
    user_lifts = yaml.load(open("config/user_lifts.yaml", "r"), Loader=yaml.FullLoader)
    user_gym = yaml.load(open("config/user_gym.yaml", "r"), Loader=yaml.FullLoader)

    # Global variables
    DF_RPE = pd.read_csv("source/rpe-calculator.csv").set_index("RPE")
    MICROPLATES = user_gym.get("microplates")

    max_lifts = []
    full_program = []
    for lift, stats in user_lifts.items():
        # clear previous stats
        top_training_range = []
        backoff_training_range = []

        one_rm = calculate_1rm(
            reps=stats.get("reps"), rpe=stats.get("rpe"), weight=stats.get("weight")
        )

        max_strength = OneRepMax(
            lift=lift
            weight=one_rm
        )

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

        compound_lift = Exercise(
                backoff_sets=backoff_training_range,
                name=lift,
                predicted_one_rm=one_rm,
                top_sets=top_training_range,
            )


        # save it to pretty print
        max_lifts.append(max_strength)
        full_program.append(compound_lift)

    # output training program
