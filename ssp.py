"""
Create strength program

Requires:
* user profile configured within `profiles/<user>/lifts.yaml`

Returns: outputs/<user>/strength_program-<yyyy-mm>.csv

Usage:
    ssp.py [-u] [-v]

Options:
    -u --user           User profile to use [default: default]
    -v --verbose        Verbose logging
    -h --help           Show this screen
"""
import numpy as np
import yaml
from docopt import docopt
import pandas as pd
import itertools
from typing import List, Optional
from dataclasses import dataclass
from rich.prompt import Prompt
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
    pct_one_rm: float


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
    related_compound: str = None
    top_sets: Optional[List[LiftWeight]] = None

    def __post_init__(self):
        """Clean the names / related compound names"""
        exercise_name = self.name.replace("-", " ").title()
        self.name = exercise_name

        if self.related_compound:
            related_compound_name = self.related_compound.replace("-", " ").title()
            self.related_compound = related_compound_name

    @property
    def weekly_sets(self) -> List[LiftWeight]:
        """
        Return the weekly training set for the exercise
        """
        if self.top_sets:
            weekly_sets = list(itertools.zip_longest(self.top_sets, self.backoff_sets))
            return weekly_sets
        else:
            return self.backoff_sets

    @property
    def output_name(self) -> str:
        """
        Return the name of the exercise for the output
        """
        if self.related_compound:
            return f"{self.name} ({self.related_compound})"
        else:
            return self.name

    @property
    def output_print(self) -> str:
        """
        Return the string to output to the user
        """

        outputs = [f"{self.output_name}\n" f"Training Max: {self.predicted_one_rm} kg"]
        if self.top_sets:
            for week, (top, backoff) in enumerate(self.weekly_sets, 1):
                output = (
                    f"W{week}: {top.weight} kg x {top.reps}\n"
                    + f"W{week}: {backoff.weight} kg x {backoff.reps}"
                )
                outputs.append(output)
        else:
            for week, weight in enumerate(self.weekly_sets, 1):
                output = f"W{week}: {weight.weight} kg x {weight.reps}"
                outputs.append(output)
        return "\n".join(outputs)


def calculate_1rm(reps, rpe, weight):
    """
    Use Brzycki's formula to calculate the theortical 1RM
    """
    one_rm_pct = DF_RPE[f"{reps}"].to_dict().get(rpe)
    one_rm = int(np.round(np.floor(weight / one_rm_pct), 0))
    return one_rm


def round_weights(weight):
    """
    Round the weights to the nearest 2.5 kg (or kg if microplates available)
    """
    if MICROPLATES:
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
    pct_one_rms = []

    for rpe in rpe_schema:
        pct_one_rm = df_ref.get(rpe)
        week_weight = pct_one_rm * one_rm
        pct_one_rms.append(pct_one_rm)
        week_weights.append(week_weight)

    # round the weights to the nearest 2.5kg
    training_range = [round_weights(i) for i in week_weights]

    # iterate through training range to make sure there is enough spacing between the numbers
    if has_weekly_difference(training_range):
        LOGGER.info("Weekly difference is too small, recalculating")
        lowest_rpe = rpe_schema[0] - 0.5
        highest_rpe = rpe_schema[-1] + 0.5
        updated_rpe_schema = [lowest_rpe] + rpe_schema[1:4] + [highest_rpe]
        training_range = calculate_training_range(one_rm, reps, updated_rpe_schema)

    weights_lifted = list(zip(training_range, itertools.repeat(reps), pct_one_rms))
    weights_lifted = [LiftWeight(*i) for i in weights_lifted]

    return weights_lifted


def has_weekly_difference(training_range):
    weekly_difference = np.diff(training_range).tolist()
    for weekly_diff in weekly_difference:
        if weekly_diff <= 2.5:
            return False
    return True


if __name__ == "__main__":
    arguments = docopt(__doc__, version="simple-strength-program v1.0")

    # gather inputs
    MICROPLATES = False

    if arguments["--user"]:
        # ask for user profile
        user_profile = Prompt.ask("Enter user profile name", default="default")
        user_profile = user_profile.lower()
        user_profile = user_profile.replace(" ", "-")
    else:
        user_profile = "default"

    LOGGER.info("Reading source data and configs")
    defined_rpes = yaml.load(open("source/rpe.yaml", "r"), Loader=yaml.FullLoader)
    exercises = yaml.load(open("config/exercise.json", "r"), Loader=yaml.FullLoader)
    accessory_rpe_schema = defined_rpes.get("accessory").get("backoff")

    LOGGER.info("Reading user inputs")
    user_lifts = yaml.load(
        open(f"profiles/{user_profile}/lifts.yaml", "r"), Loader=yaml.FullLoader
    )
    try:
        user_gym = yaml.load(
            open(f"profiles/{user_profile}gym.yaml", "r"), Loader=yaml.FullLoader
        )
    except FileNotFoundError:
        user_gym = False

    MICROPLATES = user_gym.get("microplates") if user_gym else False

    # Global variables
    DF_RPE = pd.read_csv("source/rpe-calculator.csv").set_index("RPE")

    max_lifts = []
    full_program = []
    accessory_program = []

    for compound_lift, stats in user_lifts.items():
        # clear previous stats
        top_training_range = []
        backoff_training_range = []

        lift_one_rep_max = calculate_1rm(
            reps=stats.get("reps"), rpe=stats.get("rpe"), weight=stats.get("weight")
        )

        max_strength = OneRepMax(lift=compound_lift, weight=lift_one_rep_max)

        goal = stats.get("program-goal")
        program_type = goal.split("-")[0]
        rpes = defined_rpes.get(program_type)
        exercise_volume = exercises.get("program-goals").get(f"{goal}")

        if program_type == "strength":
            top_training_range = calculate_training_range(
                lift_one_rep_max,
                reps=exercise_volume.get("top-reps"),
                rpe_schema=rpes.get("top"),
            )

        backoff_training_range = calculate_training_range(
            lift_one_rep_max,
            reps=exercise_volume.get("backoff-reps"),
            rpe_schema=rpes.get("backoff"),
        )

        compound_lift = Exercise(
            backoff_sets=backoff_training_range,
            name=compound_lift,
            predicted_one_rm=lift_one_rep_max,
            top_sets=top_training_range,
        )

        accessories = exercises.get("accessory-lifts").get(f"{compound_lift}")
        for accessory, stats in accessories.items():
            accessory_one_rm = round_weights(
                lift_one_rep_max * stats.get("max-onerm-pct")
            )
            min_reps = stats.get("min-reps")
            max_reps = stats.get("max-reps")

            low_volume = calculate_training_range(
                one_rm=accessory_one_rm, reps=min_reps, rpe_schema=accessory_rpe_schema
            )

            high_volume = calculate_training_range(
                one_rm=lift_one_rep_max, reps=max_reps, rpe_schema=accessory_rpe_schema
            )

            low_volume_exercise = Exercise(
                backoff_sets=low_volume,
                name=accessory,
                predicted_one_rm=accessory_one_rm,
                related_compound=compound_lift,
            )

            high_volume_exercise = Exercise(
                backoff_sets=high_volume,
                name=accessory,
                predicted_one_rm=lift_one_rep_max,
                related_compound=compound_lift,
            )

            accessory_program.append(low_volume_exercise)
            accessory_program.append(high_volume_exercise)

    # output training program
