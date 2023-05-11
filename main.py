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
import math
import os
import yaml
from docopt import docopt
import pandas as pd
import itertools
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from rich.prompt import Prompt
import logging
from prettytable import PrettyTable

# configs
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class LiftWeight:
    """A dataclass to store the weight for a lift"""

    weight: float
    sets: int
    reps: int

    @property
    def print_as_tuple(self) -> tuple:
        """
        Return a tuple for the lift
        """
        return (f"{self.sets} x {self.reps}", f"{self.weight}")


@dataclass
class OneRepMax:
    """A dataclass to store the one rep max for a lift"""

    lift: str
    weight: float

    @property
    def output_table(self) -> str:
        "Creates table output for prettytable package"
        output = [self.lift.title(), self.weight]
        return output


@dataclass
class Exercise:
    """A generic class for an exercise"""

    backoff_sets: List[LiftWeight]
    name: str
    predicted_one_rm: float = None
    related_compound: str = None
    top_sets: Optional[List[LiftWeight]] = None

    @property
    def pretty_name(self) -> str:
        """Clean the names / related compound names"""
        exercise_name = self.name.replace("-", " ").title()
        return exercise_name

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
    def make_weekly_sets(self) -> List[tuple]:
        """Create weekly sets and reps for output table"""
        weekly = []
        if self.top_sets:
            for _, (top, backoff) in enumerate(self.weekly_sets, 1):
                top_sets = top.print_as_tuple
                backoff_sets = backoff.print_as_tuple
                weekly.extend([top_sets, backoff_sets])
        else:
            for _, backoff in enumerate(self.weekly_sets, 1):
                backoff_sets = backoff.print_as_tuple
                weekly.extend([backoff_sets])
        return weekly

    @property
    def output_table_with_padding(self) -> List[tuple]:
        """
        Add empty row for formatting
        Returns two lists, one for sets and reps, one for weights
        """
        week_sets_reps = ["Prescribed Work"]
        week_weights = [self.pretty_name]

        empty_entries = [("", "")] * MESOCYCLE_LENGTH
        weekly_sets = self.make_weekly_sets
        weekly_sets = itertools.zip_longest(empty_entries, weekly_sets)
        weekly_flattened = list(itertools.chain.from_iterable(weekly_sets))
        week_sets_rep, week_weight = map(list, zip(*weekly_flattened))

        # formatting
        week_sets_reps.extend(week_sets_rep)
        week_sets_reps.append("")
        week_weights.extend(week_weight)
        week_weights.append("")
        return week_sets_reps, week_weights

    @property
    def output_table(self) -> List:
        """
        Returns two lists, one for sets and reps, one for weights
        """
        week_sets_reps = ["Prescribed Work"]
        week_weights = [self.pretty_name]

        weekly_sets = self.make_weekly_sets
        week_sets_rep, week_weight = map(list, zip(*weekly_sets))

        # formatting
        week_sets_reps.extend(week_sets_rep)
        week_sets_reps.append("")
        week_weights.extend(week_weight)
        week_weights.append("")

        return week_sets_reps, week_weights


def create_weekly_dates(start_date: datetime.date):
    """
    Create the weekly dates for the program
    """
    starting_monday = datetime.date(start_date + timedelta(days=7 - start_date.weekday()))

    week_mondays = []
    for week in range(1, 6):
        week_monday = starting_monday + timedelta(days=7 * week)
        week_mondays.append(week_monday)

    # Create the week dates as str
    week_dates = [i.strftime("%d/%m/%y") for i in week_mondays]
    return week_dates


def create_program_dates(start_date: datetime.date):
    """
    Create the weekly dates for the program output
    """
    single_week = create_weekly_dates(start_date)
    single_week.append("")
    exercise_dates = []
    first_week = list(itertools.chain.from_iterable(zip(*[single_week] * 2)))
    first_week.pop(-1)
    exercise_dates = [first_week] + [["Week Beginning"] + single_week] * (MAX_DAILY_EXERCISES - 1)

    exercise_dates = [item for sublist in exercise_dates for item in sublist]
    exercise_dates.insert(0, "Week Beginning")

    return exercise_dates


def calculate_1rm(reps, rpe, weight):
    """
    Use Brzycki's formula to calculate the theortical 1RM
    """
    one_rm_pct = DF_RPE[f"{reps}"].to_dict().get(rpe)
    one_rm = int(np.round(np.floor(weight / one_rm_pct), 0))
    return one_rm


def round_weights(weight, deload=False):
    """
    Round the weights to the nearest 2.5 kg (or kg if microplates available)
    """
    if deload:
        pl_weight = MIN_PLATE_WEIGHT * math.floor(weight / MIN_PLATE_WEIGHT)
    else:
        pl_weight = MIN_PLATE_WEIGHT * math.ceil(weight / MIN_PLATE_WEIGHT)

    return pl_weight


def get_one_rm_pct(reps, rpe):
    "Get the percentage of 1RM for a given number of reps and RPE"
    one_rm_pct = DF_RPE[f"{reps}"].to_dict().get(rpe)
    return one_rm_pct


def calculate_training_range(one_rm, reps, rpe_schema) -> List[float]:
    """
    Calculate the training day progression for a given 1RM
    """
    lowest_rpe = rpe_schema[0]
    highest_rpe = rpe_schema[-1]

    w1_weight = round_weights(get_one_rm_pct(reps, lowest_rpe) * one_rm, deload=True)
    w5_weight = get_one_rm_pct(reps, highest_rpe) * one_rm

    week_weights = np.linspace(w1_weight, w5_weight, MESOCYCLE_LENGTH).tolist()

    training_range = [round_weights(i) for i in week_weights]

    # check if there is minimum spacing between each week

    return training_range


def create_training_range(one_rm, sets, reps, rpe_schema) -> List[LiftWeight]:
    """Create a list of LiftWeight objects for the training range"""
    training_range = calculate_training_range(one_rm, reps, rpe_schema)

    weights_lifted = list(zip(training_range, itertools.repeat(sets), itertools.repeat(reps)))
    weights_lifted = [LiftWeight(*i) for i in weights_lifted]

    return weights_lifted


def column_pad(*columns):
    """Pad the columns to the same length"""
    max_len = max([len(c) for c in columns])
    for c in columns:
        c.extend([""] * (max_len - len(c)))


def create_empty_column_with_header(column_to_create: str) -> list:
    """Create an empty column with a header for prettytable package"""
    empty_entries = [""] * 6
    # first exercise will have double number of entries because of top and backoff sets
    first_exercise = [f"{column_to_create}"] + empty_entries * 2
    first_exercise.pop(-1)

    # every subsequent exercise does not have top sets
    single_col = [f"{column_to_create}"] + empty_entries
    single_cols = first_exercise + single_col * (MAX_DAILY_EXERCISES - 1)

    return single_cols


if __name__ == "__main__":
    arguments = docopt(__doc__, version="simple-strength-program v1.0")

    # intiates variables
    MICROPLATES = False
    START_DATE = datetime.now()
    MESOCYCLE_LENGTH = 5

    if arguments["--user"]:
        # ask for user profile
        user_profile = Prompt.ask("Enter user profile name", default="default")
        user_profile = user_profile.lower()
        user_profile = user_profile.replace(" ", "-")
    else:
        user_profile = "default"

    # available users
    home_directory = os.
    available_users = os.listdir("/profiles")

    LOGGER.info("Reading source data and configs")
    defined_rpes = yaml.load(open("source/rpe.yaml", "r"), Loader=yaml.FullLoader)
    exercises = yaml.load(open("config/exercise.json", "r"), Loader=yaml.FullLoader)
    program_layout = yaml.load(open("config/program-layout.yaml", "r"), Loader=yaml.FullLoader)
    accessory_rpe_schema = defined_rpes.get("accessory").get("backoff")

    LOGGER.info("Reading user inputs")
    user_lifts = yaml.load(open(f"profiles/{user_profile}/lifts.yaml", "r"), Loader=yaml.FullLoader)
    try:
        user_gym = yaml.load(open(f"profiles/{user_profile}/gym.yaml", "r"), Loader=yaml.FullLoader)
    except FileNotFoundError:
        logging.error("user not found; either alter the default profile or create a new one")
        user_gym = False

    # Global variables
    DF_RPE = pd.read_csv("source/rpe-calculator.csv").set_index("RPE")
    MICROPLATES = user_gym.get("microplates") if user_gym else False
    MIN_PLATE_WEIGHT = 1 if MICROPLATES else 2.5
    ACCESSORY_SETS = exercises.get("accessory-config").get("max-sets")
    MAX_DAILY_EXERCISES = len(max(list(program_layout.values()), key=len))

    max_lifts = []
    full_program = []

    for compound_lift_name, stats in user_lifts.items():
        # clear previous stats
        LOGGER.info("Creating program for %s", compound_lift_name)
        top_training_range = []
        backoff_training_range = []

        lift_one_rep_max = calculate_1rm(
            reps=stats.get("reps"), rpe=stats.get("rpe"), weight=stats.get("weight")
        )

        max_strength = OneRepMax(lift=compound_lift_name, weight=lift_one_rep_max)

        goal = stats.get("program-goal")
        program_type = goal.split("-")[0]
        rpes = defined_rpes.get(program_type)
        exercise_volume = exercises.get("program-goals").get(f"{goal}")

        if program_type == "strength":
            top_training_range = create_training_range(
                one_rm=lift_one_rep_max,
                sets=exercise_volume.get("top-sets"),
                reps=exercise_volume.get("top-reps"),
                rpe_schema=rpes.get("top"),
            )

        backoff_training_range = create_training_range(
            one_rm=lift_one_rep_max,
            sets=exercise_volume.get("backoff-sets"),
            reps=exercise_volume.get("backoff-reps"),
            rpe_schema=rpes.get("backoff"),
        )

        compound_lift = Exercise(
            backoff_sets=backoff_training_range,
            name=compound_lift_name,
            predicted_one_rm=lift_one_rep_max,
            top_sets=top_training_range,
        )

        full_program.append(compound_lift)

        accessories = exercises.get("accessory-lifts").get(f"{compound_lift_name}")
        for accessory, stats in accessories.items():
            accessory_one_rm = round_weights(lift_one_rep_max * stats.get("max-onerm-pct"))
            min_reps = stats.get("min-reps")

            accessory_training_range = create_training_range(
                one_rm=accessory_one_rm,
                sets=ACCESSORY_SETS,
                reps=min_reps,
                rpe_schema=accessory_rpe_schema,
            )

            accessory_lift = Exercise(
                backoff_sets=accessory_training_range,
                name=accessory,
                predicted_one_rm=accessory_one_rm,
                related_compound=compound_lift,
            )

            full_program.append(accessory_lift)

        max_lifts.append(max_strength)
    # create the program
    LOGGER.info("Creating list of available exercises to output")
    all_exercises = {}
    for exercise in full_program:
        all_exercises[exercise.name] = exercise

    user_stats = PrettyTable()
    user_stats.field_names = ["Lift", "Training Max"]
    for lift in max_lifts:
        max_tm_row = lift.output_table
        user_stats.add_row(max_tm_row)

    # Creating both the prescribed work and weight columns
    weight_cols = []
    sets_reps_cols = []

    for day_no, exercises in program_layout.items():
        day_weight = []
        day_sets_reps = []

        for exercise_no, exercise_name in enumerate(exercises, 1):
            planned_exercise = all_exercises.get(exercise_name)
            if planned_exercise:
                sets_reps_col, weight_col = planned_exercise.output_table
            else:
                LOGGER.info('No exercise found for "%s", adding placeholder', exercise_name)
                planned_exercise = Exercise(
                    backoff_sets=[LiftWeight(0, ACCESSORY_SETS, 12)] * MESOCYCLE_LENGTH,
                    name=exercise_name,
                )
                sets_reps_col, weight_col = planned_exercise.output_table

            has_top_set = planned_exercise.top_sets
            if (exercise_no == 1) & (has_top_set == []):
                LOGGER.info("First exercise has no top set, adding placeholder")
                sets_reps_col, weight_col = planned_exercise.output_table_with_padding
            day_sets_reps.append(sets_reps_col)
            day_weight.append(weight_col)

        flatten_weights_col = [item for sublist in day_weight for item in sublist]
        flatten_sets_reps_col = [item for sublist in day_sets_reps for item in sublist]

        sets_reps_cols.append(flatten_sets_reps_col)
        weight_cols.append(flatten_weights_col)

    column_pad(*sets_reps_cols)
    column_pad(*weight_cols)

    # add actual load columns for user input
    LOGGER.info("Creating columns")
    program_dates = create_program_dates(START_DATE)

    user_input_columns = ["Actual Load", "RPE", "Notes", ""]

    program = PrettyTable()
    program.add_column("", program_dates)
    for day, (sets_reps_col, weight_col) in enumerate(zip(sets_reps_cols, weight_cols), 1):
        program.add_column(f"Day {day}", sets_reps_col)
        program.add_column("", weight_col)

        # iterate and add empty columns with headers
        for user_input_column_name in user_input_columns:
            user_input_col = create_empty_column_with_header(user_input_column_name)
            program.add_column("", user_input_col)

    LOGGER.info("Writing to file")
    program_date = START_DATE.strftime("%Y-%m")
    output_path = f"output/{user_profile}"
    program_file = f"program-{program_date}.csv"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f"{output_path}/{program_file}", "w+") as f:
        f.write(user_stats.get_csv_string())
        f.write("\n")

        f.write(program.get_csv_string())
