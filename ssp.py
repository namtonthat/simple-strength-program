import numpy as np
import yaml
import pandas as pd

def calculate_1rm(reps, weight, rpe):
    """
    Use Brzycki's formula to calculate the theortical 1RM
    """
    one_rm_pct = df_rpe[f'{reps}'].to_dict().get(rpe)
    one_rm = int(np.round(np.floor(weight / one_rm_pct), 0))
    return one_rm

def round_weights(weight, microplates = False):
    """
    Round the weights to the nearest 2.5 kg (or kg if microplates available)
    """
    if microplates:
        pl_weight = int(np.floor(weight))
    else:
        pl_weight = round(np.ceil(weight / 2.5)) * 2.5

    return pl_weight

def calculate_training_range(one_rm, reps, rpe_schema):
    """
    Calculate the training day progression for a given 1RM
    """
    df_ref = df_rpe[f'{reps}'].to_dict()
    microplates = config["microplates"]

    training_weight = []
    for rpe in rpe_schema:
        week_weight = df_ref.get(rpe) * one_rm
        training_weight.append(week_weight)

    training_range  = [round_weights(i, microplates) for i in training_weight]

    # iterate through training range to make sure there is enough spacing between the numbers
    if check_weekly_difference(training_range):
        training_range = seperate_training_range(training_range, microplates)

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
    max_weight = np.ceil(training_range[-1]/ 2.5 ) * 2.5
    # Create even split but then increase RPE
    training_range = np.linspace(min_weight, max_weight, 5).tolist()

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

def get_sets_reps(config, program_goal):
    """
    Get the number of sets and reps for a given program goal
    Strength goals are top sets and followed by backoffs
    whereas volume sets do not contain top sets
    """
    program_schema = config["exercise_pref"][program_goal]
    program_type = program_goal.split('-')[0]
    strength_list = []

    if program_type == "strength":
        top_sets = program_schema["top_sets"]
        top_reps = program_schema["top_reps"]
        rpe_top = config["rpe_pref"][program_type]["top_sets"]
        strength_list = [top_sets, top_reps, rpe_top]

    backoff_sets = program_schema["backoff_sets"]
    backoff_reps = program_schema["backoff_reps"]
    rpe_backoff  = config["rpe_pref"][program_type]["backoff_sets"]
    backoff_list = [backoff_sets, backoff_reps, rpe_backoff]

    strength_list.extend(backoff_list)
    return strength_list

def get_user_config(lift):
    """
    Return variables assigned from dictionary
    """
    program_goal = config[lift]["program_goal"]
    program_type = program_goal.split('-')[0]
    reps = config[lift]["reps"]
    weight = config[lift]["weight"]
    rpe = config[lift]["rpe"]
    rpe_pref = config["rpe_pref"][program_type]

    return program_goal, program_type, reps, weight, rpe, rpe_pref

def get_accessory_lifts(one_rm, lift, top_reps, config):
    """
    Get the accessory lifts for the user
    """
    accessory_lifts = config["accessory_lifts"][lift]
    for accessory_lift in accessory_lifts.keys():
        lift_stats = accessory_lifts[accessory_lift]
        max_pct = lift_stats['max']
        [min_reps, max_reps] = lift_stats['reps']
        acc_one_rm = one_rm * max_pct

        min_accessory_weight = calculate_training_range(acc_one_rm, max_reps, rpe_backoff)
        max_accessory_weight = calculate_training_range(acc_one_rm, min_reps, rpe_backoff)
        return min_accessory_weight, max_accessory_weight

def output_training_range(training_range, backoff_training_range=None):
    """
    Output the training range to the console
    """
    if backoff_training_range is not None:
        for week, (top_weight, backoff_weight) in enumerate(zip(training_range, backoff_training_range), 1):
            print(f'Week {week} \n'
                f'{top_sets} x {top_reps}: {top_weight}\n'
                f'{backoff_sets} x {backoff_reps}: {backoff_weight}')
    else:
        for week, weight in enumerate(training_range, 1):
            print(f'Week {week} \n'
                f'{backoff_sets} x {backoff_reps}: {weight}')

if __name__ == '__main__':
    df_rpe = pd.read_csv('source/rpe-calculator.csv').set_index('RPE')
    user_config = yaml.load(open('config/user.yaml', "r"), Loader = yaml.FullLoader)
    exercise_config = yaml.load(open('config/exercise.json', "r"), Loader = yaml.FullLoader)

    config = {**user_config, **exercise_config}

    for lift in config.keys():
        if lift in ("squat", "bench", "deadlift", "overhead-press"):
            program_goal, program_type, reps, weight, rpe, rpe_pref  = get_user_config(lift)

            one_rm = calculate_1rm(reps, weight, rpe)
            print(f"\n{str.capitalize(lift)} (1RM): {one_rm} kg")

            if 'strength' in program_goal:
                top_sets, top_reps, rpe_top, backoff_sets, backoff_reps, rpe_backoff = get_sets_reps(config, program_goal)
                # strength section
                top_training_range = calculate_training_range(one_rm, top_reps, rpe_top)
                # backoff section
                backoff_training_range = calculate_training_range(one_rm, backoff_reps, rpe_backoff)

                acc_training_range = get_accessory_lifts(one_rm, lift, top_reps, config)
                output_training_range(top_training_range, backoff_training_range)
            else:
                # volume section
                max_sets, max_reps, rpe_max = get_sets_reps(config, program_goal)
                training_range = calculate_training_range(one_rm, max_reps, rpe_max)
                output_training_range(training_range)