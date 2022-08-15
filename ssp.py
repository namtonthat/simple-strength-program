import numpy as np
import yaml
import pandas as pd
# from ics import Calendar, Event

def calculate_1rm(reps, weight, rpe):
    """
    Use Brzycki's formula to calculate the theortical 1RM
    """
    one_rm_pct = df_rpe[f'{reps}'].to_dict().get(rpe)
    one_rm = round_weights(weight / one_rm_pct)
    # one_rm = int(weight / (1.0278 - (0.0278 * reps)))
    return one_rm

def round_weights(weight, microplates = False):
    """
    Round the weights to the nearest 2.5 kg (or kg if microplates available)
    """
    if microplates:
        pl_weight = int(np.floor(weight))
    else:
        pl_weight = round(np.floor(weight / 2.5)) * 2.5

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
    if len(set(training_range)) < len(training_range):
        training_range = seperate_training_range(training_range)

    return training_range

def seperate_training_range(training_range):
    """
    Ensures there's enough gap between the weekly values to gain strength
    by setting the highest value the maximum limit and then calculating
    even spacing between the values
    """
    max = training_range[-1]
    min = 0.875 * max
    new_training_range = np.linspace(min, max, len(training_range))
    new_training_range = [round_weights(i) for i in new_training_range]
    return new_training_range

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

    strength_list.extend([backoff_sets, backoff_reps, rpe_backoff])
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
                f'{max_sets} x {max_reps}: {weight}'
            )

def get_accessory_lifts(one_rm, lift, backoff_reps, config):
    """
    Get the accessory lifts for the user
    """
    accessory_lifts = config["accessory_lifts"][lift]
    accessory_reps = 8
    for accessory_lift in accessory_lifts.keys():
        # convert min and max percentages into RPE values as a percentage of 1RM
        [min_rpe, max_rpe] = accessory_lifts[accessory_lift]
        accessory_rpe = np.linspace(min_rpe, max_rpe, 5).tolist()
        accessory_rpe = [round(i * 20) /2 for i in accessory_rpe]
        print(accessory_reps)
        accessory_weight = calculate_training_range(one_rm, accessory_reps, accessory_rpe)
        print(f"{accessory_lift}: {accessory_weight}")

if __name__ == '__main__':
    df_rpe = pd.read_csv('source/rpe-calculator.csv').set_index('RPE')
    user_config = yaml.load(open('config/user.json', "r"), Loader = yaml.FullLoader)
    exercise_config = yaml.load(open('config/exercise.json', "r"), Loader = yaml.FullLoader)

    config = {**user_config, **exercise_config}

    for lift in config.keys():
        if lift in ("squat", "bench", "deadlift"):
            program_goal, program_type, reps, weight, rpe, rpe_pref  = get_user_config(lift)

            one_rm = calculate_1rm(reps, weight, rpe)
            print(f"\n{str.capitalize(lift)} (1RM): {one_rm} kg")

            if 'strength' in program_goal:
                top_sets, top_reps, rpe_top, backoff_sets, backoff_reps, rpe_backoff = get_sets_reps(config, program_goal)
                # strength section
                top_training_range = calculate_training_range(one_rm, top_reps, rpe_top)
                # backoff section
                backoff_training_range = calculate_training_range(one_rm, backoff_reps, rpe_backoff)

                get_accessory_lifts(one_rm, lift, backoff_reps, config)
                output_training_range(top_training_range, backoff_training_range)
            else:
            # strength section
                max_sets, max_reps, rpe_max = get_sets_reps(config, program_goal)
                print(f'{max_sets} sets of {max_reps} reps')
                training_range = calculate_training_range(one_rm, max_reps, rpe_max)
                output_training_range(training_range)