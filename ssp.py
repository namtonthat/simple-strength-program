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

def calculate_training_range(one_rm, reps, microplates):
    """
    Calculate the training day progression for a given 1RM
    """
    df_ref = df_rpe[f'{reps}'].to_dict()
    w1 = df_ref.get(6) * one_rm
    w2 = df_ref.get(7.5) * one_rm
    w3 = df_ref.get(8) * one_rm
    w4 = df_ref.get(8.5) * one_rm
    w5 = df_ref.get(9.5) * one_rm

    training_weight = [w1, w2, w3, w4, w5]
    training_range = [round_weights(i, microplates) for i in training_weight]

    return training_range



def calculate_scheme(one_rm, reps, backoff = False):
    """
    Calculate the number of reps and weight given a one_rm and exercise type
    :param one_rm: one rep max for that given training session
    :param reps: number of reps for that given training session
    :param backoff: if true, calculate the backoff scheme for that given training session
    """
    # wenders_one_rm =

    if backoff:
        sets = 3
    else:
        sets = 1

    scheme = { "weight": one_rm, "reps": reps, "sets": sets }
    return


if __name__ == '__main__':
    df_rpe = pd.read_csv('source/rpe-calculator.csv').set_index('RPE')


    config = yaml.load(open('config/current_status.json', "r"), Loader = yaml.FullLoader)

    microplates = config["microplates"]
    program_goal = config["program_goal"]
    exercise_pref = config["exercise_preference"].get(program_goal)

    for lift in config.keys():
        if lift in ("squats", "bench", "deadlifts"):
            reps = config[lift]["reps"]
            weight = config[lift]["weight"]
            rpe = config[lift]["rpe"]
            one_rm = calculate_1rm(reps, weight, rpe)
            print(f"\n{str.capitalize(lift)} (1RM): {one_rm} kg")

            if 'strength' in program_goal:
                top_sets = exercise_pref.get('top_sets')
                top_reps = exercise_pref.get('top_reps')
                backoff_sets =  exercise_pref.get('backoff_sets')
                backoff_reps =  exercise_pref.get('backoff_reps')

                # strength section
                print(f'Top set: {top_sets} sets of {top_reps} reps')
                training_range = calculate_training_range(one_rm, top_reps, microplates)
                print(f"{training_range}")


                # backoff section
                print(f'Backoffs: {backoff_sets} sets of {backoff_reps} reps')
                training_range = calculate_training_range(one_rm, backoff_reps, microplates)
                print(f"{training_range}")

            else:
            # strength section
                max_sets = exercise_pref.get('max_sets')
                max_reps = exercise_pref.get('max_reps')
                print(f'{max_sets} sets of {max_reps} reps')
                training_range = calculate_training_range(one_rm, max_reps, microplates)
                print(f"{training_range}")