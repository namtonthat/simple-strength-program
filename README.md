A simple Python script to automate the creation of a powerlifting strength program.

## How to use
1. Clone repo into local directory and navigate to repo.
2. Install packages required by `requirements.txt`.
3. Update `user.json` to reflect your current status.

**NOTE**:
- Refer to `exercise_pref` within `exercise.json` to select a goal i.e. the lower the `x` number is in `(strength|volume)-x`; the higher the amount of volume but also working less at the higher weight values.
- You can set different goals for each lift - if in doubt, there are also `basic` programs to help you get started with strength / volume building
4. Run `python3 ssp.py` to output your next 5 weeks of training.


### Volume
- Refers to only doing straight sets with reps within the 6 + rep range.

### Strength
- Refers to doing top sets and then followed by back off sets, typically 90% of your top set but it is configured off the RPE scheme defined.
