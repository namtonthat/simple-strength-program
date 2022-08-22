A simple Python script to automate the creation of a powerlifting mesocycle.

## 🏁 Getting Started
1. Clone repo into local directory and navigate to repo.
2. Install packages required by `requirements.txt`.
3. Update `user.json` to reflect your current status (see `NOTE` below).
4. Run `python3 ssp.py` to output your next 5 weeks of training.

**NOTE**:
- Refer to `exercise_pref` within `exercise.json` to select a goal i.e. the lower the `x` number is in `(strength|volume)-x`; the higher the amount of volume but also working less at the higher weight values.
- You can set different goals for each lift - if in doubt, there are also `basic` programs to help you get started with strength / volume building

### 👩‍💻 Basic Programming
| Type                   | Day 1     | Day 2        | Day 3          | Day 4            |
| ---------------------- | --------- | ------------ | -------------- | ---------------- |
| **Compound**           | Squat     | Bench        | Overhead Press | Deadlift         |
| **Accessory Compound** | Bench     | Deadlift     | Squat          | Bench            |
| Assisted Compound      | Hamstring | Lats / Delts | Lats / Delts   | Lats / Delts     |
| Accessory              | Quads     | Legs / Quads | Chest          | Triceps / Biceps |
| Accessory              | Core      | Triceps      | Biceps + Core  | Triceps          |

- Highlighted in bold is the *bare minimum required* to be completed in each mesocycle.

#### 📕 Definitions
- **Compound** - one of the main movements - defined as the squat, bench, overhead press or deadlift
- **Accessory Compound** - accessory movements to support the main compound lifts - as defined in the `exercise.json` config
  - **Squat**:
    - Safety Squat Bar
    - Front Squat
    - Paused Squats
    - High Bar Squats

  - **Bench**
    - 2 / 3 count Paused Bench
    - Incline Dumbbell Bench Press
    - Mid Grip Bench
    - Close Grip Bench

  - **Deadlift**
    - If you typical deadlift is sumo stance, then your accessory will be conventional and vice versa.

  - **Overhead Press**
    - Typically doesn't require any accessory - high rep (20+) lateral raises are effective

-  **Assisted Compound**: examples include Meadow rows, lat pulldown, chin ups, pull ups and pendulum squats etc...

### 👟 Volume
- Refers to only doing straight sets with reps within the 6 + rep range.

### 🏋️ Strength
- Refers to doing top sets and then followed by back off sets, typically 90% of your top set but the RPE scheme defined takes precedence.

