# data/generator.py

import random

from templates import (
    GREETINGS,
    FOODS,
    PROBLEMS,
    APOLOGIES,
    FOLLOW_UP,
)

NUM_SAMPLES = 10000


def generate_conversation():
    greeting = random.choice(GREETINGS)
    food = random.choice(FOODS)
    problem = random.choice(PROBLEMS)
    apology = random.choice(APOLOGIES)
    follow_up = random.choice(FOLLOW_UP)

    conversation = f"""<user>
{greeting}

My {food} {problem}.
</user>

<assistant>
{apology}

{follow_up}
</assistant>

"""

    return conversation


with open("foodie_dataset.txt", "w", encoding="utf-8") as file:
    for _ in range(NUM_SAMPLES):
        file.write(generate_conversation())

print(f"✅ Generated {NUM_SAMPLES} conversations!")
