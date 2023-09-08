# Method to generate random prompts
import random


def words_list(is_string):
    """
    Function to get a list of words, if is_string = 0, return all lists of words.
    If is_string = 1, return adjective list only.
    Else, return all lists of words as a string.
    :param is_string: 0, 1, or 2

    """
    # list of words related to street
    street = ["street", "road", "avenue"]
    # list of words related to people
    people = ["people", "pedestrians", "walkers", "travelers"]
    # list of words related to time of day
    time = ["day", "night", "morning", "afternoon", "evening", "dawn"]
    # list of words related to weather
    weather = ["sunny", "cloudy", "rainy", "snowy", "stormy", "foggy"]
    # list of words related to picture pov
    pov = ["overhead", "ground-level", "eye-level"]
    # list of adjectives
    adjectives_str = ["high-quality, realistic, beautiful, colorful, detailed, high-resolution, high-definition,"
                      " no deformation, no blur, no noise, no artifacts, no compression, no pixelation"]
    # list for cars
    cars = ["cars", "automobiles", "vehicles"]
    # list of group synonyms
    group = ["group", "crowd", "bunch", "cluster"]
    # list of lighting synonyms
    lighting = ["dark", "bright", "dim", "shiny", "obscure"]

    if is_string == 0:
        # Return lists
        return street, people, time, weather, pov, cars, group, lighting
    if is_string == 1:
        return adjectives_str
    else:
        # Assemble all the words into a string
        return f"{street}, {people}, {time}, {weather}, {pov}, {adjectives_str}, {cars}"


def group_list(group, people, street, time, weather, lighting, pov, adjectives_str):
    # random number from2 to 5
    num_people = random.randint(2, 5)

    return f"A picture of a {group} of {num_people} {people} walking on the {street}, during the {time} " \
           f"in {weather} weather and it's {lighting}," \
           f" the picture is taken from a {pov} perspective The picture is {adjectives_str}"


def car_list(street, time, weather, pov, adjectives_str, cars, lighting):
    # random number from2 to 5
    num = random.randint(2, 5)

    return f"A picture of {num} {cars} parked on the {street}, during the {time} in {weather} weather " \
           f"and it's {lighting}," \
           f" the picture is taken from a {pov} perspective The picture is {adjectives_str}"


def road_list(street, time, weather, pov, adjectives_str, cars, lighting):
    # random number from2 to 5
    num_people = random.randint(2, 5)

    return f"A picture of a {num_people} {cars} driving on the  {street}, during the {time} " \
           f"in {weather} weather and it's {lighting}," \
           f" the picture is taken from a {pov} perspective The picture is {adjectives_str}"


def normal_list(street, people, time, weather, pov, adjectives_str, cars, lighting):
    return f"a realistic picture of a {pov} {street} with {cars} driving and {people} " \
           f"walking on the sidewalk during {time} with {weather} weather and it's {lighting},  " \
           f"{adjectives_str}"


def pedestrian_list(street, people, time, weather, pov, adjectives_str, lighting):
    return f"A picture of {people} taken in the {street}, during the {time} in {weather} weather and it's {lighting}," \
           f" the picture is taken from a {pov} perspective The picture is {adjectives_str}"


def cn_prompt(time, weather, lighting, adjectives_str):
    return f"A picture during {time} time," \
           f" it is {weather} weather and the lighting is {lighting}," \
           f" the picture is {adjectives_str}"


def prompt_generator(use):
    """
    Generate a random prompt for the API
    :param use: 1 if we want cars and pedestrians, everything else if we want only pedestrians
    :return: string of prompt
    """

    prompt = ""
    street, people, time, weather, pov, cars, group, lighting = words_list(0)

    adjectives_str = words_list(1)
    street = random.choice(street)
    people = random.choice(people)
    time = random.choice(time)
    weather = random.choice(weather)
    pov = random.choice(pov)
    cars = random.choice(cars)
    group = random.choice(group)
    lighting = random.choice(lighting)

    # These prompt are best used with stable diffusion
    if use == 1:
        print("sd prompt for bdd")
        # generate a random number from 1 to 4
        num = random.randint(1, 5)
        # If the number is 1, generate a prompt with cars
        if num == 1:
            prompt = car_list(street, time, weather, pov, adjectives_str, cars, lighting)
        elif num == 2:
            prompt = road_list(street, time, weather, pov, adjectives_str, cars, lighting)
        elif num == 3:
            prompt = normal_list(street, people, time, weather, pov, adjectives_str, cars, lighting)
        elif num == 4:
            prompt = group_list(group, people, street, time, weather, lighting, pov, adjectives_str)
        elif num == 5:
            prompt = pedestrian_list(street, people, time, weather, pov, adjectives_str, lighting)
    # CN prompt for bdd
    elif use == 2:
        print("CN prompt")
        prompt = cn_prompt(time, weather, lighting, adjectives_str)

    return prompt
