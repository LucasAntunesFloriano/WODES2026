import random

VECTOR_SIZE = 15

EMITTER = 0

SMALL, MEDIUM, LARGE = 1, 2, 3
WHITE, BLACK, GRAY = 4, 5, 6
PLASTIC, METAL, WOOD = 7, 8, 9
END_SENSOR = 10

ACT1, ACT2, ACT3, ACT4 = 11, 12, 13, 14
#ACT1 trigegrs on small boxes
#ACT2 trigger on white boxes that aren't small
#ACT3 triggers on plastic boxes that aren't small or white
#ACT4 triggers for all other boxes not contemplated before

MissingCombinations = ["small", "black", "wood"]

def event_vector(index):
    v = [0] * VECTOR_SIZE
    v[index] = 1
    return v


def simulate_box():
    events = []

    size = random.choice(["small", "medium", "large"])
    color = random.choice(["white", "black", "gray"])
    material = random.choice(["plastic", "metal", "wood"])

    while ((size in MissingCombinations) and (color in MissingCombinations) and (material in MissingCombinations)):
        size = random.choice(["small", "medium", "large"])
        color = random.choice(["white", "black", "gray"])
        material = random.choice(["plastic", "metal", "wood"])
    
    events.append(event_vector(EMITTER))

    if size == "small":
        events.append(event_vector(SMALL))
    elif size == "medium":
        events.append(event_vector(MEDIUM))
    else:
        events.append(event_vector(LARGE))

    if color == "white":
        events.append(event_vector(WHITE))
    elif color == "black":
        events.append(event_vector(BLACK))
    else:
        events.append(event_vector(GRAY))

    if material == "plastic":
        events.append(event_vector(PLASTIC))
    elif material == "metal":
        events.append(event_vector(METAL))
    else:
        events.append(event_vector(WOOD))

    events.append(event_vector(END_SENSOR))

    if size == "small":
        events.append(event_vector(ACT1))
    elif size != "small" and color == "white":
        events.append(event_vector(ACT2))
    elif size != "small" and color != "white" and material == "plastic":
        events.append(event_vector(ACT3))
    else:
        events.append(event_vector(ACT4))

    return events


def simulate_factory(num_boxes=1):
    all_events = []
    for _ in range(num_boxes):
        all_events.extend(simulate_box())
    return all_events


if __name__ == "__main__":
    events = simulate_factory(num_boxes=10000)
    with open(r"C:\Users\lucas\OneDrive\Área de Trabalho\output2.txt", "w") as f:
        for e in events:
            #if e[0] == 1:
            #    print("")
            print(e, file=f)
