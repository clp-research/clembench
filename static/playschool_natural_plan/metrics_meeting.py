import collections
import datetime
import json
from typing import Any


MAX_CONSTRAINTS = 10
NUM_SAMPLES_PER_CONSTRAINTS = 100


def convert_to_time_obj(time_str: str):
    return datetime.datetime.strptime(time_str, "%I:%M%p")


def process_constraints(data: tuple[Any, ...]):
    contraints = collections.defaultdict(dict)
    for name, location, times, meeting_time in data:
        contraints[name]["location"] = location

        start_time = convert_to_time_obj(times.split("to")[0].strip())
        end_time = convert_to_time_obj(times.split("to")[1].strip())
        contraints[name]["start_time"] = start_time
        contraints[name]["end_time"] = end_time
        contraints[name]["meeting_time"] = meeting_time
    return contraints


def validator_from_text(
    plan: list[str],
    processed_constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
):
    """Compute the number of valid meetings scheduled in the plan (text format)."""

    # This function is used when the generated plan follows the few-shot format.

    # Example:
    # You start at Russian Hill at 9:00AM.
    # You travel to Marina District in 7 minutes and arrive at 9:07AM.
    # You wait until 3:45PM.
    # You meet James for 75 minutes from 3:45PM to 5:00PM.

    met_with = {}

    score = 0
    cur_location = start_location
    cur_time = convert_to_time_obj(initial_time)
    for step in plan:
        try:
            if step.startswith("You start"):
                continue
            elif step.startswith("You travel"):
                destination = step.split("travel to ")[1].split(" in")[0].strip()
                cur_time = cur_time + datetime.timedelta(
                    minutes=dist_matrix[cur_location][destination]
                )

                cur_location = destination
            elif step.startswith("You wait"):
                raw_end_time = step.split("wait until ")[1].split(".")[0].strip()
                end_time = convert_to_time_obj(raw_end_time)

                if end_time <= cur_time:
                    raise ValueError("Cannot go backwards in time")

                cur_time = end_time
            elif step.startswith("You meet"):
                person = step.split("meet ")[1].split(" for")[0].strip()
                if person in met_with:
                    raise ValueError(
                        "Person {person} already met with {met_with}".format(
                            person=person, met_with=met_with[person]
                        )
                    )

                met_with[person] = 1
                new_time = cur_time + datetime.timedelta(
                    minutes=processed_constraints[person]["meeting_time"]
                )

                if (
                    cur_location == processed_constraints[person]["location"]
                    and cur_time >= processed_constraints[person]["start_time"]
                    and new_time <= processed_constraints[person]["end_time"]
                ):
                    score += 1
                    cur_time = new_time
                else:
                    raise ValueError("Invalid meeting time or location")
            else:
                raise ValueError("Unknown plan format")

        except Exception as e:
            print("Had error: {e} with step: {step}".format(e=e, step=step))
            break

    return score


def validator_from_dict(
    plan: list[Any],
    processed_constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
):
    """Compute the number of valid meetings scheduled in the plan (JSON format)."""

    # This function is used when the generated plan does not follow
    # the few-shot format. Before calling this function, the plan should be
    # converted to a list of steps, each of which is a dictionary with the
    # following fields:
    #     {"location": LOCATION,
    #     "person_name": PERSON_TO_MEET,
    #     "start_time": START_TIME}.

    # Example:
    # Plan (text):
    # You start at Russian Hill at 9:00AM.
    # You travel to Marina District in 7 minutes and arrive at 9:07AM.
    # You wait until 3:45PM.
    # You meet James for 75 minutes from 3:45PM to 5:00PM.

    # Corresponding plan format taken by this function (JSON):
    # ```json
    # [
    #     {
    #         "location": "Russian Hill",
    #         "person_name": "N/A",
    #         "start_time": "9:00AM"
    #     },
    #     {
    #         "location": "Marina District",
    #         "person_name": "N/A",
    #         "start_time": "9:07AM"
    #     },
    #     {
    #         "location": "Marina District",
    #         "person_name": "James",
    #         "start_time": "3:45PM"
    #     }
    # ]
    # ```

    met_with = {}

    score = 0
    cur_location = start_location
    cur_time = convert_to_time_obj(initial_time)
    for step in plan[1:]:
        try:
            raw_start_time = step["start_time"]
            location = step["location"]
            if location and location != cur_location:
                cur_time = cur_time + datetime.timedelta(
                    minutes=dist_matrix[cur_location][location]
                )
                cur_location = location

            start_time = convert_to_time_obj(raw_start_time)
            if start_time < cur_time:
                raise ValueError("Start time too early")
            cur_time = start_time
            person = step["person_name"]
            if person not in processed_constraints:
                continue
            if person in met_with:
                raise ValueError(
                    "Person {person} already met with {met_with}".format(
                        person=person, met_with=met_with[person]
                    )
                )

            met_with[person] = 1
            new_time = cur_time + datetime.timedelta(
                minutes=processed_constraints[person]["meeting_time"]
            )

            if (
                cur_location == processed_constraints[person]["location"]
                and cur_time >= processed_constraints[person]["start_time"]
                and new_time <= processed_constraints[person]["end_time"]
            ):
                score += 1
                cur_time = new_time
            else:
                raise ValueError("Invalid meeting time or location")

        except ValueError as e:
            print("Had error: {e} with step: {step}".format(e=e, step=step))
            break

    return score


def parse_text_plan(plan: str):
    """Parse the text plan into a list of steps."""

    prefix = "SOLUTION:"
    if prefix in plan:
        plan = plan[plan.find(prefix) + len(prefix) :].strip()
    plan = plan.split(".")
    plan = [step.strip() for step in plan]
    final_plan = []
    for step in plan:
        if step:
            final_plan.append(step)
    return final_plan


def meeting_planning_accuracy(doc: dict, response: list[str]) -> dict[str, float]:
    doc["constraints"] = json.loads(doc["constraints"])
    start_location, initial_time = doc["constraints"][0]
    constraints = process_constraints(doc["constraints"][1:])
    dist_matrix = json.loads(doc["dist_matrix"])
    pred_plan = response[0]

    pred_plan = parse_text_plan(pred_plan)
    score = validator_from_text(
        pred_plan, constraints, start_location, initial_time, dist_matrix
    )

    golden_plan = eval(doc["golden_plan"])
    golden_score = validator_from_text(
        golden_plan, constraints, start_location, initial_time, dist_matrix
    )

    curr_acc = int(score == golden_score)

    return {"acc": curr_acc}
