import re


def hour_to_num(hr_str):
    return float(hr_str.split(":")[0]) + (0.5 if hr_str.split(":")[1] == "30" else 0.0)


def _parse_response(response: str):
    """Parse the response.

    Returns a parsed suggested meeting time in (day, start_hour, end_hour).

    Args:
      response: Raw response from the model.

    Returns:
      A tuple of (day, start_hour, end_hour).
    """
    time_strs = re.findall(r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+", response)
    if not time_strs:
        return "", -1, -1
    # If multiple matches are found, return the first one.
    time_str = time_strs[0]
    day, hour_str = (
        time_str.split(",")[0].strip(),
        time_str.split(",")[1].strip(),
    )
    start_hour, end_hour = (
        hour_str.split("-")[0].strip(),
        hour_str.split("-")[1].strip(),
    )
    return day, hour_to_num(start_hour), hour_to_num(end_hour)


def compute_solve_rate(response: str, solution: str):
    """Computes solve rate by comparing model responses to golden solutions.

    Args:
      responses: A list of model responses.
      solutions: The corresponding list of golden solutions for the same tasks.

    Returns:
      A scalr solve rate.
    """
    r_day, r_start_hour, r_end_hour = _parse_response(response)
    s_day, s_start_hour, s_end_hour = _parse_response(solution)
    return int(
        r_day == s_day and r_start_hour == s_start_hour and r_end_hour == s_end_hour
    )


def calendar_planning_accuracy(doc, response):
    response = response[0]
    solution = doc["golden_plan"]

    overall_solved_rate = compute_solve_rate(response, solution)
    return {"acc": overall_solved_rate}
