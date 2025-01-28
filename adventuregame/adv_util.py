"""
    Utility functions for adventuregame.
"""

def fact_str_to_tuple(fact_string: str, value_delimiter_l: str = "(", value_separator: str = ","):
    """
    Split a string fact and return its values as a tuple.
    """
    first_split = fact_string.split(value_delimiter_l, 1)
    fact_type = first_split[0]
    if value_separator in first_split[1]:
        values_split = first_split[1][:-1].split(value_separator, 1)
        return fact_type, values_split[0], values_split[1]
    else:
        return fact_type, first_split[1][:-1]


def fact_tuple_to_str(fact_tuple: tuple, value_delimiter_l: str = "(", value_separator: str = ",",
                      value_delimiter_r: str = ")", ):
    """
    Convert fact tuple to string version.
    """
    values = fact_tuple[1:]
    # print(values)
    values_str = value_separator.join(values)
    # print(values_str)
    fact_str = f"{fact_tuple[0]}{value_delimiter_l}{values_str}{value_delimiter_r}"
    return fact_str