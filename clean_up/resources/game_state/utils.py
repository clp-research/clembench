from typing import List, Tuple, TypedDict
import base64
import random
from PIL import Image

EMPTY_SYMBOL = "◌"
ICON_SIZE = 128 # Icons are square
TEXT_BASED = ['text', 'hybrid', 'semantic_text']
IMAGE_BASED = ['image']

class GameObject(TypedDict):
    id: str
    coord: Tuple[int, int]

class Icon(GameObject):
    freepik_id: str  # Unique identifier for the icon
    url: str = None  # URL to the icon image
    img: str = None  # Base64 encoded image string

def png_to_base64(png_path):
    """
    Convert a PNG image to a base64 encoded string.
    """
    with open(png_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def number_to_letter(number: int) -> str:
    """
    Converts a number to a lowercase letter (1 -> a, 2 -> b, ..., 26 -> z).
    :param number: The number to convert
    :return: The corresponding lowercase letter
    """
    if 1 <= number <= 26:
        return chr(number + 96)
    raise ValueError(f"Number {number} is out of bounds for lowercase letter conversion (1-26)")

def letter_to_number(letter: str) -> int:
    """
    Converts a lowercase letter to a number (a -> 1, b -> 2, ..., z -> 26).
    :param letter: The lowercase letter to convert
    :return: The corresponding number
    """
    if len(letter) == 1 and letter.isalpha():
        return ord(letter.lower()) - 96
    raise ValueError(f"Letter '{letter}' is not a valid single lowercase letter (a-z)")

def place_objects(modality: str, objects: List[GameObject], background: str) -> List[GameObject]:
    """
    Place objects on the grid based on the modality.
    :param modality: The modality of the game (e.g., 'text', 'image')
    :param objects: List of GameObject to place
    :param grid: The grid string representation
    :return: List of GameObject with updated coordinates
    """
    if modality in TEXT_BASED:
        return place_grid_objects(objects, background)
    elif modality in  IMAGE_BASED:
        # Load the background image to get dimensions
        dim = Image.open(background).size
        return place_icons(objects, dim)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def place_grid_objects(objects: List[GameObject], grid: str) -> List[GameObject]:
    """
    Sample unique coordinates for each object in objects.
    :param grid: The grid string representation
    :param width: The width of the grid
    :param objects: List of GameObject to place
    :return: List of GameObject with updated coordinates
    """
    width = grid.index('\n') + 1 if '\n' in grid else len(grid)
    # + 1 to account for the newline character
    empty_indices = [i for i, char in enumerate(grid) if char == EMPTY_SYMBOL]
    random.shuffle(empty_indices)
    if len(empty_indices) < len(objects):
        raise ValueError("Not enough empty positions in the grid to place all objects.")
    for obj in objects:
        # For some reason, sample() produces conspicuously many duplicate indices when called twice
        # random.choice() works better
        index = random.choice(empty_indices)
        empty_indices.remove(index)  # Ensure unique placement
        x = index % width
        y = index // width
        obj['coord'] = (x, y)
    return objects

def place_icons(objects: List[Icon], img_size: Tuple[int, int]) -> List[Icon]:
    """
    Place icons on the background image and assign them randomized IDs.
    :param objects: List of Icon objects to place
    :param img_size: Size of the icons (width, height)
    :return: List of placed Icon objects with updated coordinates
    """
    width, height = img_size
    min_x = ICON_SIZE / 2 + 10
    min_y = ICON_SIZE / 2 + 10
    max_x = width - ICON_SIZE / 2 - 10
    max_y = height - ICON_SIZE / 2 - 10

    min_diff = ICON_SIZE + 20

    sampled_coordinates = []

    random.shuffle(objects)

    for obj in objects:
        # Ensure the coordinates are within the bounds of the image
        overlap = True
        while overlap:
            overlap = False
            x = random.randint(int(min_x), int(max_x))
            y = random.randint(int(min_y), int(max_y))
            for coord in sampled_coordinates:
                # Ensure the new coordinate is at least min_diff away from existing ones
                if abs(coord[0] - x) < min_diff and abs(coord[1] - y) < min_diff:
                    overlap = True
                    break
        sampled_coordinates.append((x, y))
        obj['id'] = chr(ord('A') + len(sampled_coordinates) - 1)
        obj['coord'] = (x, y)

    return objects

def parse_grid(grid: str) -> list[list[str]]:
    """
    Parses the grid from a string into a 2D list.
    """
    grid = grid.strip().split("\n")
    parsed_grid = []
    for row in grid:
        parsed_row = []
        for char in row:
            parsed_row.append([char])
        parsed_grid.append(parsed_row)
    return parsed_grid