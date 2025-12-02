from typing import Optional
import dspy

class OptimizerSignature(dspy.Signature):
    """
    Given multiple instruction–code pairs, generate a single reusable and optimized
    Python function that performs all described operations in one call.

    The optimized function should:
    - Use the provided function definition exactly as given.
    - Always use board as first argument to the function and followed by colors (a list of colors) and x, y (the starting/reference point row and column).
    - Do not use shapes as parameters. Keep shapes as fixed attributes (e.g., shapes like 'washer', 'nut', 'bridge') and use them only inside the function. Use the exact shape names as in the code snippets.
    - Do not hardcode color values inside the function.
    - Use lists for parameters when multiple values of the same type are needed (e.g., multiple colors).
    - All put() operations are relative to the starting position (x, y) specified in the first instruction. Use that as the reference point and as an input parameter to the function. The function should compute the relative offsets of all other shapes based on how their coordinates differ from the first placement.
       - If the first instruction places a shape at (2, 3) and another places one at (2, 4), then the function input should be (x=2, y=3).
       - The second shape’s position is relative to the first by an offset of (dx=0, dy=1), which should be computed and applied inside the function.
    - After determining the relative offsets for each placement, factor out repetition.
    - Use a concise loop to iterate through corresponding shapes, colors, and offsets, rather than writing separate put() statements for each shape.
    - Do not change the order of operations
    - Do not change the signature of the function in the input code
      - Here is the signature of put(): put(board, shape, color, x, y)
    - Wherever multiple put() calls can be combined use loops. Do not repeat similar lines of code.
    - When executed once, it should perform all of the described actions together.
    - Use clean, valid Python syntax (PEP-8 compliant).
    - Maintain semantic equivalence with all provided code snippets.
    - Output a valid standalone Python function definition.

    Given:
    - prompt: Prompt with the following details:
        - function definition: definition of the function including its name
        - function usage: how the function will be used
        - code across turns: a list of code snippets corresponding to different instructions
        - optional error feedback from previous attempts

    Produce:
        - A single optimized Python function as a string.
    
    Make sure the Python function is valid and executable by executing `exec()` on it.
    """

    prompt = dspy.InputField(
        desc="Prompt with function name, usage, instruction-code pairs, and optional error feedback."
    )
    #history: Optional[dspy.History] = dspy.InputField()    

    optimized_function = dspy.OutputField(
        desc="A single optimized Python function as a string."
    )