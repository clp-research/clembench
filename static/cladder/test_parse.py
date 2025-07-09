import unittest
from master import parse_response

RESPONSE = """No.

To answer this question, we need to apply the concept of causal inference, 
specifically the backdoor criterion and the frontdoor adjustment.
"""


class ParseTestCase(unittest.TestCase):
    def test_parse_response(self):
        response = parse_response(RESPONSE, ["yes", "no"])
        self.assertEqual(response, "No")


if __name__ == '__main__':
    unittest.main()
