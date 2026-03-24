
import unittest

from clemcore.backends import CustomResponseModel
from clemcore.clemgame.master import ParseError
from clemcore.clemgame import GameSpec
from master import DealOrNoDeal


def create_dummy_game() -> DealOrNoDeal:
    game_spec = GameSpec.from_dict({
        'game_name': 'dond', 
        'description': 'Deal Or No Deal game between two agents that have to agree on how to divide a set of items between them.', 
        'main_game': 'dond', 
        'players': 2, 
        'languages': ['en', 'it', 'de'], 
        'benchmark': ['3.0'],
        'game_path': './'
    })
    experiment = {
        'name': 'test',
        'mode': 'coop',
        'language': 'de',
        'max_turns': 3,
        'initial_prompt': '<initial-prompt>',
        'proposal_early': '<proposal-prompt>',
        'proposal_timeout': '<proposal-prompt>'
    }
    game = {
        'game_id': 0,
        'item_types': ['book', 'ball', 'hat'],
        'item_types_plural': ['books', 'balls', 'hats'],
        'item_counts': [1, 2, 3],
        'player_a_values': [3, 2, 1],
        'player_b_values': [1, 3, 1],
    }
    players = [CustomResponseModel(), CustomResponseModel()]
    master = DealOrNoDeal(
        game_spec, experiment, players  # type: ignore
    )
    master.setup(**game)
    return master


class DealOrNoDealTestCase(unittest.TestCase):
    def test_dond_messages_alternate(self):
        master = create_dummy_game()
        master.play()

    def test_parse_empty_proposal(self):
        master = create_dummy_game()
        self.assertEqual(
            master._parse_response(master.player_a, '[Proposal:]'), [0, 0, 0]
        )

    def test_parse_wrong_singular(self):
        master = create_dummy_game()
        self.assertEqual(
            master._parse_response(
                master.player_a, '[Proposal: 2 book, 3 ball, 4 hat]'), [2, 3, 4]
        )

    def test_parse_wrong_order(self):
        master = create_dummy_game()
        self.assertEqual(
            master._parse_response(
                master.player_a, '[Proposal: 4 hat, 2 books, 3 ball]'), [2, 3, 4]
        )

    def test_parse_wrong_plural(self):
        master = create_dummy_game()
        self.assertEqual(
            master._parse_response(
                master.player_a, '[proposal: 1 hats, 1 books, 1 balls]'), [1, 1, 1]
        )

    def test_parse_proposal_not_alone(self):
        master = create_dummy_game()
        self.assertEqual(
            master._parse_response(
                master.player_a,
                'This is some unrelated text [Proposal: 4 hat, 2 books, 3 ball] that is not the proposal'
            ), [2, 3, 4]
        )

    def test_parse_invalid_multiple(self):
        master = create_dummy_game()
        self.assertRaises(
            ParseError,
            lambda: master._parse_response(
                master.player_a, '[Proposal: 4 hat, 2 books, 3 hat]'
            )
        )

    def test_parse_invalid_syntax1(self):
        master = create_dummy_game()
        self.assertRaises(
            ParseError,
            lambda: master._parse_response(
                master.player_a, '[4 hat, 2 books, 3 hat]'
            )
        )

    def test_parse_invalid_syntax2(self):
        master = create_dummy_game()
        self.assertRaises(
            ParseError,
            lambda: master._parse_response(
                master.player_a, '[Proposal: 4 hat 2 books 3 hat]'
            )
        )

    def test_parse_no_proposal(self):
        master = create_dummy_game()
        self.assertEqual(
            master._parse_response(
                master.player_a, 'Hello world. This is a fun game.'
            ),
            'Hello world. This is a fun game.'
        )


if __name__ == '__main__':
    unittest.main()
