import unittest
from social_arena.policies import final_text


class ProviderTests(unittest.TestCase):
    def message(self, text, phase=None):
        return {'type': 'message', 'phase': phase, 'content': [{'type': 'output_text', 'text': text}]}

    def test_final_answer_excludes_commentary(self):
        answer = '{"action":{"text":"Approve."}}'
        result = {'status': 'completed', 'output': [self.message('Private deliberation', 'commentary'), self.message(answer, 'final_answer')]}
        self.assertEqual(final_text(result), answer)

    def test_unphased_legacy_and_ambiguous_or_incomplete(self):
        message = self.message('{"action":{"approve":true}}')
        self.assertEqual(final_text({'status': 'completed', 'output': [message]}), message['content'][0]['text'])
        self.assertEqual(final_text({'status': 'incomplete', 'output': [message]}), '')
        self.assertEqual(final_text({'status': 'completed', 'output': [message, message]}), '')
        self.assertEqual(final_text({'status': 'completed', 'output': [self.message('{}', 'commentary')]}), '')
