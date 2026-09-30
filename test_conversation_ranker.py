import unittest
from http.client import RemoteDisconnected
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from conversation_ranker import NoRedirectHandler, ensure_local_url, make_blocks, rank_results, request_json, score_block


class MakeBlocksTests(unittest.TestCase):
    def test_blocks_respect_size_and_overlap(self):
        blocks = make_blocks("one two three four five six", block_words=4, overlap_words=1)
        self.assertEqual(blocks, ["one two three four", "four five six"])

    def test_empty_input_has_no_blocks(self):
        self.assertEqual(make_blocks(" \n\t", block_words=10, overlap_words=2), [])


class OpenJevTests(unittest.TestCase):
    def test_only_loopback_http_is_allowed(self):
        self.assertEqual(ensure_local_url("http://127.0.0.1:8080/"), "http://127.0.0.1:8080")
        self.assertEqual(ensure_local_url("http://[::1]:8080"), "http://[::1]:8080")
        with self.assertRaises(ValueError):
            ensure_local_url("http://localhost:8080")
        with self.assertRaises(ValueError):
            ensure_local_url("https://example.com")
        with self.assertRaises(ValueError):
            ensure_local_url("http://192.168.1.10:8080")

    def test_redirects_are_refused(self):
        handler = NoRedirectHandler()
        with self.assertRaises(HTTPError) as error:
            handler.redirect_request(Request("http://127.0.0.1"), None, 302, "Found", {}, "https://example.com")
        self.assertIn("redirect refused", str(error.exception))

    @patch("conversation_ranker.build_opener")
    def test_remote_disconnect_is_reported_as_a_retriable_connection_error(self, build_opener):
        build_opener.return_value.open.side_effect = RemoteDisconnected("Remote end closed connection without response")

        with self.assertRaisesRegex(RuntimeError, "Could not reach local OpenJev.*Remote end closed connection"):
            request_json("http://127.0.0.1:8080/v1/models")

    @patch("conversation_ranker.request_json")
    def test_score_block_sends_questions_and_reads_yes_probability(self, request_json):
        request_json.return_value = {
            "answers": {
                "q0": {"type": "noul", "noul": 0.83},
                "q1": {"type": "noul", "noul": 0.17},
            }
        }

        scores = score_block(
            "http://127.0.0.1:8080",
            "verdict-1.4",
            "The conversation block",
            ["Does this mention politics?", "Does this mention vision?"],
            None,
        )

        self.assertEqual(scores, [0.83, 0.17])
        payload = request_json.call_args.args[1]
        self.assertEqual(payload["state"], "The conversation block")
        self.assertEqual(payload["model"], "verdict-1.4")
        self.assertEqual(payload["questions"]["q0"]["type"], "noul")
        self.assertEqual(payload["questions"]["q1"]["instructions"], "Does this mention vision?")

    def test_rank_results_sorts_descending_and_caps_count(self):
        blocks = ["first", "second", "third"]
        ranked = rank_results(blocks, ["Q?"], [[0.4], [0.9], [0.7]], top=2)
        self.assertEqual([item["block"] for item in ranked["Q?"]], [2, 3])
        self.assertEqual([item["text"] for item in ranked["Q?"]], ["second", "third"])


if __name__ == "__main__":
    unittest.main()