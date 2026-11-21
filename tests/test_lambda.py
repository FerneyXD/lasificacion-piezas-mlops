import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

os.environ.update(ENDPOINT_NAME="ep", CLASS_NAMES="engranaje,perno,tuerca", THRESHOLD="0.8",
                  AWS_DEFAULT_REGION="us-east-1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda_fn"))
import lambda_function as lf  # noqa: E402


def event(key):
    return {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": key}}}]}


def endpoint_returns(probs):
    lf.smr = MagicMock()
    lf.smr.invoke_endpoint.return_value = {"Body": io.BytesIO(json.dumps({"predictions": [probs]}).encode())}
    lf.s3 = MagicMock()
    lf.s3.get_object.return_value = {"Body": io.BytesIO(b"img")}


class LambdaTests(unittest.TestCase):
    def test_confident_prediction_moves_to_class_folder(self):
        endpoint_returns([0.05, 0.90, 0.05])
        out = lf.lambda_handler(event("entrada/a.jpg"), None)
        self.assertEqual(out["processed"][0]["dest"], "clasificadas/perno/a.jpg")
        lf.s3.delete_object.assert_called_once_with(Bucket="b", Key="entrada/a.jpg")

    def test_low_confidence_goes_to_no_identificadas(self):
        endpoint_returns([0.40, 0.35, 0.25])
        out = lf.lambda_handler(event("entrada/a.png"), None)
        self.assertEqual(out["processed"][0]["dest"], "no-identificadas/a.png")

    def test_probability_equal_to_threshold_is_not_accepted(self):
        endpoint_returns([0.8, 0.1, 0.1])
        out = lf.lambda_handler(event("entrada/a.jpg"), None)
        self.assertEqual(out["processed"][0]["dest"], "no-identificadas/a.jpg")

    def test_keys_outside_incoming_prefix_are_ignored(self):
        endpoint_returns([0.9, 0.05, 0.05])
        out = lf.lambda_handler(event("clasificadas/perno/a.jpg"), None)
        self.assertEqual(out["processed"], [])
        lf.smr.invoke_endpoint.assert_not_called()

    def test_url_encoded_key(self):
        endpoint_returns([0.9, 0.05, 0.05])
        out = lf.lambda_handler(event("entrada/pieza+uno%C3%B1.jpg"), None)
        self.assertEqual(out["processed"][0]["dest"], "clasificadas/engranaje/pieza unoñ.jpg")


if __name__ == "__main__":
    unittest.main()
