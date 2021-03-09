import unittest
import requests


class SmokeTest(unittest.TestCase):

    def testCalculation(self):
        url = 'http://localhost:8000/vectors/'
        req_body = {'text': 'The London Eye is a ferris wheel at the River Thames.'}

        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        # TODO: Make dynamic when supporting models with other dimensions
        self.assertEqual(768, len(resBody['vector']))


if __name__ == "__main__":
    unittest.main()
