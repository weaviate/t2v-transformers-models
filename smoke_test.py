import time
import unittest
import requests


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.url = 'http://localhost:8000'

        for i in range(0, 100):
            try:
                res = requests.get(self.url + '/.well-known/ready')
                if res.status_code == 204:
                    return
                else:
                    raise Exception("status code is {}".format(res.status_code))
            except Exception as e:
                print("Attempt {}: {}".format(i, e))
                time.sleep(1)

        raise Exception("did not start up")

    def test_well_known_ready(self):
        res = requests.get(self.url + '/.well-known/ready')

        self.assertEqual(res.status_code, 204)

    def test_well_known_live(self):
        res = requests.get(self.url + '/.well-known/live')

        self.assertEqual(res.status_code, 204)

    def test_meta(self):
        res = requests.get(self.url + '/meta')

        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), dict)

    def test_vectorizing(self):
        def try_to_vectorize(url):
            print(f"url: {url}")
            req_body = {'text': 'The London Eye is a ferris wheel at the River Thames.'}

            res = requests.post(url, json=req_body)
            resBody = res.json()

            self.assertEqual(200, res.status_code)

            # below tests that what we deem a reasonable vector is returned. We are
            # aware of 384 and 768 dim vectors, which should both fall in that
            # range
            self.assertTrue(len(resBody['vector']) > 100)

        try_to_vectorize(self.url + "/vectors/")
        try_to_vectorize(self.url + "/vectors")


if __name__ == "__main__":
    unittest.main()
