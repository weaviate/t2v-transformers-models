import time
import unittest
import requests


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.url = "http://localhost:8000"

        for i in range(0, 100):
            try:
                res = requests.get(self.url + "/.well-known/ready")
                if res.status_code == 204:
                    return
                else:
                    raise Exception("status code is {}".format(res.status_code))
            except Exception as e:
                print("Attempt {}: {}".format(i, e))
                time.sleep(1)

        raise Exception("did not start up")

    def test_well_known_ready(self):
        res = requests.get(self.url + "/.well-known/ready")

        self.assertEqual(res.status_code, 204)

    def test_well_known_live(self):
        res = requests.get(self.url + "/.well-known/live")

        self.assertEqual(res.status_code, 204)

    def test_meta_unauthorized(self):
        res = requests.get(self.url + "/meta")

        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "Unauthorized")

        headers = {"Authorization": "Bearer bad-token"}
        res = requests.get(self.url + "/meta", headers=headers)

        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "Unauthorized")

    def test_meta(self):
        headers = {"Authorization": "Bearer token1"}
        res = requests.get(self.url + "/meta", headers=headers)

        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), dict)

    def test_vectorizing_unauthorized(self):
        req_body = {"text": "The London Eye is a ferris wheel at the River Thames."}
        res = requests.post(self.url + "/vectors", json=req_body)

        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "Unauthorized")

        headers = {"Authorization": "Bearer bad-token"}
        res = requests.post(self.url + "/vectors", json=req_body, headers=headers)

        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "Unauthorized")

    def test_vectorizing(self):
        def get_req_body(task_type: str = ""):
            req_body = {"text": "The London Eye is a ferris wheel at the River Thames."}
            if task_type != "":
                req_body["config"] = {"task_type": task_type}
            return req_body

        def try_to_vectorize(url, task_type: str = ""):
            print(f"url: {url}")
            req_body = get_req_body(task_type)

            headers = {"Authorization": "Bearer token2"}
            res = requests.post(url, json=req_body, headers=headers)
            resBody = res.json()

            self.assertEqual(200, res.status_code)

            # below tests that what we deem a reasonable vector is returned. We are
            # aware of 384 and 768 dim vectors, which should both fall in that
            # range
            self.assertTrue(len(resBody["vector"]) > 100)
            print(f"vector dimensions are: {len(resBody['vector'])}")

        try_to_vectorize(self.url + "/vectors/")
        try_to_vectorize(self.url + "/vectors")


if __name__ == "__main__":
    unittest.main()
