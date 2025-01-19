import time
import unittest
import requests
import threading
import time

sentences = [
    "Python is easy to learn.",
    "AI can enhance decision-making processes.",
    "SpaceX is revolutionizing space travel.",
    "Python excels in data analysis.",
    "AI algorithms learn from data.",
    "Mars rovers explore new terrains.",
    "Python supports multiple programming paradigms.",
    "AI improves user experience on websites.",
    "The International Space Station orbits Earth.",
    "Python's syntax is very readable.",
    "AI in healthcare can predict outcomes.",
    "Astronauts conduct experiments in space.",
    "Python is widely used in web development.",
    "Machine learning is a subset of AI.",
    "NASA aims to return humans to the Moon.",
    "Python libraries simplify complex tasks.",
    "Autonomous vehicles rely on AI technologies.",
    "Voyager 1 has left our solar system.",
    "Python is open-source and community-driven.",
    "Voice assistants use AI to understand speech.",
    "Telescopes help in observing distant galaxies.",
    "Python's popularity grows each year.",
    "AI can identify patterns in big data.",
    "Satellites provide crucial weather data.",
    "Python can run on many operating systems.",
    "Neural networks mimic human brain functions.",
    "Space debris is a growing concern in orbit.",
    "Python scripts automate repetitive tasks.",
    "AI ethics is a growing field of study.",
    "The Hubble Space Telescope has changed our view of the cosmos.",
]


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

    def _get_req_body(self, text: str, task_type: str = ""):
        req_body = {"text": text}
        if task_type != "":
            req_body["config"] = {"task_type": task_type}
        return req_body

    def _try_to_vectorize(self, url: str, text: str, task_type: str = ""):
        req_body = self._get_req_body(text, task_type)

        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)

        # below tests that what we deem a reasonable vector is returned. We are
        # aware of 384 and 768 dim vectors, which should both fall in that
        # range
        self.assertTrue(len(resBody["vector"]) > 100)
        # print(f"vector dimensions are: {len(resBody['vector'])}")

    def test_well_known_ready(self):
        res = requests.get(self.url + "/.well-known/ready")

        self.assertEqual(res.status_code, 204)

    def test_well_known_live(self):
        res = requests.get(self.url + "/.well-known/live")

        self.assertEqual(res.status_code, 204)

    def test_meta(self):
        res = requests.get(self.url + "/meta")

        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), dict)

    def test_vectorizing(self):
        self._try_to_vectorize(
            self.url + "/vectors/",
            "The London Eye is a ferris wheel at the River Thames.",
        )
        self._try_to_vectorize(
            self.url + "/vectors",
            "The London Eye is a ferris wheel at the River Thames.",
        )

    def _test_vectorizing_sentences(self):
        for sentence in sentences:
            self._try_to_vectorize(self.url + "/vectors/", sentence)
            self._try_to_vectorize(self.url + "/vectors", sentence)

    def test_vectorizing_sentences_parallel(self):
        start = time.time()
        threads = []
        for _ in range(10):
            t = threading.Thread(target=self._test_vectorizing_sentences)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        end = time.time()
        print(f"test_vectorizing_sentences_parallel took: {end - start}s")

    def test_vectorizing_sentences(self):
        start = time.time()
        for _ in range(10):
            self._test_vectorizing_sentences()
        end = time.time()
        print(f"test_vectorizing_sentences took: {end - start}s")


if __name__ == "__main__":
    unittest.main()
