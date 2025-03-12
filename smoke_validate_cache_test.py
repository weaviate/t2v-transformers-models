import time
import unittest
import requests
import threading
import time
from smoke_test import sentences


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
        return text, resBody["vector"]

    def _test_vectorizing_sentences(self):
        for sentence in sentences:
            self._try_to_vectorize(self.url + "/vectors/", sentence)
            self._try_to_vectorize(self.url + "/vectors", sentence)

    def test_vectorize_payload_with_config(self):
        weaviate_facts = [
            "Vector database for semantic search.",
            "Supports similarity-based queries.",
            "Integrates with ML for classification.",
        ]
        for _ in range(10):
            for fact in weaviate_facts:
                self._try_to_vectorize(self.url + "/vectors/", fact, "query")
                self._try_to_vectorize(self.url + "/vectors", fact, "passage")

    def test_vectorizing_cached_results(self):
        start = time.time()
        before = {}
        for sentence in sentences:
            txt, vector = self._try_to_vectorize(self.url + "/vectors/", sentence)
            before[txt] = vector

        threads = []
        for _ in range(10):
            t = threading.Thread(target=self._test_vectorizing_sentences)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        after = {}
        for sentence in sentences:
            txt, vector = self._try_to_vectorize(self.url + "/vectors/", sentence)
            after[txt] = vector

        for key, value in before.items():
            print(
                f"{key} vec.len: {len(value)} after.len: {len(after.get(key))} equal: {value == after[key]}"
            )
            self.assertEqual(value, after[key])

        end = time.time()
        print(f"test_vectorizing_sentences_parallel took: {end - start}s")


if __name__ == "__main__":
    unittest.main()
