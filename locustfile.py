import json
from locust import HttpUser, TaskSet, task, between


class NERServiceTaskSet(TaskSet):
    @task
    def test_ner_endpoint(self):
        payload = {
            "texts": [
                "OpenAI is located in San Francisco."
            ],  # Example text for NERRequest
            "threshold": 0.8,
        }
        self.client.post("/predict", json=payload)


class NERServiceUser(HttpUser):
    tasks = [NERServiceTaskSet]
    # wait_time = between(1, 5)  # Simulates a wait time between 1 and 5 seconds between tasks

    def on_start(self):
        """on_start is called when a Locust start before any task is scheduled"""
        print("Starting load test...")

    def on_stop(self):
        """on_stop is called when the TaskSet is stopping"""
        print("Ending load test...")