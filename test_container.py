#!/usr/bin/env python

import logging
import numpy as np

import requests
import docker

from util import read_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


client = docker.from_env()


SUBMISSION_IMAGE = "submission-open-loop:v0.1.0"

# set up a submission (agent) container
# assumes an image "submission:v0.1.0" was created using `make` in reference/environment
seeds = read_seeds()
scores = []
for i, seed in enumerate(seeds):
    print(f"INFO: evaluating seed {i} of {len(seeds)}")
    submission = client.containers.run(
        image=SUBMISSION_IMAGE,
        name="agent",
        network="evalai_rangl",
        detach=False,
        auto_remove=True,
        environment={
            "RANGL_SEED": seed,
            "RANGL_ENVIRONMENT_URL": "http://nztc:5000",
        },
    )
    logger.debug(f"Created submission")
    logger.debug(f"Completed submission")

    # TODO evaluation script should be executed here, but while prototyping we do it here

    # fetch results from environment
    output = submission.decode("utf-8").strip()

    print("output")
    print(output)

    # assumption: final line in stdout is the instance id
    instance_id = output.split("\n")[-1]
    logger.debug(f"Instance id: {instance_id}")

    # send a request to the score endpoint
    ENVIRONMENT_URL = "http://localhost:5000/"

    # fetch score for submission
    url = f"{ENVIRONMENT_URL}/v1/envs/{instance_id}/score/"
    response = requests.get(f"{ENVIRONMENT_URL}/v1/envs/{instance_id}/score/").json()

    score = response["score"]["value1"]
    print("score:", score)
    scores.append(score)

print("Evaluation completed using {} seeds.".format(len(seeds)))
print("Final average score: ", np.average(scores))
