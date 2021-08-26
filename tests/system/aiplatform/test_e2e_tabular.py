# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import uuid
from urllib import request

import pytest

from google.cloud import aiplatform
from tests.system.aiplatform import e2e_base


_BLOB_PATH = "california-housing-data.csv"
_DATASET_SRC = "https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv"
_DIR_NAME = os.path.dirname(os.path.abspath(__file__))
_LOCAL_TRAINING_SCRIPT_PATH = os.path.join(
    _DIR_NAME, "test_resources/california_housing_training_script.py"
)


@pytest.mark.usefixtures("prepare_staging_bucket", "delete_staging_bucket", "teardown")
class TestEndToEndTabular(e2e_base.TestEndToEnd):
    """End to end system test of the Vertex SDK with tabular data adapted from
    reference notebook http://shortn/_eyoNx3SN0X"""

    _temp_prefix = "temp-vertex-sdk-e2e-tabular"

    def test_end_to_end_tabular(self, shared_state):
        """Build dataset, train a custom and AutoML model, deploy, and get predictions"""

        assert shared_state["bucket"]
        bucket = shared_state["bucket"]

        blob = bucket.blob(_BLOB_PATH)

        # Download the CSV file into memory and save it directory to staging bucket
        with request.urlopen(_DATASET_SRC) as response:
            data = response.read()
            blob.upload_from_string(data)

        # Collection of resources generated by this test, to be deleted during teardown
        shared_state["resources"] = []

        aiplatform.init(
            project=e2e_base._PROJECT,
            location=e2e_base._LOCATION,
            staging_bucket=shared_state["staging_bucket_name"],
        )

        # Create and import to single managed dataset for both training jobs

        dataset_gcs_source = f'gs://{shared_state["staging_bucket_name"]}/{_BLOB_PATH}'

        ds = aiplatform.TabularDataset.create(
            display_name=f"{self._temp_prefix}-dataset-{uuid.uuid4()}",
            gcs_source=[dataset_gcs_source],
            sync=False,
        )

        shared_state["resources"].extend([ds])

        # Define both training jobs

        custom_job = aiplatform.CustomTrainingJob(
            display_name=f"{self._temp_prefix}-train-housing-custom-{uuid.uuid4()}",
            script_path=_LOCAL_TRAINING_SCRIPT_PATH,
            container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-2:latest",
            requirements=["gcsfs==0.7.1"],
            model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-2:latest",
        )

        automl_job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"{self._temp_prefix}-train-housing-automl-{uuid.uuid4()}",
            optimization_prediction_type="regression",
            optimization_objective="minimize-rmse",
        )

        # Kick off both training jobs, AutoML job will take approx one hour to run

        custom_model = custom_job.run(
            ds,
            replica_count=1,
            model_display_name=f"{self._temp_prefix}-custom-housing-model-{uuid.uuid4()}",
            sync=False,
        )

        automl_model = automl_job.run(
            dataset=ds,
            target_column="median_house_value",
            model_display_name=f"{self._temp_prefix}-automl-housing-model-{uuid.uuid4()}",
            sync=False,
        )

        shared_state["resources"].extend(
            [automl_job, automl_model, custom_job, custom_model]
        )

        # Deploy both models after training completes
        custom_endpoint = custom_model.deploy(machine_type="n1-standard-4", sync=False)
        automl_endpoint = automl_model.deploy(machine_type="n1-standard-4", sync=False)
        shared_state["resources"].extend([automl_endpoint, custom_endpoint])

        # Send online prediction with same instance to both deployed models
        # This sample is taken from an observation where median_house_value = 94600
        custom_endpoint.wait()
        custom_prediction = custom_endpoint.predict(
            [
                {
                    "longitude": -124.35,
                    "latitude": 40.54,
                    "housing_median_age": 52.0,
                    "total_rooms": 1820.0,
                    "total_bedrooms": 300.0,
                    "population": 806,
                    "households": 270.0,
                    "median_income": 3.014700,
                },
            ]
        )
        automl_endpoint.wait()
        automl_prediction = automl_endpoint.predict(
            [
                {
                    "longitude": "-124.35",
                    "latitude": "40.54",
                    "housing_median_age": "52.0",
                    "total_rooms": "1820.0",
                    "total_bedrooms": "300.0",
                    "population": "806",
                    "households": "270.0",
                    "median_income": "3.014700",
                },
            ]
        )

        # Ensure a single prediction was returned
        assert len(custom_prediction.predictions) == 1
        assert len(automl_prediction.predictions) == 1

        # Ensure the models are remotely accurate
        try:
            automl_result = automl_prediction.predictions[0]["value"]
            custom_result = custom_prediction.predictions[0][0]
            assert 200000 > automl_result > 50000
            assert 200000 > custom_result > 50000
        except KeyError as e:
            raise RuntimeError("Unexpected prediction response structure:", e)
