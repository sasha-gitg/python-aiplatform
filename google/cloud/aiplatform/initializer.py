# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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
import logging
from typing import Dict, Optional

import google.auth
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import utils


def validate_region(region: str):
    """Validates region against supported regions.

    Args:
        region: region to validate
    Raises:
        ValueError if region is not in supported regions.
    """
    region = region.lower()
    if region not in SUPPORTED_REGIONS:
        raise ValueError(
            f"Unsupported region for AI Platform, select from {SUPPORTED_REGIONS}"
        )


class Init:
    """Stores common parameters and options for API calls."""

    def __init__(self):
        self._project = None
        self._experiment = None
        self._location = None
        self._staging_bucket = None
        self._credentials = None

    def init(
        self,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        experiment: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Updates common initalization parameters with provided options.

        Args:
            project: The default project to use when making API calls.
            location: The default location to use when making API calls. If not set defaults to us-central-1
            experiment: The experiment to assign
            staging_bucket: The default staging bucket to use to stage artifacts when making API calls.
            credentials: The default custom credentials to use when making API calls. If not provided,
             crendentials will be ascertained from the environment.
        """
        if project:
            self._project = project
        if location:
            utils.validate_region(location)
            self._location = location
        if experiment:
            logging.warning("Experiments currently not supported.")
            self._experiment = experiment
        if staging_bucket:
            self._staging_bucket = staging_bucket
        if credentials:
            self._credentials = credentials

    @property
    def project(self) -> str:
        """Default project."""
        if self._project:
            return self._project

        _, project_id = google.auth.default()
        return project_id

    @property
    def location(self) -> str:
        """Default location."""
        return self._location if self._location else utils.DEFAULT_REGION

    @property
    def experiment(self) -> Optional[str]:
        """Default experiment, if provided."""
        return self._experiment

    @property
    def staging_bucket(self) -> Optional[str]:
        """Default staging bucket, if provided."""
        return self._staging_bucket

    @property
    def credentials(self) -> Optional[auth_credentials.Credentials]:
        """Default credentials, if provided."""
        return self._credentials

    def get_client_options(
        self,
        location_override: Optional[str] = None,
        prediction_client: Optional[bool] = False,
    ) -> Dict[str, str]:
        """Creates client_options for GAPIC service client using location and client type.

        Args:
            location_override (str):
                Set this parameter to get client options for a location different
                from location set by initializer. Must be a GCP region
                supported by AI Platform (Unified).

            prediction_client (bool):
                True if service client is a PredictionServiceClient, otherwise
                defaults to False. This is used to provide a prediction-specific
                API endpoint.

        Returns:
            clients_options (dict):
                A dictionary containing client_options with one key, for example
                { "api_endpoint": "us-central1-aiplatform.googleapis.com" } or
                { "api_endpoint": "asia-east1-prediction-aiplatform.googleapis.com" }
        """
        if not (self.location or location_override):
            raise ValueError(
                "No location found. Provide or initialize SDK with a location."
            )

        region = self.location if not location_override else location_override
        region = region.lower()
        prediction = "prediction-" if prediction_client else ""

        utils.validate_region(region)

        client_options = {"api_endpoint": f"{region}-{prediction}{utils.PROD_API_ENDPOINT}"}

        return client_options


# singleton to store init parameters: ie, aiplatform.init(project=..., location=...)
singleton = Init()
