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

import datetime
import functools
import logging
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, Optional, Sequence


from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import datasets
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import models
from google.cloud.aiplatform import schema
from google.cloud.aiplatform import utils
from google.cloud.aiplatform_v1beta1 import AcceleratorType
from google.cloud.aiplatform_v1beta1 import CustomJobSpec
from google.cloud.aiplatform_v1beta1 import FractionSplit
from google.cloud.aiplatform_v1beta1 import GcsDestination
from google.cloud.aiplatform_v1beta1 import InputDataConfig
from google.cloud.aiplatform_v1beta1 import JobServiceClient
from google.cloud.aiplatform_v1beta1 import Model
from google.cloud.aiplatform_v1beta1 import ModelContainerSpec
from google.cloud.aiplatform_v1beta1 import PipelineServiceClient
from google.cloud.aiplatform_v1beta1 import PipelineState
from google.cloud.aiplatform_v1beta1 import TrainingPipeline
from google.cloud import storage
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.rpc import code_pb2

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)

PIPELINE_COMPLETE_STATES = set(
    [PipelineState.PIPELINE_STATE_SUCCEEDED,
    PipelineState.PIPELINE_STATE_FAILED,
    PipelineState.PIPELINE_STATE_CANCELLED,
    PipelineState.PIPELINE_STATE_PAUSED])


def _timestamped_gcs_dir(root_gcs_path: str, dir_name_prefix: str):
    timestamp = datetime.datetime.now().isoformat(sep="-", timespec="milliseconds")
    dir_name = "-".join([dir_name_prefix, timestamp])
    if root_gcs_path.endswith('/'):
        root_gcs_path = root_gcs_path[:-1]
    gcs_path = '/'.join([root_gcs_path, dir_name])
    if not gcs_path.startswith('gs://'):
        return 'gs://' + gcs_path


def _timestamped_copy_to_gcs(
    local_file_path: str,
    gcs_dir: str,
    project: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> str:
    """Copies a local file to a GCS path.

    The file copied to GCS is the name of the local file prepended with an
    "aiplatform-{timestamp}-" string.

    Args:
        local_file_path (str): Required. Local file to copy to GCS.
        gcs_dir (str):
            Required. The GCS directory to copy to.
        project (str):
            Project that contains the staging bucket. Default will be used if not
            provided. Model Builder callers should pass this in.
        credentials (auth_credentials.Credentials):
            Custom credentials to use with bucket. Model Builder callers should pass
            this in.
    Returns:
        gcs_path (str): The path of the copied file in gcs.
    """
    if gcs_dir.startswith("gs://"):
        gcs_dir = gcs_dir[5:]
    if gcs_dir.endswith("/"):
        gcs_dir = gcs_dir[:-1]

    gcs_parts = gcs_dir.split("/", 1)
    gcs_bucket = gcs_parts[0]
    gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

    local_file_name = pathlib.Path(local_file_path).name
    timestamp = datetime.datetime.now().isoformat(sep="-", timespec="milliseconds")
    blob_path = "-".join(["aiplatform", timestamp, local_file_name])

    if gcs_blob_prefix:
        blob_path = "/".join([gcs_blob_prefix, blob_path])

    # TODO(b/171202993) add user agent
    client = storage.Client(project=project, credentials=credentials)
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file_path)

    gcs_path = "".join(["gs://", "/".join([blob.bucket.name, blob.name])])
    return gcs_path


def _get_python_executable() -> str:
    """Returns Python executable.

    Raises:
        EnvironmentError if Python executable is not found.
    Returns:
        Python executable to use for setuptools packaging.
    """

    python_executable = sys.executable

    if not python_executable:
        raise EnvironmentError("Cannot find Python executable for packaging.")
    return python_executable


class _TrainingScriptPythonPackager:
    """Converts a Python script into Python package suitable for aiplatform training.

    Copies the script to specified location.

    Class Attributes:
        _TRAINER_FOLDER: Constant folder name to build package.
        _ROOT_MODULE: Constant root name of module.
        _TEST_MODULE_NAME: Constant name of module that will store script.
        _SETUP_PY_VERSION: Constant version of this created python package.
        _SETUP_PY_TEMPLATE: Constant template used to generate setup.py file.
        _SETUP_PY_SOURCE_DISTRIBUTION_CMD:
            Constant command to generate the source distribution package.

    Attributes:
        script_path: local path of script to package
        requirements: list of Python dependencies to add to package

    Usage:

    packager = TrainingScriptPythonPackager('my_script.py', ['pandas', 'pytorch'])
    gcs_path = packager.package_and_copy_to_gcs(
        gcs_staging_dir='my-bucket',
        project='my-prject')
    module_name = packager.module_name

    The package after installed can be executed as:
    python -m aiplatform_custom_trainer_script.task

    """

    _TRAINER_FOLDER = "trainer"
    _ROOT_MODULE = "aiplatform_custom_trainer_script"
    _TASK_MODULE_NAME = "task"
    _SETUP_PY_VERSION = "0.1"

    _SETUP_PY_TEMPLATE = """from setuptools import find_packages
from setuptools import setup

setup(
    name='{name}',
    version='{version}',
    packages=find_packages(),
    install_requires=({requirements}),
    include_package_data=True,
    description='My training application.'
)"""

    _SETUP_PY_SOURCE_DISTRIBUTION_CMD = (
        "{python_executable} setup.py sdist --formats=gztar"
    )

    # Module name that can be executed during training. ie. python -m
    module_name = f"{_ROOT_MODULE}.{_TASK_MODULE_NAME}"

    def __init__(self, script_path: str, requirements: Optional[Sequence[str]] = None):
        """Initializes packager.

        Args:
            script_path (str): Required. Local path to script.
            requirements (Sequence[str]):
                List of python packages dependencies of script.
        """

        self.script_path = script_path
        self.requirements = requirements or []

    def make_package(self, package_directory: str) -> str:
        """Converts script into a Python package suitable for python module execution.

        Args:
            package_directory (str): Directory to build package in.
        Returns:
            source_distribution_path (str): Path to built package.
        Raises:
            RunTimeError if package creation fails.
        """
        # The root folder to builder the package in
        package_path = pathlib.Path(package_directory)

        # Root directory of the package
        trainer_root_path = package_path / self._TRAINER_FOLDER

        # The root module of the python package
        trainer_path = trainer_root_path / self._ROOT_MODULE

        # __init__.py path in root module
        init_path = trainer_path / "__init__.py"

        # The module that will contain the script
        script_out_path = trainer_path / f"{self._TASK_MODULE_NAME}.py"

        # The path to setup.py in the package.
        setup_py_path = trainer_root_path / "setup.py"

        # The path to the generated source distribution.
        source_distribution_path = (
            trainer_root_path
            / "dist"
            / f"{self._ROOT_MODULE}-{self._SETUP_PY_VERSION}.tar.gz"
        )

        trainer_root_path.mkdir()
        trainer_path.mkdir()

        # Make empty __init__.py
        with init_path.open("w"):
            pass

        # Format the setup.py file.
        setup_py_output = self._SETUP_PY_TEMPLATE.format(
            name=self._ROOT_MODULE,
            requirements=",".join(f'"{r}"' for r in self.requirements),
            version=self._SETUP_PY_VERSION,
        )

        # Write setup.py
        with setup_py_path.open("w") as fp:
            fp.write(setup_py_output)

        # Copy script as module of python package.
        shutil.copy(self.script_path, script_out_path)

        # Run setup.py to create the source distribution.
        setup_cmd = self._SETUP_PY_SOURCE_DISTRIBUTION_CMD.format(
            python_executable=_get_python_executable()
        ).split()

        p = subprocess.Popen(
            args=setup_cmd,
            cwd=trainer_root_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, error = p.communicate()

        # Raise informative error if packaging fails.
        if p.returncode != 0:
            raise RuntimeError(
                "Packaging of training script failed with code %d\n%s \n%s"
                % (p.returncode, output.decode(), error.decode())
            )

        return str(source_distribution_path)

    def package_and_copy(self, copy_method: Callable[[str], str]) -> str:
        """Packages the script and executes copy with given copy_method.

        Args:
            copy_method Callable[[str], str]
                Takes a string path, copies to a desired location, and returns the
                output path location.
        Returns:
            output_path str: Location of copied package.
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            source_distribution_path = self.make_package(tmpdirname)
            output_location = copy_method(source_distribution_path) 
            _LOGGER.info("Training script copied to:\n%s." % output_location)
            return output_location

    def package_and_copy_to_gcs(
        self,
        gcs_staging_dir: str,
        project: str = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> str:
        """Packages script in Python package and copies package to GCS bucket.

        Args
            gcs_staging_dir (str): Required. GCS Staging directory.
            project (str): Required. Project where GCS Staging bucket is located.
            credentials (auth_credentials.Credentials):
                Optional credentials used with GCS client.
        Returns:
            GCS location of Python package.
        """

        copy_method = functools.partial(
            _timestamped_copy_to_gcs,
            gcs_dir=gcs_staging_dir,
            project=project,
            credentials=credentials,
        )
        return self.package_and_copy(copy_method=copy_method)


class CustomJob(base.AiPlatformResourceNoun):
    client_class = JobServiceClient
    _is_client_prediction_client = False

    def __init__(self, custom_job_name, project=None, location=None, credentials=None):
        super().__init__(project=project, location=location, credentials=credentials)
        self._gca_resource = self.api_client.get_custom_job(name=custom_job_name)


class CustomTrainingJob(base.AiPlatformResourceNoun):
    """Class to launch a Custom Training Job in AI Platform using a script.

    Takes a training implementation as a python script and executes that script
    in Cloud AI Platform Training. 
    """

    client_class = PipelineServiceClient
    _is_client_prediction_client = False

    # TODO() add remainder of model optional arguments
    def __init__(
        self,
        display_name: str,
        script_path: str,
        container_uri: str,
        requirements: Optional[Sequence[str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        staging_bucket: Optional[str] = None,
        model_serving_container_image_uri: Optional[str] = None,
        model_serving_container_predict_route: Optional[str] = None,
        model_serving_container_health_route: Optional[str] = None,
    ):
        """Constructs a Custom Training Job.

        Args:
            project (str):
                Optional project to retrieve model from. If not set, project set in
                aiplatform.init will be used.
            location (str):
                Optional location to retrieve model from. If not set, location set in
                aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional credentials to use to retrieve the model.
        """
        super().__init__(project=project, location=location, credentials=credentials)
        self._display_name = display_name
        self._script_path = script_path
        self._container_uri = container_uri
        self._requirements = requirements
        self._staging_bucket = staging_bucket
        self._project = project
        self._credentials = credentials
        self._model_serving_container_image_uri = model_serving_container_image_uri
        self._model_serving_container_predict_route = model_serving_container_predict_route
        self._model_serving_container_health_route = model_serving_container_health_route
        self._gca_resource = None



    # TODO() add filter split, training_pipline.FilterSplit
    # TODO() add predefined filter split, training_pipeline.PredfinedFilterSplit
    # TODO() add timestamp split, training_pipeline.TimestampSplit
    # TODO() add scheduling, custom_job.Scheduling
    def run(
        self,
        dataset: Optional[datasets.Dataset] = None,
        base_output_dir: Optional[str] = None,
        args: Optional[Dict[str, str]] = None,
        replica_count: str=0,
        machine_type: str="n1-standard-4",
        accelerator_type: str='ACCELERATOR_TYPE_UNSPECIFIED',
        accelerator_count: int=0,
        model_display_name: Optional[str] = None,

        training_fraction_split: float = 0.8,
        validation_fraction_split: float = 0.1,
        test_fraction_split: float = 0.1,


    ) -> models.Model:
        """Runs the custom training job.

        Any of ``training_fraction_split``, ``validation_fraction_split`` and
        ``test_fraction_split`` may optionally be provided, they must sum to up to 1. If
        the provided ones sum to less than 1, the remainder is assigned to sets as
        decided by AI Platform.If none of the fractions are set, by default roughly 80%
        of data will be used for training, 10% for validation, and 10% for test.

        Args:

            base_output_dir (str):
                GCS output directory of job. If not provided a 
                timestamped directory in the staging directory will be used.
            accelerator_type (str):
                Hardware accelerator type. One of ACCELERATOR_TYPE_UNSPECIFIED,
                NVIDIA_TESLA_K80, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100, NVIDIA_TESLA_P4,
                NVIDIA_TESLA_T4, TPU_V2, TPU_V3
            training_fraction_split (float):
                The fraction of the input data that is to be
                used to train the Model.
            validation_fraction_split (float):
                The fraction of the input data that is to be
                used to validate the Model.
            test_fraction_split (float):
                The fraction of the input data that is to be
                used to evaluate the Model.

        Returns:
            The trainer model resource.
        """

        if self._has_run:
            raise RunTimeError("Custom Training has already run.")


        def flatten_args(args):
            return [f"--{key}={value}" for key, value in args.items()]


        python_packager = _TrainingScriptPythonPackager(
            script_path=self._script_path,
            requirements=self._requirements
        )

        package_gcs_uri = python_packager.package_and_copy_to_gcs(
            gcs_staging_dir = self._staging_bucket or initializer.global_config.staging_bucket,
            project = self._project or initializer.global_config.project,
            credentials = self._credentials or initializer.global_config.credentials,
        )

        base_output_dir = base_output_dir or _timestamped_gcs_dir(
            self._staging_bucket or initializer.global_config.staging_bucket,
            'aiplatform-custom-training')

        worker_pool_spec= {
                "replicaCount": replica_count,
                "machineSpec": {
                  "machineType": machine_type
                },
                "pythonPackageSpec": {
                  "executorImageUri": self._container_uri,
                  "pythonModule": python_packager.module_name,
                  "packageUris": [package_gcs_uri]
                },
              }


        # validate accelerator type
        accelerator_enum = getattr(AcceleratorType, accelerator_type,
                AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED)

        if accelerator_enum != AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED:
            worker_pool_spec["machineSpec"]["acceleratorType"] = accelerator_type
            worker_pool_spec["machineSpec"]["acceleratorCount"] = accelerator_count

        if args:
            worker_pool_spec["pythonPackageSpec"]["args"] = flatten_args(args)


        # Create fraction split spec
        fraction_split = FractionSplit(
            training_fraction=training_fraction_split,
            validation_fraction=validation_fraction_split,
            test_fraction=test_fraction_split
        )

        
        managed_model = None
        # Validate if model should be uploaded and create model payload
        if model_display_name:

            # if args need for model is incomplete
            # TODO (b/162273530) lift requirement for predict/health route when
            # validation lifted and move these args down
            if not all([self._model_serving_container_image_uri,
            self._model_serving_container_predict_route,
            self._model_serving_container_health_route]):
                
                raise ValueError("""model_display_name was provided but 
                    model_serving_container_image_uri,
                    model_serving_container_predict_route, and
                    model_serving_container_health_route were not provided when this 
                    custom pipeline was constructed.
                    """)


            container_spec = ModelContainerSpec(
                image_uri=self._model_serving_container_image_uri,
                predict_route=self._model_serving_container_predict_route,
                health_route=self._model_serving_container_health_route
            )

            managed_model = Model(
                display_name=model_display_name,
                container_spec=container_spec
            )

        input_data_config=None
        if dataset:
            #create input data config
            input_data_config = InputDataConfig(
                fraction_split=fraction_split,
                dataset_id=dataset.name,
                gcs_destination=GcsDestination(output_uri_prefix=base_output_dir))
            
        # create training pipeline
        training_pipeline = TrainingPipeline(
            display_name = self._display_name,
            training_task_definition = schema.training_job.definition.custom_task,
            training_task_inputs= json_format.ParseDict(
                {
                    "workerPoolSpecs": [worker_pool_spec],
                    'baseOutputDirectory': {"output_uri_prefix": base_output_dir}
                }, Value()),
            model_to_upload=managed_model,
            input_data_config=input_data_config
        )

        training_pipeline = self.api_client.create_training_pipeline(
                parent=initializer.global_config.common_location_path(
                    self.project, self.location),
                training_pipeline=training_pipeline)

        print(training_pipeline)

        self._gca_resource = training_pipeline

        _LOGGER.info("View Training:\n%s" % self._dashboard_uri())
        _LOGGER.info("Training Output directory:\n%s " % base_output_dir)

        return self.get_model()


    def _sync_gca_resource(self):
        self._gca_resource = self.api_client.get_training_pipeline(
                name=self.resource_name
            )

    def _block_until_complete(self):
        wait = 5 # start at five seconds
        max_wait = 60 * 1  # 1 minute wait
        multiplier = 2 # scale wait by 2 every iteration

        while self.state not in PIPELINE_COMPLETE_STATES:
            self._sync_gca_resource()
            time.sleep(wait)
            _LOGGER.info("Training %s current state: %s" %
                (self._gca_resource.name, self._gca_resource.state))
            wait = min(wait * multiplier, max_wait)

        self._log_failure()

        if self._gca_resource.model_to_upload and not self.is_failed:
            _LOGGER.info("Model available at %s" %
                self._gca_resource.model_to_upload.name)

    def _log_failure(self):
        if self._gca_resource.error.code != code_pb2.OK:
            _LOGGER.error("Training failed with:\n%s" % self._gca_resource.error)

    # TODO(asobran) add project and location
    @classmethod
    def get(cls, training_pipline_name: str):
        self = cls.__new__(cls)
        self.api_client = cls._instantiate_client()
        self._gca_resource = self.api_client.get_training_pipeline(
            name=training_pipline_name)
        return self


    # TODO (asobran) take project and location
    @classmethod
    def list(cls):
        def instantiate_pipeline(gca_resource, api_client):
            # TODO (asobran) populate with additional attributes 
            self = cls.__new__(cls)
            self._gca_resource = gca_resource
            self.api_client = api_client
            return self

        api_client = cls._instantiate_client()

        return [instantiate_pipeline(pipeline, api_client) for pipeline in
        api_client.list_training_pipelines(
            parent=initializer.global_config.common_location_path())]

    
    def get_model(self) -> Optional[models.Model]:
        self._block_until_complete()
        
        if not self._gca_resource.model_to_upload:
            raise RuntimeError(
                f"Training Pipeline {self.resource_name} is not configured to upload a "
                "Model. Create the Training Pipeline with "
                "model_serving_container_image_uri and model_display_name passed in."
            )


        if self.is_failed:
            raise RuntimeError(
                f"Training Pipeline {self.resource_name} failed. No model available.")

        model_name = getattr(self._gca_resource.model_to_upload, 'name', None)

        if model_name:
            # TODO(asobran) update when resource name supported by Model
            return models.Model(
                self._gca_resource.model_to_upload.name.split('/')[-1])


    @property
    def _has_run(self) -> bool:
        return self._gca_resource is not None

    def _assert_has_run(self):
        if not self._has_run:
            raise RuntimeError(
                "TrainingPipeline has not been launched. You must run this"
                " TrainingPipeline using TrainingPipeline.run. ")

    @property
    def state(self) -> PipelineState:
        return self._gca_resource.state

    @property
    def is_succeeded(self) -> bool:
        self._assert_has_run()
        return self.state == PipelineState.PIPELINE_STATE_SUCCEEDED

    @property
    def is_failed(self) -> bool:
        self._assert_has_run()
        return self.state == PipelineState.PIPELINE_STATE_FAILED
    

    @property
    def backing_custom_job(self) -> Optional[CustomJob]:
        self._assert_has_run()
        self._get_training_pipeline()
        training_task_metadata = self._gca_resource.training_task_metadata
        if training_task_metadata:
            custom_job_name = training_task_metadata.get('backingCustomJob')
            if custom_job_name:
                # TODO(asobran) pass in pipeline project and location
                return CustomJob(custom_job_name)


    def _dashboard_uri(self):
        fields = utils.extract_fields_from_resource_name(self.resource_name)
        url = f'https://console.cloud.google.com/ai/platform/locations/{fields.location}/training/{fields.id}?project={fields.project}'
        return url


class AutoMLTablesTrainingJob(base.AiPlatformResourceNoun):
    pass
