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

import abc
from concurrent import futures
import datetime
import functools
import inspect
import logging
import pandas as pd
import sys
import threading
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import proto

from google.api_core import operation
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import encryption_spec as gca_encryption_spec
from google.cloud import aiplatform


class VertexModel:

    _data_serialization_mapping = {
        pd.DataFrame : (deserialize_data_in_memory, serialize_data_in_memory)
    }

    # Default to local training on creation, at least for this prototype.

    """ Parent class that users can extend to use the Vertex AI SDK
    
    class MyModel(VertexModel):
    
        def __init__(self):
            super().__init__()
            
        def fit(self):
            print('my_fit')
    
    my_model = VertexModel()
    
    my_model.training_mode = 'cloud'
    my_model.training_mode = 'local'
    
    """
    
    
    
    def __init__(self):
        self.training_mode = 'local'
        
        # if self.training_mode == 'cloud':
        
        # ensure self.fit is referencing subclass' MyModel.fit instead of VertexModel.fit (abcmethod)
        # print(self) __main__.MyModel
        # print(inspect.getsource(self.fit))
        self.fit = vertex_function_wrapper(self.fit)
        self.predict = vertex_function_wrapper(self.predict)
        self.batch_predict = vertex_function_wrapper(self.batch_predict)
        self.eval = vertex_function_wrapper(self.eval)

    # capture as another class and reference that class
    #  make serialization private to signal to the customer to not use this method
    def _serialize_data_in_memory(self, artifact_uri, obj: pd.Dataframe, temp_dir: str, dataset_type: str):
        """ Provides out-of-the-box serialization for input """

        # Designate csv path and write the pandas DataFrame to the path
        # Convention: file name is my_training_dataset, my_test_dataset, etc.
        path_to_csv = temp_dir + "/" + "my_" + dataset_type + "_dataset.csv"
        obj.to_csv(path_to_csv)

        gcs_bucket, gcs_blob_prefix = extract_bucket_and_prefix_from_gcs_path(artifact_uri)

        local_file_name = pathlib.Path(path_to_csv).name
        blob_path = local_file_name

        if gcs_blob_prefix:
            blob_path = "/".join([gcs_blob_prefix, blob_path])

        client = storage.Client(project=initializer.global_config.project, 
                                credentials=initializer.global_config.credentials))

        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(path_to_csv)

        gcs_path = "".join(["gs://", "/".join([blob.bucket.name, blob.name])])
        return gcs_path

    def deserialize_data_in_memory(cls, artifact_uri):
        """ Provides out-of-the-box deserialization after training and prediction is complete """
        
        gcs_bucket, gcs_blob = utils.extract_bucket_and_prefix_from_gcs_path(
            artifact_uri
        )

        client = storage.Client(project=initializer.global_config.project, 
                                credentials=initializer.global_config.credentials)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_blob)

        # Incrementally download the CSV file until the header is retrieved
        first_new_line_index = -1
        start_index = 0
        increment = 1000
        line = ""

        try:
            logger = logging.getLogger("google.resumable_media._helpers")
            logging_warning_filter = utils.LoggingFilter(logging.INFO)
            logger.addFilter(logging_warning_filter)

            while first_new_line_index == -1:
                line += blob.download_as_bytes(
                    start=start_index, end=start_index + increment
                ).decode("utf-8")

                first_new_line_index = line.find("\n")
                start_index += increment

            header_line = line[:first_new_line_index]

            # Split to make it an iterable
            header_line = header_line.split("\n")[:1]

            csv_reader = csv.reader(header_line, delimiter=",")
        except (ValueError, RuntimeError) as err:
            raise RuntimeError(
                "There was a problem extracting the headers from the CSV file at '{}': {}".format(
                    gcs_csv_file_path, err
                )
            )
        finally:
            logger.removeFilter(logging_warning_filter)

        # Return a pandas DataFrame read from the csv in the cloud
        return pandas.read_csv(next(csv_reader))

    def serialize_remote_dataloader:
        # writes the referenced data to the run-time bucket
        pass

    def deserialize_remote_dataloader:
        # read the data from a run-time bucket 
        # and reformat to a DataLoader
        pass

    def serialize_local_dataloader:
        # finds the local source, and copies 
        # data to the user-designated staging bucket
        pass

    def deserialize_local_dataloader:
        # read the data from user-designated staging bucket and
        # reformat to a DataLoader
        pass

    def serialize_dataloader:
        # introspect to determine which method is called
        pass

    def deserialize_dataloader:
        # introspect to determine which method is called
        pass

    @abc.abstractmethod
    def fit(self, data, epochs, learning_rate, dataset: pd.DataFrame):
        """ Train model. """
        pass

    @abc.abstractmethod
    def predict(self, data):
        """ Make predictions on training data. """
        pass

    @abc.abstractmethod
    def batch_predict(self, data, target):
        """ Make predictions on training data. """
        pass

    @abc.abstractmethod
    def eval(self, data):
        """ Evaluate model. """
        pass

    def vertex_function_wrapper(method):

        # how to determine function type within another function??
        def f(*args, **kwargs):
            dataset = kwargs['dataset']

            # serializing parameters (data) to the method
            serializer = fit_method.__self__.__class__._data_serialization_mapping[type(dataset)][1]
            serializer(dataset, staging_bucket + 'dataset.csv', args[1], '~/temp_dir', 'training')
            
            # serialize the code in the class to be packaged and copied to GCS (as separate class)

            """            
            # Edit run of job here?
            job = aiplatform.CustomTrainingJob(...)
            job.run(
                args = ['dataset',  staging_bucket + 'dataset.csv',
                        'dataset_type', dataset] 
            ) 
            """

        if (fit_method.__name__ == 'fit'):
            f(inspect.signature(fit_method))
           

        # TODO: wrapper for predict and/or eval (could be the same function)

        # def p(*args, **kwargs):
        
        
# Source example
import pandas

class Model:
    
    def __init__(self):
        self.x = 10
        
    def f(self):
        print(pandas.DataFrame)

m = Model()

m.f()

m.x

import inspect

m.__class__

class SourceMaker:
    
    def __init__(self, cls_name: str):
        self.source = ["class {}".format(cls_name)]
        
    def add_method(self, method_str: str):
        self.source.extend(method_str.split('\n'))
        
    # add a method
    # append "m = Model() \nm.fit"
        


def make_class_source(obj):
    source_maker = SourceMaker(obj.__class__.__name__)
    
    for key, value in inspect.getmembers(m):
        #print(key, value)
        if inspect.ismethod(value): 
            source_maker.add_method(inspect.getsource(value))
    return source_maker.source

print('\n'.join(make_class_source(m)))

inspect.getsource(m.f).split('\n')

class A(dict):
    pass

# will get superclasses
inspect.getmro(A)
