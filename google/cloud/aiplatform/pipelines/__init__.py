from google.cloud import aiplatform
from google.cloud.aiplatform.pipelines import utils

DatasetCreateOp = utils.convert_method_to_component(aiplatform.Dataset.create)

CustomContainerTrainingJobRunOp = utils.convert_method_to_component(aiplatform.CustomContainerTrainingJob.run,
                                                  should_serialize_init=True)

ModelDeployOp = utils.convert_method_to_component(aiplatform.Model.deploy, should_serialize_init=True)
