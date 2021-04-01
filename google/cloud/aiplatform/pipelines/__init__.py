from google.cloud import aiplatform
from google.cloud.aiplatform.pipelines import utils

DatasetCreateOp = utils.convert_method_to_component(aiplatform.Dataset.create)

# needs to handle sequence of string outputs
# DatasetExportDataOp = utils.convert_method_to_component(
# 	aiplatform.Dataset.export_data,
# 	should_serialize_init=True)


ImageDatasetCreateOp = utils.convert_method_to_component(
	aiplatform.ImageDataset.create)

# needs to handle optional outputs
# CustomContainerTrainingJobRunOp = utils.convert_method_to_component(
# 	aiplatform.CustomContainerTrainingJob.run,
#     should_serialize_init=True)

AutoMLImageTrainingJobRunOp = utils.convert_method_to_component(
	aiplatform.AutoMLImageTrainingJob.run,
	should_serialize_init=True)

ModelDeployOp = utils.convert_method_to_component(
	aiplatform.Model.deploy,
	should_serialize_init=True)

ModelUploadOp = utils.convert_method_to_component(
	aiplatform.Model.upload)
