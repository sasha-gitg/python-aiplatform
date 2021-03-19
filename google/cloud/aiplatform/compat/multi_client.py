from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    client as v1beta1_prediction_service_client,
)

from google.cloud.aiplatform_v1.services.prediction_service import (
    client as v1_prediction_service_client,
)


# can possibly be a base class
class ClientWithOverride:
    client_map = {
        'v1': v1_prediction_service_client.PredictionServiceClient,
        'v1beta1': v1beta1_prediction_service_client.PredictionServiceClient
    }
    default = 'v1'

    def __init__(self, *args, **kwargs):
        self._clients = {key: client(*args, **kwargs) for key, client in self.client_map.items()}

    def __getattr__(self, attr):
        return getattr(self._client[self.default], attr)
    
    def select_version(self, version: str):
        return self._clients[version]
    
# Usage
client = ClientWithOverride()
client.predict() # no change for most APIs

client.select_version('v1beta1').explain() #explicitly have to call when selecting a non default version

if explain:
    client.select_version('v1beta1').batch_predict()
else:
    client.batch_predict()
