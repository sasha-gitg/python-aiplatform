from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    client as v1beta1_prediction_service_client,
)

from google.cloud.aiplatform_v1.services.prediction_service import (
    client as v1_prediction_service_client,
)


# can possibly be a base class
class ClientWithOverride:
    base_client = v1_prediction_service_client.PredictionServiceClient
    override_client = v1beta1_prediction_service_client.PredictionServiceClient
    override_apis = ("explain",)

    def __init__(self, *args, **kwargs):
        self._client = self.base_client(*args, **kwargs)
        self._override_client = self.override_client(*args, **kwargs)
        self._override_apis = set(self.override_apis)

    def __getattr__(self, attr):
        if attr in self._override_apis:
            return getattr(self._override_client, attr)
        return getattr(self._client, attr)
