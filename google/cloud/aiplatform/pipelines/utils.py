import inspect
from typing import Any, Callable, Tuple

from google.cloud import aiplatform
import kfp
from kfp import components

# prefix for keyword arguments to separate constructor and method args 
INIT_KEY = 'init'
METHOD_KEY = 'method'

# map of MB SDK type to Metadata type
resource_to_metadata_type = {
    aiplatform.Dataset: "Dataset",
    aiplatform.Model: "Model",
    aiplatform.Endpoint: "Artifact",
    aiplatform.BatchPredictionJob: "Artifact"
}

def map_resource_to_metadata_type(mb_sdk_type) -> Tuple[str, str]:
    """Maps an MB SDK type to the Metadata type.

    Returns:
        Tuple of component parameter name and metadata type 
    """
    for key in resource_to_metadata_type.keys():
        if issubclass(mb_sdk_type, key):
            return key.__name__.split('.')[-1].lower(), resource_to_metadata_type[key]

def is_resource_name_parameter_name(param_name):
    return param_name != 'display_name' and param_name.endswith('_name')

# These parameters are removed from MB SDK Methods
params_to_remove = set(["self", "credentials"])
def filter_signature(signature: inspect.Signature,
    is_init_signature=False,
    self_type=None):
    """Removes unused params from signature.

    Args:
        signature (inspect.Signature) Model Builder SDK Method Signature.
    Returns:
        Signature with parameters removed. 
    """
    new_params = []
    for param in signature.parameters.values():
        if param.name not in params_to_remove:
            # change resource name signatures to resource types
            # to enforce metadata entry
            if is_init_signature and is_resource_name_parameter_name(param.name):
                new_params.append(inspect.Parameter(
                    name=param.name[:-len('_name')],
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=param.default,
                    annotation=self_type))
            else:
                new_params.append(param)

    return inspect.Signature(
        parameters=new_params, 
        return_annotation=signature.return_annotation)

def resolve_annotation(annotation: Any) -> Any:
    """Resolves annotation type against a MB SDK type.

    Use this for Optional, Union, Forward References
    """

    # handle forward refernce string
    if isinstance(annotation, str):
        resolved_annotation = getattr(aiplatform, annotation, None)
        if resolved_annotation:
            return resolved_annotation
    return annotation

def get_parameter_type(signature: inspect.Signature, param_name) -> Any:
    """Returns the expected type of the input parameter.

    Args:
        signature (inspect.Signature): Model Builder SDK Method Signature.
        param_name (str): Name of parameter to get type
    Returns:
        Signature with parameters removed. 
    """
    # TODO(handle Union types)
    # TODO(handle Forward references)
    return signature.parameters[param_name].annotation


def convert_method_to_component(method: Callable, should_serialize_init=False):
    method_name = method.__name__
    method_signature = inspect.signature(method)

    # get class name and constructor signature
    if inspect.ismethod(method):
        cls = method.__self__
        cls_name = cls.__name__
        init_signature = inspect.signature(method.__self__.__init__)
    else:
        cls = getattr(inspect.getmodule(method),
                      method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        cls_name = cls.__name__
        init_signature = inspect.signature(cls.__init__)

    # remove unusers parameters
    method_signature = filter_signature(method_signature)
    init_signature = filter_signature(init_signature, is_init_signature=True, self_type=cls)

    init_arg_names = set(init_signature.parameters.keys()) if should_serialize_init else set([])

    output_type = resolve_annotation(method_signature.return_annotation)
    output_metadata_name, output_metadata_type = map_resource_to_metadata_type(output_type) 

    def make_args(sa):
        additional_args = []
        for key, args in sa.items():
            for arg_key, value in args.items():
                additional_args.append(f"    - --{key}.{arg_key}={value}")
        return '\n'.join(additional_args)

    def f(**kwargs):
        inputs = ["inputs:"]
        input_args = []
        input_kwargs = {}

        serialized_args = {"init": {}, "method": {}}

        init_kwargs = {}
        method_kwargs = {}

        for key, value in kwargs.items():
            if key in init_arg_names:
                prefix_key = "init"
                init_kwargs[key] = value
                signature = init_signature
            else:
                prefix_key = "method"
                method_kwargs[key] = value
                signature = method_signature
            
            if isinstance(value, kfp.dsl._pipeline_param.PipelineParam):
                param_type = get_parameter_type(signature, key)
                metadata_type = map_resource_to_metadata_type(param_type)[1]
                inputs.append(f"- {{name: {key}, type: {metadata_type}}}")
                input_args.append('\n'.join([
                    f'    - --{prefix_key}.{key}',
                    f'    - {{inputUri: {key}}}']))
#""" % (f'{prefix_key}.{key}', key))
                input_kwargs[key] = value
            else:
                serialized_args[prefix_key][key] = value

        # validate parameters
        if should_serialize_init:
            init_signature.bind(**init_kwargs)
        method_signature.bind(**method_kwargs)


        inputs = "\n".join(inputs) if len(inputs) > 1 else ''
        input_args = "\n".join(input_args) if input_args else ''
        component_text = "\n".join([
        f'name: {cls_name}-{method_name}',
        f'{inputs}',
        'outputs:',
        f'- {{name: {output_metadata_name}, type: {output_metadata_type}}}',
        'implementation:',
        '  container:',
        '    image: gcr.io/sashaproject-1/mb_sdk_component:latest',
        '    command:',
        '    - python3',
        '    - remote_runner.py',
        f'    - --cls_name={cls_name}',
        f'    - --method_name={method_name}',
        f'{make_args(serialized_args)}',
        '    args:',
        '    - --resource_name_output_uri',
        f'    - {{outputUri: {output_metadata_name}}}',
        f'{input_args}'
        ])

        print(component_text)

        return components.load_component_from_text(component_text)(**input_kwargs)

    f.__doc__ = method.__doc__
    f.__signature__ = method_signature

    return f
