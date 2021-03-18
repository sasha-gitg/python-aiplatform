import inspect
from kfp import components
import kfp

INIT_KEY = 'init'
METHOD_KEY = 'method'

def convert_method_to_component(method, should_serialize_init=False):
    method_name = method.__name__

    if inspect.ismethod(method):
        cls_name = method.__self__.__name__
        init_signature = inspect.signature(method.__self__.__init__)
    else:
        cls = getattr(inspect.getmodule(method),
                      method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        cls_name = cls.__name__
        init_signature = inspect.signature(cls.__init__)

    init_arg_names = set(init_signature.parameters.keys()) if should_serialize_init else set([])

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

        for key, value in kwargs.items():
            prefix_key = "init" if key in init_arg_names else "method"
            if isinstance(value, kfp.dsl._pipeline_param.PipelineParam):
                name = key
                inputs.append("- {name: %s, type: Artifact}" % (name))
                input_args.append("""
    - --%s
    - {inputUri: %s}
""" % (f'{prefix_key}.{key}', key))
                input_kwargs[key] = value
            else:
                serialized_args[prefix_key][key] = value

        inputs = "\n".join(inputs) if len(inputs) > 1 else ''
        input_args = "\n".join(input_args) if input_args else ''
        component_text = """
name: %s-%s
%s
outputs:
- {name: resource_name_output, type: Artifact}
implementation:
  container:
    image: gcr.io/sashaproject-1/mb_sdk_component:latest
    command:
    - python3
    - remote_runner.py
    - --cls_name=%s
    - --method_name=%s
%s
    args:
    - --resource_name_output_uri
    - {outputUri: resource_name_output}
%s
""" % (cls_name,
       method_name,
       inputs,
       cls_name,
       method_name,
       make_args(serialized_args),
       input_args)

        print(component_text)

        return components.load_component_from_text(component_text)(**input_kwargs)

    return f
