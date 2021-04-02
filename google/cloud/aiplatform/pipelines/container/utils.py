import inspect
from typing import Any, Callable, Optional, Tuple, Union

from google.cloud import aiplatform


def get_forward_reference(annotation) -> Optional[aiplatform.base.AiPlatformResourceNoun]:
    """Resolves forward references to AiPlatform Class."""

    def get_aiplatform_class_by_name(_annotation):
        """Resolves str annotation to AiPlatfrom Class."""
        if isinstance(_annotation, str):
            return getattr(aiplatform, _annotation, None)

    ai_platform_class = get_aiplatform_class_by_name(annotation)
    if ai_platform_class:
        return ai_platform_class

    try:
        # Python 3.7+
        from typing import ForwardRef
        if isinstance(annotation, ForwardRef):
            annotation = annotation.__forward_arg__
            ai_platform_class = get_aiplatform_class_by_name(annotation)
            if ai_platform_class:
                return ai_platform_class

    except ImportError:
        pass

def resolve_annotation(annotation: Any) -> Any:
    """Resolves annotation type against a MB SDK type.

    Use this for Optional, Union, Forward References
    """

    # handle forward refernce string

    # if this is an Ai Platform resource noun
    if inspect.isclass(annotation):
        if issubclass(annotation, aiplatform.base.AiPlatformResourceNoun):
            return annotation
    
    # handle forward references
    resolved_annotation = get_forward_reference(annotation)
    if resolved_annotation:
        return resolved_annotation

    # handle option types
    if getattr(annotation, '__origin__', None) is Union:
        # assume optional type
        # TODO check for optional type
        resolved_annotation = get_forward_reference(annotation.__args__[0])
        if resolved_annotation:
            return resolved_annotation
        else:
            return annotation.__args__[0]

    return annotation

def get_serializer(annotation: Any) -> Optional[Callable]:
    """Get serailizer for objects to pass them as strings.
    
    Remote runner will deserialize.
    # TODO handle proto.Message
    """ 
    if getattr(annotation, '__origin__', None) in (dict, list):
        return json.dumps

def get_deserializer(annotation: Any) -> Optional[Callable]:
    """Get deserailizer for objects to pass them as strings.
    
    Remote runner will deserialize.
    # TODO handle proto.Message
    """ 
    if getattr(annotation, '__origin__', None) in (dict, list):
        return json.loads