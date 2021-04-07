import collections
import inspect
import json
from typing import Any, Callable, Optional, Union

from google.cloud import aiplatform


def get_forward_reference(annotation: Any) -> Optional[aiplatform.base.AiPlatformResourceNoun]:
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

    Args:
        annotation: Annotation to resolve
    Returns:
        Direct annotation 
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

    # handle optional types
    if getattr(annotation, '__origin__', None) is Union:
        # assume optional type
        # TODO check for optional type
        resolved_annotation = get_forward_reference(annotation.__args__[0])
        if resolved_annotation:
            return resolved_annotation
        else:
            return annotation.__args__[0]

    if annotation is inspect._empty:
        return None

    return annotation

def is_serializable_to_json(annotation: Any) -> bool:
    """Checks if type is serializable.

    Args:
        annotation: parameter annotation
    Returns:
        True if serializable to json.
    """
    serializable_types = (dict, list, collections.abc.Sequence)
    return getattr(annotation, '__origin__', None) in serializable_types

def is_mb_sdk_resource_noun_type(mb_sdk_type: Any) -> bool:
    """Determines if type passed in should be a metadata type.

    Args:
        mb_sdk_type: Type to check
    Returns:
        True if this is a resource noun
    """
    if inspect.isclass(mb_sdk_type):
        return issubclass(mb_sdk_type, aiplatform.base.AiPlatformResourceNoun)
    return False

def get_serializer(annotation: Any) -> Optional[Callable]:
    """Get serailizer for objects to pass them as strings.
    
    Remote runner will deserialize.
    # TODO handle proto.Message

    Args:
        annotation: Parameter annotation
    Returns:
        serializer for that annotation type

    """ 
    if is_serializable_to_json(annotation):
        return json.dumps

def get_deserializer(annotation: Any) -> Optional[Callable[..., str]]:
    """Get deserailizer for objects to pass them as strings.
    
    Remote runner will deserialize.
    # TODO handle proto.Message
    Args:
        annotation: parameter annotatoin
    Returns:
        deserializer for annotation type
    """ 
    if is_serializable_to_json(annotation):
        return json.loads