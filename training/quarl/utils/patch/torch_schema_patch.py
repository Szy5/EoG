import functools
import inspect
import types
import typing
from collections import abc

_PATCH_FLAG = "_kgqa_pep585_schema_patch_applied"


def _normalize_annotation(annotation):
    origin = typing.get_origin(annotation)
    if origin is None:
        return annotation

    args = tuple(_normalize_annotation(arg) for arg in typing.get_args(annotation))

    if origin is list:
        return typing.List[args[0]] if args else typing.List
    if origin is dict:
        return typing.Dict[args[0], args[1]] if len(args) == 2 else typing.Dict
    if origin is tuple:
        return typing.Tuple[args] if len(args) != 2 or args[1] is not Ellipsis else typing.Tuple[args[0], ...]
    if origin is set:
        return typing.Set[args[0]] if args else typing.Set
    if origin is abc.Sequence:
        return typing.Sequence[args[0]] if args else typing.Sequence
    if origin is abc.Callable:
        return annotation
    if origin is typing.Union:
        return typing.Union[args]

    return annotation


def _clone_with_normalized_annotations(func):
    cloned = types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
    cloned = functools.update_wrapper(cloned, func)
    cloned.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
    normalized_annotations = {
        key: _normalize_annotation(value) for key, value in getattr(func, "__annotations__", {}).items()
    }
    cloned.__annotations__ = normalized_annotations
    try:
        signature = inspect.signature(func)
        cloned.__signature__ = signature.replace(
            parameters=[
                parameter.replace(
                    annotation=normalized_annotations.get(parameter.name, parameter.annotation),
                )
                for parameter in signature.parameters.values()
            ],
            return_annotation=normalized_annotations.get("return", signature.return_annotation),
        )
    except (TypeError, ValueError):
        pass
    return cloned


def apply_torch_schema_patch():
    import torch

    infer_schema = getattr(torch.library, "infer_schema", None)
    if infer_schema is None or getattr(torch.library, _PATCH_FLAG, False):
        return

    @functools.wraps(infer_schema)
    def patched_infer_schema(prototype_function, mutates_args, op_name=None):
        try:
            return infer_schema(prototype_function, mutates_args=mutates_args, op_name=op_name)
        except ValueError as exc:
            message = str(exc)
            if "unsupported type list[" not in message:
                raise
            normalized = _clone_with_normalized_annotations(prototype_function)
            return infer_schema(normalized, mutates_args=mutates_args, op_name=op_name)

    torch.library.infer_schema = patched_infer_schema
    setattr(torch.library, _PATCH_FLAG, True)
