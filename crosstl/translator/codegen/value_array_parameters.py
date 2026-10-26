"""Copy written ``in`` array parameters into locals for CUDA and HIP.

HLSL passes an array parameter without ``out`` or ``inout`` by value, and the
HLSL importer records it with the explicit ``in`` qualifier. A C++ array
parameter decays to a pointer, so a write in the callee would reach the
caller's array. A written ``in`` array with fixed dimensions is therefore
received under another name and copied element by element into a local array
that keeps the source name, as Metal output does.

Unwritten ``in`` arrays keep pointer passing, ``out`` and ``inout`` arrays keep
reference semantics, and arrays without a qualifier keep their declaration,
because sources such as CUDA treat them as pointers.
"""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import split_array_type_suffix
from .generic_function_utils import iter_function_nodes
from .resource_arrays import split_array_suffix

# Builtins that write through an argument, by argument position.
_WRITING_BUILTIN_ARGUMENTS = {"frexp": (1,), "modf": (1,), "sincos": (1, 2)}
_INCREMENT_OPERATORS = frozenset({"++", "--"})


def collect_function_parameters_by_name(ast_node, type_name_string):
    """Map each function name to ``(qualifiers, is_array)`` per parameter, per overload."""
    parameters_by_name = {}
    for function in iter_function_nodes(ast_node):
        parameters = [
            (
                parameter_qualifier_names(parameter),
                bool(
                    split_array_type_suffix(
                        type_name_string(getattr(parameter, "param_type", ""))
                    )[1]
                ),
            )
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        ]
        parameters_by_name.setdefault(function.name, []).append(parameters)
    return parameters_by_name


def parameter_qualifier_names(parameter):
    return {
        str(qualifier).strip().lower()
        for qualifier in getattr(parameter, "qualifiers", None) or ()
    }


def fixed_array_dimensions(type_name):
    """The dimension texts of a fixed-size array type, or ``None``."""
    _base_type, suffix = split_array_type_suffix(type_name)
    dimensions = split_array_suffix(suffix)
    if not dimensions or any(not dimension.strip() for dimension in dimensions):
        return None
    return [dimension.strip() for dimension in dimensions]


def _lvalue_root_name(expr):
    while True:
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, MemberAccessNode):
            expr = expr.object_expr
        elif isinstance(expr, ArrayAccessNode):
            expr = expr.array_expr
        elif isinstance(expr, SwizzleNode):
            expr = expr.vector_expr
        else:
            return None


def _body_nodes(body):
    if isinstance(body, (list, tuple)):
        for statement in body:
            yield from _body_nodes(statement)
        return
    walk = getattr(body, "walk", None)
    if walk is not None:
        yield from walk()


def _call_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    if isinstance(function, str):
        return function
    return None


def _call_writes_argument(call, index, parameters_by_name):
    name = _call_name(call)
    if name in _WRITING_BUILTIN_ARGUMENTS:
        return index in _WRITING_BUILTIN_ARGUMENTS[name]
    for parameters in parameters_by_name.get(name, ()):
        if index >= len(parameters):
            continue
        qualifiers, is_array = parameters[index]
        if qualifiers & {"out", "inout"}:
            return True
        if is_array and not qualifiers & {"in", "const"}:
            # An array parameter without a qualifier is a pointer the callee
            # may write through.
            return True
    return False


def written_parameter_names(body, parameters_by_name):
    """Names whose variable the body assigns, increments or passes to a writer."""
    names = set()
    for node in _body_nodes(body):
        if isinstance(node, AssignmentNode):
            names.add(_lvalue_root_name(node.target))
        elif isinstance(node, UnaryOpNode) and node.operator in _INCREMENT_OPERATORS:
            names.add(_lvalue_root_name(node.operand))
        elif isinstance(node, FunctionCallNode):
            for index, argument in enumerate(node.arguments or ()):
                if _call_writes_argument(node, index, parameters_by_name):
                    names.add(_lvalue_root_name(argument))
    names.discard(None)
    return names


def plan_value_array_copies(function, parameters, type_name_string, parameters_by_name):
    """Map each written ``in`` array parameter to the name it is received under."""
    candidates = [
        parameter
        for parameter in parameters
        if "in" in parameter_qualifier_names(parameter)
        and fixed_array_dimensions(
            type_name_string(getattr(parameter, "param_type", ""))
        )
        is not None
    ]
    if not candidates:
        return {}
    body = getattr(function, "body", None)
    written = written_parameter_names(body, parameters_by_name)
    used_names = {getattr(parameter, "name", None) for parameter in parameters}
    for node in _body_nodes(body):
        if isinstance(node, (IdentifierNode, VariableNode)):
            used_names.add(node.name)
    copies = {}
    for parameter in candidates:
        if parameter.name not in written:
            continue
        incoming = f"{parameter.name}_crosstl_input"
        suffix = 2
        while incoming in used_names:
            incoming = f"{parameter.name}_crosstl_input{suffix}"
            suffix += 1
        used_names.add(incoming)
        copies[parameter.name] = incoming
    return copies


def value_array_copy_lines(declaration, name, incoming, dimensions):
    """Declare the local array ``name`` and copy ``incoming`` into it."""
    indices = [f"crosstl_{name}_i{level}" for level in range(len(dimensions))]
    subscript = "".join(f"[{index}]" for index in indices)
    statement = f"{name}{subscript} = {incoming}{subscript};"
    for level in reversed(range(len(dimensions))):
        index = indices[level]
        statement = (
            f"for (int {index} = 0; {index} < int({dimensions[level]}); "
            f"++{index}) {{ {statement} }}"
        )
    return [f"{declaration};", statement]
