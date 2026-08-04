import ast
import json
import re
import uuid
from typing import Any, Optional

from .. import config
from ..models import ChatCompletionRequest, MalformedToolCallError
from .token import json_safe
from .text import extract_litert_tool_calls


# ---------------------------------------------------------------------------
# Tool definition helpers
# ---------------------------------------------------------------------------


def enrich_tool_definition(
    tool: dict[str, Any],
) -> dict[str, Any]:
    return json_safe(tool)


def normalize_tool_definitions(
    request: ChatCompletionRequest,
) -> list[dict[str, Any]]:
    if request.tool_choice == "none":
        return []

    tools = request.tools if isinstance(request.tools, list) else []
    definitions = []

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        definitions.append(enrich_tool_definition(tool))

    if isinstance(request.tool_choice, dict):
        function = request.tool_choice.get("function")
        required_name = (
            function.get("name") if isinstance(function, dict) else None
        )

        if required_name:
            definitions = [
                definition
                for definition in definitions
                if definition.get("function", {}).get("name") == required_name
            ]

    return definitions


def tool_definition_map(
    request: ChatCompletionRequest,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for definition in normalize_tool_definitions(request):
        function = definition.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if isinstance(name, str) and name:
            result[name] = function

    return result


# ---------------------------------------------------------------------------
# Protocol-leak detection & JSON Schema validation
# ---------------------------------------------------------------------------

_PROTOCOL_LEAK_PATTERNS = (
    re.compile(
        r"(?:^|[\r\n\]}])(?:user|assistant|model|tool)\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[(?:user|assistant|model|tool)\](?:\r?\n|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\|(?:user|assistant|model|tool|turn|channel|end)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s.}\]])call:[A-Za-z_][A-Za-z0-9_.-]*\{",
        re.IGNORECASE,
    ),
    re.compile(r"}response:", re.IGNORECASE),
)


def value_contains_protocol_leak(value: Any) -> bool:
    if isinstance(value, str):
        if len(value) > config.MAX_TOOL_ARGUMENT_STRING_LENGTH:
            return True

        return any(
            pattern.search(value)
            for pattern in _PROTOCOL_LEAK_PATTERNS
        )

    if isinstance(value, dict):
        return any(
            value_contains_protocol_leak(key)
            or value_contains_protocol_leak(item)
            for key, item in value.items()
        )

    if isinstance(value, list):
        return any(
            value_contains_protocol_leak(item)
            for item in value
        )

    return False


def json_schema_type_matches(
    value: Any,
    expected_type: Any,
) -> bool:
    if isinstance(expected_type, list):
        return any(
            json_schema_type_matches(value, item)
            for item in expected_type
        )

    if expected_type == "null":
        return value is None

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if expected_type == "array":
        return isinstance(value, list)

    if expected_type == "object":
        return isinstance(value, dict)

    return True


def validate_schema_value(
    value: Any,
    schema: Any,
    path: str,
) -> list[str]:
    if not isinstance(schema, dict):
        return []

    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type is not None and not json_schema_type_matches(
        value,
        expected_type,
    ):
        errors.append(
            f"{path} has the wrong type; expected {expected_type!r}."
        )
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path} is not one of the allowed values.")

    if isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(
                        f"{path}.{key} is required."
                    )

        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                errors.append(
                    f"{path} contains unknown properties: "
                    + ", ".join(unknown)
                )

        for key, item in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                errors.extend(
                    validate_schema_value(
                        item,
                        child_schema,
                        f"{path}.{key}",
                    )
                )

    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    validate_schema_value(
                        item,
                        item_schema,
                        f"{path}[{index}]",
                    )
                )

    return errors


def validate_litert_tool_calls(
    tool_calls: list[dict[str, Any]],
    request: ChatCompletionRequest,
) -> None:
    definitions = tool_definition_map(request)

    if not tool_calls:
        return

    if not definitions:
        raise MalformedToolCallError(
            "The model returned tool calls but the request supplied no tools."
        )

    errors: list[str] = []

    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function")
        if not isinstance(function, dict):
            errors.append(
                f"tool_calls[{index}] has no function object."
            )
            continue

        name = function.get("name")
        arguments = function.get("arguments")

        if not isinstance(name, str) or not name:
            errors.append(
                f"tool_calls[{index}] has no valid function name."
            )
            continue

        definition = definitions.get(name)
        if definition is None:
            errors.append(
                f"tool_calls[{index}] requested unknown tool {name!r}."
            )
            continue

        if not isinstance(arguments, dict):
            errors.append(
                f"tool_calls[{index}] arguments are not an object."
            )
            continue

        if value_contains_protocol_leak(arguments):
            errors.append(
                f"tool_calls[{index}] for {name!r} contains "
                "chat-template or tool-protocol leakage."
            )
            continue

        schema = definition.get("parameters")
        errors.extend(
            validate_schema_value(
                arguments,
                schema,
                f"tool_calls[{index}].arguments",
            )
        )

    if errors:
        raise MalformedToolCallError(" ".join(errors))


# ---------------------------------------------------------------------------
# Tool-call recovery from text (handles malformed model output)
# ---------------------------------------------------------------------------


def normalize_recovered_tool_call(
    name: Any,
    arguments: Any,
) -> Optional[dict[str, Any]]:
    if not isinstance(name, str) or not name.strip():
        return None

    if isinstance(arguments, str):
        parsed_arguments: Any = None

        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            try:
                parsed_arguments = ast.literal_eval(arguments)
            except (ValueError, SyntaxError):
                parsed_arguments = None

        arguments = parsed_arguments

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return None

    return {
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


def recover_tool_calls_from_object(
    value: Any,
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        recovered: list[dict[str, Any]] = []

        for item in value:
            recovered.extend(recover_tool_calls_from_object(item))

        return recovered

    if not isinstance(value, dict):
        return []

    direct = extract_litert_tool_calls(value)
    if direct:
        return direct

    function = value.get("function")
    if isinstance(function, dict):
        recovered_call = normalize_recovered_tool_call(
            function.get("name"),
            function.get("arguments", {}),
        )
        if recovered_call is not None:
            return [recovered_call]

    if value.get("type") in {"tool_call", "function"}:
        recovered_call = normalize_recovered_tool_call(
            value.get("name"),
            value.get("arguments", {}),
        )
        if recovered_call is not None:
            return [recovered_call]

    recovered = []

    for key in (
        "message",
        "response",
        "output",
        "content",
        "tool_calls",
    ):
        if key in value:
            recovered.extend(
                recover_tool_calls_from_object(value.get(key))
            )

    return recovered


def recover_tool_calls_from_text(
    text: str,
) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        recovered = recover_tool_calls_from_object(parsed)
        if recovered:
            return recovered

    for match in re.finditer(
        r"(?:^|[\s<\[\]}>])call:"
        r"([A-Za-z_][A-Za-z0-9_.-]*)\s*",
        stripped,
        flags=re.IGNORECASE,
    ):
        name = match.group(1)
        brace_index = stripped.find("{", match.end())

        if brace_index < 0:
            continue

        depth = 0
        quote: Optional[str] = None
        escaped = False
        argument_text = None

        for index in range(brace_index, len(stripped)):
            character = stripped[index]

            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue

            if character in {"'", '"'}:
                quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1

                if depth == 0:
                    argument_text = stripped[brace_index : index + 1]
                    break

        if argument_text is None:
            continue

        try:
            arguments = json.loads(argument_text)
        except json.JSONDecodeError:
            try:
                arguments = ast.literal_eval(argument_text)
            except (ValueError, SyntaxError):
                continue

        recovered_call = normalize_recovered_tool_call(
            name,
            arguments,
        )
        if recovered_call is not None:
            return [recovered_call]

    return []


def looks_like_tool_protocol(text: str) -> bool:
    stripped = text.lstrip()
    lowered = stripped.casefold()

    return (
        lowered.startswith("call:")
        or lowered.startswith("<|tool")
        or lowered.startswith("<start_function_call>")
        or '"tool_calls"' in lowered
        or '"tool_call"' in lowered
    )


def litert_tool_calls_to_openai(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name")

        if not isinstance(name, str) or not name:
            continue

        arguments = function.get("arguments", {})

        result.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )

    return result


def tool_call_key(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function", {})
    return json.dumps(
        {
            "name": function.get("name"),
            "arguments": function.get("arguments", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
