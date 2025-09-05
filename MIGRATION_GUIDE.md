# Migration to PydanticAI Evals

This document outlines the migration from custom evaluators to PydanticAI's built-in `pydantic_evals` system.

## What Changed

### 1. Replaced Custom Evaluators

The old custom evaluator system in `promptdev/assertions/builtin.py` has been replaced with PydanticAI's `pydantic_evals` framework:

- **Before**: Custom `BaseEvaluator` classes with manual JSON parsing
- **After**: Type-safe evaluators using `pydantic_evals.Evaluator`

### 2. Fixed Schema Resolution

Schema references (`$ref`) in assertion templates now work correctly:

```yaml
# This now works properly
assertion_templates:
  myTemplate:
    type: "json_schema"
    value:
      $ref: "#/schemas/mySchema"

schemas:
  mySchema:
    type: "object"
    properties: {...}
```

### 3. Added LLM-based Evaluation

New `llm_judge` evaluator for semantic assessment:

```yaml
assert:
  - type: "llm_judge"
    rubric: "Evaluate accuracy and completeness"
    model: "openai:gpt-4"
```

## New Evaluator Types

| Type            | Description                     | Example                                | Promptfoo Compatible |
| --------------- | ------------------------------- | -------------------------------------- | -------------------- |
| `json_schema`   | Validates JSON against schema   | `type: "json_schema"`                  | ✅                    |
| `contains-json` | Promptfoo JSON validation       | `type: "contains-json"`                | ✅                    |
| `python`        | Custom Python assertions        | `type: "python", value: "./assert.py"` | ✅                    |
| `exact`         | Exact string matching           | `type: "exact", value: "expected"`     | ✅                    |
| `contains`      | Substring matching              | `type: "contains", value: "text"`      | ✅                    |
| `is_instance`   | Type checking                   | `type: "is_instance", value: "str"`    | ✅                    |
| `llm_judge`     | LLM-based evaluation            | `type: "llm_judge", rubric: "..."`     | ✅                    |
| `llm-rubric`    | Promptfoo LLM rubric evaluation | `type: "llm-rubric", value: "rubric"`  | ✅                    |
| `g-eval`        | G-Eval methodology              | `type: "g-eval", value: "criteria"`    | ✅                    |

## Benefits

1. **Type Safety**: Built on Pydantic's validation framework
2. **Better Error Handling**: More informative error messages
3. **Advanced Evaluation**: LLM-based semantic assessment
4. **Standards Compliance**: Uses PydanticAI's established patterns
5. **Future-Proof**: Leverages actively maintained evaluation framework

## Promptfoo Compatibility

PromptDev now provides **complete compatibility** with promptfoo configurations:

### ✅ **All Evaluator Types Supported**
- `contains-json` - JSON validation against schemas  
- `llm-rubric` - LLM-based rubric evaluation
- `g-eval` - G-Eval methodology for quality assessment
- `python` - Custom Python assertion functions
- Plus all standard evaluators (exact, contains, etc.)

### ✅ **Configuration Format**
- YAML configs work as-is
- `defaultTest` instead of `default_test` 
- `assertionTemplates` instead of `assertion_templates`
- Direct schema definitions (e.g., `calendarEventSummarySchema:`)
- Template references (`$ref: '#/assertionTemplates/...'`)
- Schema references (`$ref: '#/schemaName'`)

### ✅ **Python Assertions**
- Existing `get_assert` functions work unchanged
- Same function signature: `get_assert(output: str, context: dict)`
- Same return types: `bool`, `float`, or `dict` with `pass`/`score`/`reason`

### ✅ **Data Sources**
- JSONL datasets are supported
- File references (`file://./path/to/data.jsonl`)
- Inline test cases
- Variable substitution

## Performance

The new system provides:

- Better parallelization through `pydantic_evals`
- More efficient JSON parsing
- Reduced memory usage for large datasets
- Improved error recovery

## Examples

See `examples/calendar_event_summary_pydantic_evals.yaml` for a complete example using the new features.
