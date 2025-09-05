def get_assert():
    def assert_expected(output, context):
        """
        Custom assertion function that validates the extracted JSON data.

        Args:
            output: The agent's output string
            context: Dictionary containing 'vars' and 'expected' keys

        Returns:
            Score between 0.0 and 1.0
        """
        import json
        import re

        if not output:
            return 0.0

        try:
            # Extract JSON from the output (handle markdown code blocks)
            json_match = re.search(r"```json\s*\n(.*?)\n```", output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return 0.0

            # Parse the JSON
            parsed = json.loads(json_str)

            # Get expected values from context
            vars_dict = context.get("vars", {})
            expected_name = vars_dict.get("expected_name")
            expected_out_of_office = vars_dict.get("expected_out_of_office")
            expected_event_type = vars_dict.get("expected_event_type")

            # Validate the extracted data
            results = []
            total_checks = 3
            passed_checks = 0

            # Check name
            actual_name = parsed.get("name")
            name_passed = actual_name == expected_name
            if name_passed:
                passed_checks += 1
            else:
                results.append(
                    {
                        "field": "Name",
                        "actual": actual_name,
                        "expected": expected_name,
                        "passed": False,
                    }
                )

            # Check out_of_office
            actual_out_of_office = parsed.get("out_of_office")
            out_of_office_passed = actual_out_of_office == expected_out_of_office
            if out_of_office_passed:
                passed_checks += 1
            else:
                results.append(
                    {
                        "field": "Out of Office",
                        "actual": actual_out_of_office,
                        "expected": expected_out_of_office,
                        "passed": False,
                    }
                )

            # Check event_type (case insensitive)
            actual_event_type = parsed.get("event_type", "")
            expected_event_type_str = str(expected_event_type)
            event_type_passed = actual_event_type.lower() == expected_event_type_str.lower()
            if event_type_passed:
                passed_checks += 1
            else:
                results.append(
                    {
                        "field": "Event Type",
                        "actual": actual_event_type,
                        "expected": expected_event_type_str,
                        "passed": False,
                    }
                )

            # Return detailed results for promptfoo format
            score = passed_checks / total_checks
            return {
                "pass": score == 1.0,
                "score": score,
                "reason": f"Field validation results: {len(results)} failed checks"
                if results
                else "All fields match",
                "details": results,
            }

        except (json.JSONDecodeError, KeyError, AttributeError):
            return 0.0

    return assert_expected
