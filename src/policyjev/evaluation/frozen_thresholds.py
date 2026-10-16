"""Write the frozen table of validation-fitted thresholds used on the test split."""

from policyjev.evaluation.calibration import configured_methods
from policyjev.evaluation.splits import load_split_scores_bundle, write_thresholds_table
from policyjev.evaluation.stack import STACK_SYSTEM
from policyjev.evaluation.validation_report import SYSTEMS


def main() -> None:
    """Fit thresholds on out-of-fold calibrated validation scores and save the table."""
    systems = SYSTEMS + [STACK_SYSTEM]
    data = load_split_scores_bundle("validation", systems, configured_methods())
    path = write_thresholds_table(data.thresholds, data.validation_f1)
    print(f"wrote thresholds for {', '.join(systems)} to {path}")


if __name__ == "__main__":
    main()
