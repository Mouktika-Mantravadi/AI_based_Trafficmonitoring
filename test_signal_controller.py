"""
Verification and Test Script for Adaptive Traffic Signal Controller.
Tests sample LOW, MEDIUM, and HIGH traffic density conditions to verify green time selection and signal sequence logic.
"""

import time
import config
from src.density_calculator import DensityCalculator
from src.signal_controller import SignalController


def test_adaptive_signal_logic():
    print("==================================================")
    print(" AI Traffic Control - Adaptive Signal Logic Test ")
    print("==================================================")
    print(f" Configured Green Durations : {config.SIGNAL_DURATIONS}")
    print(f" Configured Yellow Duration : {config.YELLOW_DURATION} seconds")
    print("==================================================\n")

    signal_controller = SignalController()
    density_calculator = DensityCalculator()

    # Define test cases for LOW, MEDIUM, and HIGH traffic scenarios
    test_scenarios = [
        {
            "name": "Scenario 1: LOW Traffic Density",
            "counts": {"car": 2, "motorcycle": 2, "bus": 0, "truck": 0},
            "expected_density": "LOW",
            "expected_green": 15
        },
        {
            "name": "Scenario 2: MEDIUM Traffic Density",
            "counts": {"car": 8, "motorcycle": 4, "bus": 1, "truck": 1},
            "expected_density": "MEDIUM",
            "expected_green": 30
        },
        {
            "name": "Scenario 3: HIGH Traffic Density",
            "counts": {"car": 15, "motorcycle": 8, "bus": 2, "truck": 2},
            "expected_density": "HIGH",
            "expected_green": 45
        }
    ]

    all_passed = True

    for scenario in test_scenarios:
        print(f"--- Running {scenario['name']} ---")
        print(f" Vehicle Counts: {scenario['counts']}")

        # Step 4 Density Calculation
        density_data = density_calculator.calculate_density(scenario["counts"])
        category = density_data["density_category"]
        score = density_data["weighted_score"]

        # Step 5 Signal Timing Calculation
        signal_data = signal_controller.compute_signal_timing(density_data)
        green_time = signal_data["recommended_green_time"]
        state = signal_data["current_state"]

        print(f" Calculated PCU Score : {score} PCU")
        print(f" Determined Density   : {category} (Expected: {scenario['expected_density']})")
        print(f" Recommended Green    : {green_time} seconds (Expected: {scenario['expected_green']}s)")
        print(f" Initial Signal State : {state}")

        # Verification check
        category_match = (category == scenario["expected_density"])
        green_match = (green_time == scenario["expected_green"])

        if category_match and green_match:
            print(" [PASS] Green duration correctly assigned!\n")
        else:
            print(" [FAIL] Test criteria not met!\n")
            all_passed = False

    print("--- Testing Signal Sequence State Transitions ---")
    demo_controller = SignalController()
    print(f" Initial State: {demo_controller.current_state}")
    
    # Test GREEN -> YELLOW transition after duration
    print(" Advancing state transition logic...")
    active_state = demo_controller.update_signal_sequence(green_duration=1)
    print(f" Active State after Green expiry: {active_state}")

    print("\n==================================================")
    if all_passed:
        print(" [SUCCESS] All 3 Traffic Density Cases Verified Cleanly!")
    else:
        print(" [ERROR] One or more test cases failed.")
    print("==================================================")


if __name__ == "__main__":
    test_adaptive_signal_logic()
