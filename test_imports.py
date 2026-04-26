
try:
    from single_run import run_single_simulation
    print("single_run imported successfully")
except ImportError as e:
    print(f"Failed to import single_run: {e}")
except Exception as e:
    print(f"Error importing single_run: {e}")

try:
    from terminal_ui import AdvancedPhaseTracker
    print("terminal_ui imported successfully")
except ImportError as e:
    print(f"Failed to import terminal_ui: {e}")
except Exception as e:
    print(f"Error importing terminal_ui: {e}")
