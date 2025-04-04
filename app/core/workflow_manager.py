from core.workflows import image_classification, object_detection, time_series, regression

def run_task(task_name, dataset_path, goal=None):
    task_map = {
        "Image Classification": image_classification.run,
        "Object Detection": object_detection.run,
        "Time Series Forecasting": time_series.run,
        "Regression": regression.run,
    }

    run_func = task_map.get(task_name)
    if run_func:
        return run_func(dataset_path, goal)
    else:
        return {"error": f"No workflow found for task: {task_name}"}
