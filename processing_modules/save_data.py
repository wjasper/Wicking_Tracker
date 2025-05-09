import os
import json
import datetime


def save_data(df, plot_image, height_plot_image):
    experiment_name = input("Enter Experiment Name: ")

    # Generate folder name with UTC timestamp
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{experiment_name}_{timestamp}"
    output_path = os.path.join(".", "output", folder_name)

    os.makedirs(output_path, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_path, "data.csv")
    df.to_csv(csv_path, index=False)

    # Save Height Plot Image
    height_path = os.path.join(output_path, "height_plot.png")
    if height_plot_image is not None:
        height_plot_image.save(height_path)
    else:
        print("[WARNING] No height plot image to save — skipping height_plot.png")

    # Save Wicking Rate Plot Image
    wicking_path = os.path.join(output_path, "wicking_plot.png")
    if plot_image is not None:
        plot_image.save(wicking_path)
    else:
        print("[WARNING] No wicking rate plot image to save — skipping wicking_plot.png")

    # Metadata
    operator_name = input("Enter Operator Name: ")
    comment = input("Enter Comments: ")

    metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "operator": operator_name,
        "comment": comment
    }

    # Save metadata as JSON
    json_path = os.path.join(output_path, "metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Data and plots saved to: {output_path}")
    