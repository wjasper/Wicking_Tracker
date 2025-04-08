import os
import json
import datetime


def save_data(df, plot_image):

    experiment_name = input("Enter Experiment Name: ")

    # Generate folder name with UTC timestamp
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{experiment_name}_{timestamp}"
    output_path = os.path.join("output", folder_name)
 
    os.makedirs(output_path, exist_ok=True)
 
    # Save CSV
    csv_path = os.path.join(output_path, "data.csv")
    df.to_csv(csv_path, index=False)
    
    # Save Image
    image_path = os.path.join(output_path, "plot.png")
    plot_image.save(image_path)
 
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

 
    print(f"Data saved to: {csv_path}")
    