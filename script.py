import os
import pandas as pd

# Set your main output directory
output_dir = "output"  # Adjust this if your path is different

# Loop through all subdirectories
for folder in os.listdir(output_dir):
    folder_path = os.path.join(output_dir, folder)
    csv_path = os.path.join(folder_path, "data.csv")

    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)

            if "Time_Uniform" in df.columns and "Height_Model" in df.columns:
                df["Modeled Avg Wicking Rate"] = df.apply(
                    lambda row: row["Height_Model"] / row["Time_Uniform"] if row["Time_Uniform"] > 0 else 0,
                    axis=1
                )
                df.to_csv(csv_path, index=False)
                print(f"Updated: {csv_path}")
            else:
                print(f"⚠️ Skipped (missing columns): {csv_path}")
        except Exception as e:
            print(f"Error in {csv_path}: {e}")
