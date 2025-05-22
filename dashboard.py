import dash
from dash import dcc, html, Input, Output
from datetime import datetime
import pytz
import pandas as pd
import plotly.express as px
import os
import json

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Wicking Experiment Dashboard"

# Get available experiment folders
data_dir = "D:\Wicking_Github\Wicking_Tracker\output"
experiment_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

#UTC TO Time
def format_timestamp(utc_str):
    try:
        dt = datetime.strptime(utc_str, "%Y%m%d_%H%M%S")
        dt_utc = pytz.utc.localize(dt)
        dt_local = dt_utc.astimezone(pytz.timezone("America/New_York"))
        return dt_local.strftime("%b %d, %Y at %#I:%M %p %Z")  # %-I--for rasberry pi
    except Exception as e:
        return utc_str  # fallback if format fails

options = []
for folder in experiment_folders:
    metadata_path = os.path.join(data_dir, folder, "metadata.json")
    label = folder
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        if "operator" in meta:
            label = f"{folder} (👤 {meta['operator']})"
    options.append({"label": label, "value": folder})

# App Layout
app.layout = html.Div([
    # Title
    html.H1("📊 Wicking Tracker Dashboard", style={
        'textAlign': 'center',
        'marginTop': '20px',
        'marginBottom': '10px',
        'color': '#2c3e50',
        'fontFamily': 'Segoe UI, sans-serif'
    }),

    # Dropdown area
    html.Div([
        
        dcc.Dropdown(
            id="folder-dropdown",
            options=options,
            placeholder="Select one or more experiments",
            multi=True,
            searchable=True,
            value=[experiment_folders[0]] if experiment_folders else [],
            style={
                "padding": "5px",
                "fontSize": "16px",
                "fontFamily": "Segoe UI",
                "borderRadius": "6px"
            }
        ),
    ], style={"padding": "0 20px"}),

    # Content area: Sidebar + Graphs
    html.Div([
        # Sidebar: Metadata
        html.Div(id="metadata-display", style={
            "width": "25%",
            "overflowY": "auto",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #ccc",
            "boxShadow": "inset -1px 0 0 rgba(0, 0, 0, 0.05)",
            "flexGrow": "1",
            "minHeight": "0"
        }),

        # Main panel: Graphs + Image
        html.Div([
            dcc.Graph(id="height-graph", config={"displayModeBar": True}),
            dcc.Graph(id="rate-graph", config={"displayModeBar": True}),
            html.Div(id="image-display", style={"marginTop": "40px"})
        ], style={
            "width": "75%",
            "padding": "20px"
        })
    ], style={
        "display": "flex",
        "flexDirection": "row",
        "flexGrow": "1",
        "height": "100%",
        "overflow": "hidden",
        "marginTop": "20px",
        "borderTop": "1px solid #dee2e6"
    })
], style={
    "backgroundColor": "#ffffff",
    "display": "flex",
    "flexDirection": "column",
    "minHeight": "100vh",
    "boxSizing": "border-box",
    "fontFamily": "Segoe UI, sans-serif"
})


def extract_date(folder_name):
    try:
        # Example: ExperimentName_20250522_134522
        parts = folder_name.split("_")
        date_str = parts[-2] + "_" + parts[-1]
        return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
    except:
        return datetime.min  # fallback for invalid formats

# Load and sort experiments by date (newest first)
experiment_folders = sorted(
    [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))],
    key=extract_date,
    reverse=True
)

# Callback to update plots and metadata
@app.callback(
    [Output("height-graph", "figure"),
     Output("rate-graph", "figure"),
     Output("metadata-display", "children")],
    [Input("folder-dropdown", "value")]
)
def update_dashboard(selected_folders):
    if not selected_folders:
        return {}, {}, "Select one or more experiments."

    all_dfs = []
    all_meta = []

    for folder in selected_folders:
        folder_path = os.path.join(data_dir, folder)

        try:
            df = pd.read_csv(os.path.join(folder_path, "data.csv"))
            df["Experiment"] = folder  # Add a label column
            all_dfs.append(df)

            metadata_path = os.path.join(folder_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    meta = json.load(f)
                meta["Folder"] = folder
                all_meta.append(meta)

        except Exception as e:
            print(f"Error loading {folder}: {e}")

    if not all_dfs:
        return {}, {}, "No valid data found."

    combined_df = pd.concat(all_dfs)

    fig_height = px.line(
        combined_df,
        x="Time",
        y="Height",
        color="Experiment",
        title="Height Over Time"
    )

    if "Wicking Rate" in combined_df.columns:
        fig_rate = px.line(
            combined_df,
            x="Time",
            y="Wicking Rate",
            color="Experiment",
            title="Wicking Rate Over Time"
        )
    else:
        fig_rate = px.line(title="No 'Wicking Rate' column found.")
    

    # Display metadata
    meta_display = html.Div([
        html.Div([
            html.H3(f"{m['Folder']} Metadata", style={
                "color": "#2c3e50",
                "marginBottom": "10px",
                "fontSize": "1.2em",
                "borderBottom": "2px solid #e9ecef",
                "paddingBottom": "6px"
            }),
            html.Ul([
                html.Li(
                f"{k}: {format_timestamp(v) if k == 'timestamp' else v}",
                style={
                    "marginBottom": "6px",
                    "color": "#495057",
                    "listStyleType": "none",
                    "padding": "4px 0",
                    "borderBottom": "1px solid #e9ecef"
                }
            ) for k, v in m.items() if k != "Folder"
            ], style={"padding": "0", "margin": "0"})
        ], style={"marginBottom": "20px"})
        for m in all_meta
    ]) if all_meta else html.Div("No metadata available.", style={
        "color": "#6c757d",
        "fontStyle": "italic",
        "textAlign": "center",
        "padding": "20px"
    })

    return fig_height, fig_rate, meta_display


if __name__ == "__main__":
    app.run(debug=True)
