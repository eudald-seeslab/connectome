import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np


def plot_manifold_3d_mult_colours(
    test_results, algorithm="tsne", painting_option="colour"
):
    # Create a mapping from shape labels to Plotly marker symbols
    shape_mapping = {"circle": "circle", "star": "diamond"}
    shape_colors = {"circle": "blue", "star": "red"}

    # Apply the shape mapping to create a column with marker symbols
    test_results["marker_symbol"] = test_results["shape"].map(shape_mapping)

    # Create the 3D scatter plot
    fig = go.Figure()

    if painting_option == "colour":
        # Add scatter3d trace for color coding
        fig.add_trace(
            go.Scatter3d(
                x=test_results[f"{algorithm}_Component_1"],
                y=test_results[f"{algorithm}_Component_2"],
                z=test_results[f"{algorithm}_Component_3"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=test_results["colour"],  # Use the actual colors
                    symbol="circle",
                    opacity=0.8,
                    line=dict(width=0),  # Remove the white border of the points
                ),
                name="colour",
            )
        )
    elif painting_option == "shape":
        # Add scatter3d trace for each unique shape
        for shape in test_results["shape"].unique():
            df_shape = test_results[test_results["shape"] == shape]
            fig.add_trace(
                go.Scatter3d(
                    x=df_shape[f"{algorithm}_Component_1"],
                    y=df_shape[f"{algorithm}_Component_2"],
                    z=df_shape[f"{algorithm}_Component_3"],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=shape_colors[shape],
                        line=dict(width=0),  # Remove the white border of the points
                    ),
                    name=shape,
                )
            )

    # Update layout to remove the background and set other properties
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
        ),
        title=f"{algorithm.upper()} of Decision Making Neuron Activations ({painting_option})",
        showlegend=True,
    )

    # Show the plot
    fig.show()


def plot_manifold_3d(test_results, algorithm="tsne"):
    shape_colors = {"circle": "blue", "star": "red"}

    # Create the 3D scatter plot
    fig = go.Figure()

    # Add scatter3d trace for color coding
    for shape in test_results["shape"].unique():
        df_shape = test_results[test_results["shape"] == shape]
        fig.add_trace(
            go.Scatter3d(
                x=df_shape[f"{algorithm}_Component_1"],
                y=df_shape[f"{algorithm}_Component_2"],
                z=df_shape[f"{algorithm}_Component_3"],
                mode="markers",
                marker=dict(
                    size=2,
                    color=shape_colors[shape],
                    opacity=0.8,
                    line=dict(width=0),  # Remove the white border of the points
                ),
                name=shape,
            )
        )

    # Update layout to remove the background and set other properties
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
        ),
        title=f"{algorithm.upper()} of Decision Making Neuron Activations",
        showlegend=True,
    )

    # Show the plot
    fig.show()


def create_dash_app_three_sliders(df, algorithm="tsne", port=8050):
    shape_colors = {"circle": "blue", "star": "red"}

    # Create the Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="3d-scatter-plot"),
            html.Div(
                [
                    html.Label("Radius:"),
                    dcc.Slider(
                        id="radius-slider",
                        min=df["radius"].min(),
                        max=df["radius"].max(),
                        step=1,
                        value=df["radius"].min(),
                        marks={int(i): str(int(i)) for i in df["radius"].unique()},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Distance:"),
                    dcc.Slider(
                        id="distance-slider",
                        min=df["distance"].min(),
                        max=df["distance"].max(),
                        step=1,
                        value=df["distance"].min(),
                        marks={int(i): str(int(i)) for i in df["distance"].unique()},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Angle:"),
                    dcc.Slider(
                        id="angle-slider",
                        min=df["angle"].min(),
                        max=df["angle"].max(),
                        step=1,
                        value=df["angle"].min(),
                        marks={int(i): str(int(i)) for i in df["distance"].unique()},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Colour:"),
                    dcc.Dropdown(
                        id="colour-dropdown",
                        options=[
                            {"label": col, "value": col}
                            for col in df["colour"].unique()
                        ],
                        value=df["colour"].unique()[0],
                    ),
                ]
            ),
        ]
    )

    @app.callback(
        Output("3d-scatter-plot", "figure"),
        Input("radius-slider", "value"),
        Input("distance-slider", "value"),
        Input("angle-slider", "value"),
        Input("colour-dropdown", "value"),
    )
    def update_figure(
        selected_radius, selected_distance, selected_angle, selected_colour
    ):

        df["highlight"] = (
            (df["radius"] == selected_radius)
            & (df["distance"] == selected_distance)
            & (df["angle"] == selected_angle)
            & (df["colour"] == selected_colour)
        )

        fig = go.Figure()

        for shape in df["shape"].unique():
            df_shape = df[df["shape"] == shape]
            fig.add_trace(
                go.Scatter3d(
                    x=df_shape[f"{algorithm}_Component_1"],
                    y=df_shape[f"{algorithm}_Component_2"],
                    z=df_shape[f"{algorithm}_Component_3"],
                    mode="markers",
                    marker=dict(
                        # Highlight selected points with size and color
                        size=np.where(df_shape["highlight"], 10, 3),
                        color=np.where(
                            df_shape["highlight"], "green", shape_colors[shape]
                        ),
                        opacity=0.8,
                        line=dict(width=0),  # Remove the white border of the points
                    ),
                    name=shape,
                )
            )

        # Update layout to remove the background and set other properties
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False
                ),
            ),
            title=f"{algorithm.upper()} of Decision Making Neuron Activations",
            showlegend=True,
        )

        return fig

    # Run the Dash app
    app.run_server(debug=True, port=port)


def create_dash_app_two_sliders(df, algorithm="tsne", port=8050):
    shape_colors = {"circle": "blue", "star": "red"}

    # Create the Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="3d-scatter-plot"),
            html.Div(
                [
                    html.Label("Distance:"),
                    dcc.Slider(
                        id="distance-slider",
                        min=df["distance"].min(),
                        max=df["distance"].max(),
                        step=1,
                        value=df["distance"].min(),
                        marks={int(i): str(int(i)) for i in df["distance"].unique()},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Angle:"),
                    dcc.Slider(
                        id="angle-slider",
                        min=df["angle"].min(),
                        max=df["angle"].max(),
                        step=1,
                        value=df["angle"].min(),
                        marks={int(i): str(int(i)) for i in df["angle"].unique()},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Colour:"),
                    dcc.Dropdown(
                        id="colour-dropdown",
                        options=[
                            {"label": col, "value": col}
                            for col in df["colour"].unique()
                        ],
                        value=df["colour"].unique()[0],
                    ),
                ]
            ),
        ]
    )

    @app.callback(
        Output("3d-scatter-plot", "figure"),
        Input("distance-slider", "value"),
        Input("angle-slider", "value"),
        Input("colour-dropdown", "value"),
    )
    def update_figure(selected_distance, selected_angle, selected_colour):

        df["highlight"] = (
            (df["distance"] == selected_distance)
            & (df["angle"] == selected_angle)
            & (df["colour"] == selected_colour)
        )

        fig = go.Figure()

        for shape in df["shape"].unique():
            df_shape = df[df["shape"] == shape]

            # Trace for non-highlighted points
            fig.add_trace(
                go.Scatter3d(
                    x=df_shape.loc[~df_shape["highlight"], f"{algorithm}_Component_1"],
                    y=df_shape.loc[~df_shape["highlight"], f"{algorithm}_Component_2"],
                    z=df_shape.loc[~df_shape["highlight"], f"{algorithm}_Component_3"],
                    mode="markers",
                    marker=dict(
                        size=1,
                        color=shape_colors[shape],
                        opacity=0.1,
                        line=dict(width=0),  # Remove the white border of the points
                    ),
                    name=f"{shape} (non-highlighted)",
                )
            )

            # Trace for highlighted points
            fig.add_trace(
                go.Scatter3d(
                    x=df_shape.loc[df_shape["highlight"], f"{algorithm}_Component_1"],
                    y=df_shape.loc[df_shape["highlight"], f"{algorithm}_Component_2"],
                    z=df_shape.loc[df_shape["highlight"], f"{algorithm}_Component_3"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="green",
                        opacity=1.0,
                        line=dict(width=0),  # Remove the white border of the points
                    ),
                    name=f"{shape} (highlighted)",
                )
            )

        # Update layout to remove the background and set other properties
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False
                ),
            ),
            title=f"{algorithm.upper()} of Decision Making Neuron Activations",
            showlegend=True,
        )

        return fig

    # Run the Dash app
    app.run_server(debug=True, port=port)
