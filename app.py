import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# Create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load Data
df = pd.read_csv("./data/power_and_oat.csv")

def create_dropdown_options(df):
    ls = list(df.columns)
    options = [{"label": str(col_name), "value": str(col_name)} for col_name in ls]
    return ls, options

x_filter_list, x_filter_options = create_dropdown_options(df)
y_filter_list, y_filter_options = create_dropdown_options(df)

# Create app layout
app.layout = html.Div(
    [
        html.Div([
                html.H1("Rietz Dash Example",),
                html.H4("Sample Meter Data",)
            ]),
        html.Div([
            html.H6("Select Variable for X-Axis"),
            dcc.Dropdown(
                id="x_axis",
                value="OAT",
                className="dcc_control",
                multi=False,
                options=x_filter_options
            ),
            html.H6("Select Variable for Y-Axis"),
            dcc.Dropdown(
                id="y_axis",
                value="kW",
                className="dcc_control",
                multi=False,
                options=y_filter_options
            ),
            dcc.Graph(id="main_graph")
        ])
    ]
)

# Callback for main map plot
@app.callback( # reference elements by id
    [Output("main_graph", "figure")],
    [Input("x_axis", "value"), Input("y_axis", "value")]
)
def make_main_figure(x_axis, y_axis):
    data = go.Scatter(
        x=df[x_axis],
        y=df[y_axis],
        mode="markers",
    )

    layout = {
        "title_text": f"{y_axis} vs {x_axis}",
        "xaxis": {"title": x_axis},
        "yaxis": {"title": y_axis}
    }

    figure = go.Figure(data=data, layout=layout)
    return [figure]

# Main
if __name__ == "__main__":
    app.run_server(debug=True)