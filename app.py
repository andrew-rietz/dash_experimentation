import pandas as pd
import dash
import io
import base64
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State


# Create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create app layout
app.layout = html.Div(
    [
        html.Div([
                html.H1("Rietz Dash Example",),
                html.H4("Sample Meter Data",)
            ]),
        html.Div([
            dcc.Upload(
                id="user_upload",
                children=html.Div([
                    "Drag and Drop CSV or",
                    html.A("Select CSV Files")
                ]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px"
                },
                # Allow multiple files to be uploaded
                # multiple=True
            ),
        ]),
        html.Div(id="output-data-table"),
        html.Div([
            html.H6("Select Variable for X-Axis"),
            dcc.RadioItems(
                id="x_axis",
                value="",
                className="dcc_control",
                # options=x_filter_options,
                inputStyle={
                    "margin-left": "10px",
                    "margin-right": "2px",
                }
            ),
            html.H6("Select Variable for Y-Axis"),
            dcc.RadioItems(
                id="y_axis",
                value="",
                className="dcc_control",
                # options=y_filter_options,
                inputStyle={
                    "margin-left": "10px",
                    "margin-right": "2px",
                }                
            ),
            dcc.Graph(id="main_graph")
        ])
    ]
)

df = pd.DataFrame()


def create_dropdown_options(df):
    ls = list(df.columns)
    options = [{"label": str(col_name), "value": str(col_name)} for col_name in ls]
    return ls, options


def parse_csv_contents(file_content, filename):
    content_type, content_string = file_content.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    # try:
    #     if ".csv" in filename:
    #         df = pd.read_csv(
    #             io.StringIO(decoded.decode("utf-8"))
    #         )
    #     else:
    #         return "The provided file must be in CSV format"
    # except Exception as e:
    #     print(e)
    #     return html.Div([
    #         "There was an error processing this file."
    #     ])
    # return df


@app.callback(Output("output-data-table", "children"),
            #   [Input("user_upload", "file_content")],
            #   [State("user_upload", "filename")]
)
def load_table(df, columns, filename):
    return [
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=columns,
            editable=True,
            filter_action="native",
            row_selectable="multi",
            style_as_list_view=True,
        )
    ]


# Callback for data table display
@app.callback([Output("x_axis", "options"),
               Output("y_axis", "options")],
              [Input("user_upload", "contents")],
              [State("user_upload", "filename")]
)
def update_output(contents, filename):
    if contents is None:
        return [], []
    
    df = parse_csv_contents(contents, filename)
    columns = [{"name": c_name, "id": c_name}
               for c_name in df.columns]
    columns["deletable"] = True
    columns["renamable"] = True

    x_filter_list, x_filter_options = create_dropdown_options(df)
    y_filter_list, y_filter_options = create_dropdown_options(df)

    load_table(df, columns, filename)

    
    return x_filter_options, y_filter_options


# Callback for main plot
@app.callback( # reference elements by id
    [Output("main_graph", "figure")],
    [Input("x_axis", "value"), Input("y_axis", "value")]
)
def make_main_figure(df, x_axis, y_axis):
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