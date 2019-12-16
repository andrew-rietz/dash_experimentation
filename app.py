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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# Create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set global vars. NOTE: These are independent of session
polynomial_degrees = [{"label": i, "value": i} for i in range(1, 6)]

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
                    "Drag and Drop CSV or ",
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
            ),
        ]),
        html.Div([
            dash_table.DataTable(
                id="output-data-table",
                editable=True,
                filter_action="native",
                style_table={
                    'maxHeight': '200px',
                    'overflowY': 'scroll',
                    'overflowX': 'scroll',
                    'whiteSpace': 'normal'
                },
                fixed_rows={"headers": True, "data": 0},
                filter_query='',
        )]),
        html.Div([
            html.H6("Select Variable for X-Axis"),
            dcc.RadioItems(
                id="x_axis",
                value="",
                className="dcc_control",
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
                inputStyle={
                    "margin-left": "10px",
                    "margin-right": "2px",
                }                
            ),
            html.H6("Select degree of polynomial for trendline"),
            dcc.RadioItems(
                id="poly_degrees",
                value=1,
                options=polynomial_degrees,
                className="dcc_control",
                inputStyle={
                    "margin-left": "10px",
                    "margin-right": "2px",
                }                
            ),
            dcc.Graph(id="main_graph")
        ])
    ]
)


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def convert_operator(operator):
    operator_map = {
        'ge': '>=',
        'le': '<=',
        'lt': '<',
        'gt': '>',
        'ne': '!=',
        'eq': '='
    }
    return operator_map.get(operator.strip()) or f"({operator})"


def create_dropdown_options(df):
    ls = list(df.columns)
    options = [{"label": str(col_name), "value": str(col_name)} for col_name in ls]
    return ls, options


def parse_csv_contents(file_content, filename):
    content_type, content_string = file_content.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

def filter_df(base_df, filter):
    df = base_df.copy(deep=True)
    filtering_expressions = filter.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            df = df.loc[getattr(df[col_name], operator)(filter_value)]
        elif operator == 'contains':
            df = df.loc[df[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            df = df.loc[df[col_name].str.startswith(filter_value)]

    return df

def fit_polynomial(df, x_col, y_col, degree):
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(df[[x_col]])

    regression_model = LinearRegression()
    regression_model.fit(x_poly, df[y_col])
    poly_prediction = regression_model.predict(x_poly)

    r_squared = f"{r2_score(df[y_col], poly_prediction):.4f}"
    intercept = regression_model.intercept_
    coefficients = regression_model.coef_
    coeff_str = " + ".join([
        f"[{x_col}]*[{coefficients[i]:.4f}]^{i}]<br>" for i in range(1, len(coefficients))
    ])
    fit_equation = (
        f"[{y_col}] = {intercept:.4f} + {coeff_str}"
    )
    return pd.DataFrame(data=poly_prediction, index=df.index), r_squared, fit_equation

def dash_table_filters(filter):
    filter_str = f"Filters:<br>{'-' * 15}<br>"
    filtering_expressions = filter.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        filter_str += f"[{col_name}]  {convert_operator(operator)} [{filter_value}]<br>"
    return filter_str

def describe_filter_annotation(layout, filter_str):        
    filter_annot_text = f"{filter_str}"
    layout["margin"]["r"] = layout["margin"]["r"] + 5 * max([len(row) for row in filter_annot_text.split("<br>")])
    layout["width"] = layout["width"] + 5 * max([len(row) for row in filter_annot_text.split("<br>")])
    layout["annotations"] = (*layout["annotations"], {
        "x": 1.05,
        "y": 0.75,
        "showarrow": False,
        "text": filter_annot_text,
        "xref": "paper",
        "yref": "paper",
        "align": "left",
        "yanchor": "top",
        "xanchor": "left"
    })
    return layout


def describe_regression_annotation(layout, r_squared, fit_equation):
    regression_annot_text = f"Regression:<br>{'-' * 15}<br>{fit_equation}<br>(R-Squared: {r_squared})"
    layout["margin"]["b"] = layout["margin"]["b"] + 15 * len(regression_annot_text.split("<br>"))
    layout["height"] = layout["height"] + 15 * len(regression_annot_text.split("<br>"))
    layout["annotations"] = (*layout["annotations"], {
        "x": 0,
        "y": -0.15,
        "showarrow": False,
        "text": regression_annot_text,
        "xref": "paper",
        "yref": "paper",
        "align": "left",
        "yanchor": "top"
    })
    return layout

# Callback for data table display
@app.callback([Output("output-data-table", "data"),
               Output("output-data-table", "columns"),
               Output("x_axis", "value"),
               Output("y_axis", "value")],
              [Input("user_upload", "contents")],
              [State("user_upload", "filename")]
)
def update_output(contents, filename):
    if contents is None:
        return [], [], "", ""
    
    df = parse_csv_contents(contents, filename)
    columns = [{"name": c_name, "id": c_name, "deletable": True, "renamable": True}
               for c_name in df.columns]

    return df.to_dict("records"), columns, columns[0].get("name"), columns[0].get("name")

@app.callback(
    [Output("x_axis", "options"),
     Output("y_axis", "options")],
    [Input('output-data-table', 'data')]
)
def set_x_y_options(table_rows):
    df = pd.DataFrame(table_rows)
    if(df.empty or len(df.columns) < 1):
        return [], []
    
    x_filter_list, x_filter_options = create_dropdown_options(df)
    y_filter_list, y_filter_options = create_dropdown_options(df)

    return x_filter_options, y_filter_options


# Callback for main plot
@app.callback(
    [Output("main_graph", "figure")],
    [Input("x_axis", "value"), 
     Input("y_axis", "value"),
     Input("poly_degrees", "value"),
     Input('output-data-table', 'data'),
     Input('output-data-table', 'filter_query')]
)
def make_main_figure(x_axis, y_axis, poly_degrees, table_rows, filter):
    df = pd.DataFrame(table_rows)
    if(df.empty or len(df.columns) < 1):
        return [go.Figure(
            data=go.Scatter(
                x=[],
                y=[],
                mode="markers",
            ),
            layout={}
        )]

    df = filter_df(df, filter)
    data = [go.Scatter(
        x=df[x_axis],
        y=df[y_axis],
        mode="markers",
        name=y_axis,
    )]

    layout = {
        "title_text": f"{y_axis} vs {x_axis}",
        "xaxis": {"title": x_axis, "automargin": True},
        "yaxis": {"title": y_axis, "automargin": True},
        "annotations": (),
        "autosize": False,
        "width": 850,
        "height": 650,
        "margin": {
            "b": 100,
        },
        "paper_bgcolor": "LightSteelBlue"
    }

    if((x_axis != df.columns[0]) and (y_axis != df.columns[0])):
        regression_line_data, r_squared, fit_equation = fit_polynomial(df, x_axis, y_axis, poly_degrees)
        df["regression"] = regression_line_data
        data.append(go.Scatter(
            x=df[x_axis],
            y=df["regression"],
            mode="markers",
            name="Regression Line",
        ))
        layout = describe_regression_annotation(layout, r_squared, fit_equation)

    if filter:
        filter_str = dash_table_filters(filter)
        layout = describe_filter_annotation(layout, filter_str)

    figure = go.Figure(data=data, layout=layout)
    return [figure] 

# Main
if __name__ == "__main__":
    app.run_server(debug=True)