from dash import Dash
import dash_bootstrap_components as dbc
from dashboard.components.layout import create_layout

app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           use_pages=True)
server = app.server

app.layout = create_layout(app)



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
