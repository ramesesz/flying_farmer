import datetime
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate

from PIL import Image
import io
import base64

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as T

model_path = r'resnet101_model_best_checkpoint.pth'
load = torch.load(model_path, map_location=torch.device('cpu'))

model = models.resnet101(pretrained=True)
num_ftrs = model.fc.in_features
number_of_classes = 4
model.fc = nn.Linear(num_ftrs, number_of_classes)
model.load_state_dict(load['model'])

classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
mean = [0.4014, 0.3769, 0.3608]
std = [0.2224, 0.2138, 0.2114]

image_transforms = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

styleCss = ['assets/style.css']
logo_path = 'assets/flyingFarmerLogo.png'
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Img(src=logo_path, className='logo-image'),
                html.H1(children='Rice Disease Detection', className='logo-title')
                ], className='logo'),
            html.Div(id='output-image-upload', children='')],
            className='contents-upload'),
        html.Div([
            dcc.Upload(id='upload-image', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                       multiple=True),
            html.Div([
                html.H3(children='Content-Info', className='content-title'),
                html.Div(id='output-filename-upload', children='', className='content-info'),
                html.Div(id='output-date-upload', children='', className='content-info'),
                html.Div(id='output-prediction-upload', children='', className='content-info')],
                className='image-info'),
            html.Div([
                html.H3(children='Response', className='box-title'),
                html.Div(id='output-response', children='', className='box-info'),
                html.H3(children='Category', className='box-title'),
                html.Div(id='output-category', children='', className='box-info'),
                html.H3(children='Scientific Name', className='box-title'),
                html.Div(id='output-scientific-name', children='', className='box-info'),
                html.H3(children='Cause', className='box-title'),
                html.Div(id='output-cause', children='', className='box-info'),
                html.H3(children='Cure', className='box-title'),
                html.Div(id='output-cure', children='', className='box-info'),
                html.H3(children='Prevention', className='box-title'),
                html.Div(id='output-prevention', children='', className='box-info'),
                html.H3(children='Resources', className='box-title'),
                html.Div(id='output-resource', children='', className='box-info')],
                className='recomendation-info')],
            className='sidebar')],
        className='app-container')],
    className='app-body')

# access data from the storage or csv to dataframe
def access_anomalies():
    path = 'anomalies/anomalies.csv'
    data = pd.read_csv(path, sep=';')
    return data

def access_cause():
    path = 'anomalies/causes.csv'
    data = pd.read_csv(path, sep=';')
    return data

def access_cure():
    path = 'anomalies/cure.csv'
    data = pd.read_csv(path, sep=';')
    return data

def access_prevention():
    path = 'anomalies/prevention.csv'
    data = pd.read_csv(path, sep=';')
    return data

def access_resource():
    path = 'anomalies/resource.csv'
    data = pd.read_csv(path, sep=';')
    return data

def resource_extract(ID):
    resource = access_resource()
    resource = resource[resource['ID'] == ID.item()]['resource']
    return resource

def prevention_extract(ID):
    prevention = access_prevention()
    preventions = prevention[prevention['ID'] == ID.item()]['prevention']
    return preventions

def cure_extract(ID):
    cure = access_cure()
    cures = cure[cure['ID'] == ID.item()]['cure']
    return cures

def cause_extract(ID):
    cause = access_cause()
    causes = cause[cause['ID'] == ID.item()]['causes']
    return causes

# Merge the result with combined data
def id_extract_anomalies(result):
    #load all datas
    anomalies = access_anomalies()
    #parse the correct data
    anomalies = anomalies[anomalies['anomalies'] == result]
    ID = anomalies[anomalies['anomalies'] == result]['ID']
    category = anomalies[anomalies['anomalies'] == result]['category']
    response = anomalies[anomalies['anomalies'] == result]['response']
    scientific_name = anomalies[anomalies['anomalies'] == result]['scientific_name']
    return ID, category, response, scientific_name


# classification
def classify(image):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    image = image_transforms(image)
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return classes[predicted.item()]

# visualization

predictions = []
categories = []
responses = []
names = []
causes = []
cures = []
preventions = []
resources = []

@app.callback(Output('output-image-upload', 'children'),
              Output('output-filename-upload', 'children'),
              Output('output-date-upload', 'children'),
              Output('output-prediction-upload', 'children'),
              Output('output-response', 'children'),
              Output('output-category', 'children'),
              Output('output-scientific-name', 'children'),
              Output('output-cause', 'children'),
              Output('output-cure', 'children'),
              Output('output-prevention', 'children'),
              Output('output-resource', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    prediction = 'Healthy'
    if list_of_contents is None:
        raise PreventUpdate
    else:
        for i in list_of_contents:
            encoded_image = i.split(",")[1] # the crucial part of decode base64.
            decoded_image = base64.b64decode(encoded_image)
            bytes_image = io.BytesIO(decoded_image)
            image = Image.open(bytes_image).convert('RGB')
            prediction = str(classify(image))
            predictions.append(prediction)

            if prediction != 'Healthy':
                ID, category, response, scientific_name = id_extract_anomalies(prediction)
                categories.append(category)
                responses.append(response)
                names.append(scientific_name)
                causes.append(cause_extract(ID))
                cures.append(cure_extract(ID))
                preventions.append(prevention_extract(ID))
                resources.append(resource_extract(ID))
            else:
                ID, category, response, scientific_name = id_extract_anomalies(prediction)
                responses.append(response)

        img = html.Div()
        filename = html.H5()
        if prediction == 'Healty':
            for c, n, p, d, r in zip(list_of_contents, list_of_names, predictions, list_of_dates, responses):
                img = html.Img(src=c, style={
                    'max-width' : '65%'
                })
                filename = html.H5(n)
                date = html.H6(datetime.datetime.fromtimestamp(d))
                prediction = html.H5(p)
                response = html.H5(r)
                category = html.H5('')
                name = html.H5('')
                cause = html.H5('')
                cure = html.H5('')
                prevention = html.H5('')
                resource = html.H5('')

            return img, filename, date, prediction, response, category, name, cause, cure, prevention, resource
        else:
            for con, fn, p, d, r, cat, na, cau, cu, pre, res  in zip(list_of_contents, list_of_names, predictions, list_of_dates, responses, categories, names, causes, cures, preventions, resources):
                img = html.Img(src=con, style={
                    'max-width' : '70%'
                })
                filename = html.H5(fn)
                date = html.H6(datetime.datetime.fromtimestamp(d))
                prediction = html.H5(p)
                response = html.H5(r)
                category = html.H5(cat)
                name = html.H5(na)

                cause = html.H5(cau)
                if len(cau) > 1:
                    cause = [html.H5(i) for i in cau]

                cure = html.H5(cu)
                if len(cu) > 1:
                    cure = [html.H5(i) for i in cu]

                prevention = html.H5(pre)
                if len(pre) > 1:
                    prevention = [html.H5(i) for i in pre]

                resource = html.H5(res)
                if len(res) > 1:
                    resource = [html.H5(i) for i in res]

            return img, filename, date, prediction, response, category, name, cause, cure, prevention, resource


if __name__ == '__main__':
    app.run_server(debug=True)
