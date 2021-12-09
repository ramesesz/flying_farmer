import pandas as pd

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
    return

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
    causes = cause[cause['ID'] == ID.item]['causes']
    return causes

# Merge the result with combined data
def id_extract_anomalies(result):
    #load all datas
    anomalies = access_anomalies()
    #parse the correct data
    anomalies = anomalies[anomalies['name'] == result]
    ID = anomalies[anomalies['name'] == result]['ID']
    category = anomalies[anomalies['name'] == result]['category']
    response = anomalies[anomalies['name'] == result]['response']
    scientific_name = anomalies[anomalies['name'] == result]['scientific_name']
    return ID, category, response, scientific_name

if __name__ == '__main__':
    result = 'BrownSpot'
    anomalies = access_anomalies()
    # parse the correct data
    ID = anomalies[anomalies['anomalies'] == result]['ID']
    cure = access_cure()
    cures = cure[cure['ID'] == ID.item()]['cure']
    print(cures)
    '''
    ID = anomalies[anomalies['anomalies'] == result]['ID']
    ID, category, response, scientific_name = id_extract_anomalies(result)
    print(ID)
    #print(cause_extract(ID).item)
    cause = access_cause()
    causes = cause[cause['ID'] == ID.item]['causes']
    print(cause[cause['ID'] == ID.item])
    '''

