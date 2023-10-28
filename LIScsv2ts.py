"""
Task: Import water quality data from multiple source csvs, split them by Station and Parameter TS
Created on %(date)s'

@author: Shubhneet Singh 
ssin@dhigroup.com
DHI,US
"""

import os
import re
import time
import math
import numpy as np
import pandas as pd
from datetime import date

task = 'Task: Import data from multiple source csvs, split them by Station and Parameter TS\n'
day = date.today().strftime("%B%d, %Y")
tool_starttime = time.time()

#Task directory
wdir = r"C:\Users\ssin\OneDrive - DHI\Desktop\LIS\\"
os.chdir(wdir)

# Data files:
listdir = os.listdir(wdir)

# Create log for comments and exceptions:
log_file = open("Readme.txt","w+")
log_file.write('{} \nDeveloped by Shubhneet Singh\nssin@dhigroup.com\n{}, 2021\n\n'.format(task, day))

datasummary_csv = r'C:\Users\ssin\OneDrive - DHI\Desktop\LIS\4 Database\DataSummary.csv'
# Check if the file already exists
if os.path.isfile(datasummary_csv):
    # If it does, delete it
    os.remove(datasummary_csv)
pd.DataFrame().to_csv(datasummary_csv, index=False)

unit_lookup = {
    "Water Temperature": "Celsius",
    "Salinity": "psu",
    "Total Chlorophyll": "mg/L",
    "Biochemical Oxygen Demand (5-day)": "mg O2/L",
    "Dissolved Phosphate": "mg P/L",
    "Dissolved Ammonia": "mg N/L",
    "Nitrate and Nitrite": "mg N/L",
    "Dissolved Silica": "mg Si/L",
    "Dissolved Oxygen": "mg O2/L",
    "Total Particulate Organic Carbon": "mg C/L",
    "Total Dissolved Organic Carbon": "mg C/L",
    "Total Suspended Solids": "mg/L",
    "Total Particulate Organic Nitrogen": "mg N/L",
    "Total Particulate Organic Phosphorus": "mg P/L",
    "Biogenic Silica": "mg Si/L",
    "Total Organic Carbon": "mg C/L",
    "Total Nitrogen": "mg N/L",
    "Total Phosphorus": "mg P/L",
    "Total Kjeldahl Nitrogen": "mg N/L"}

       
#%% Functions: 
    
def get_CTDEEPSummer():    

    agency = 'CTDEEP Summer'  
    agency_path = r"2 Data\CTDEEP Summer\CTDEEP Summer Stations Data.csv"
    agency_df = pd.read_csv(agency_path,                        
                            index_col = 'station',
                            usecols = ['date', 'time','timezone',
                                       'station','CharacteristicName',
                                       'result', 'depth', 'resultunit'],
                            skipinitialspace = True,
                            skip_blank_lines = True,
                            on_bad_lines = 'warn',
                            memory_map = True)
    
    agency_df['datetime'] = pd.to_datetime(agency_df['date'] + ' ' + agency_df['time'])
    agency_df = agency_df.drop(columns=['date', 'time'])

    agency_duplicate = agency_df[agency_df.duplicated()]
    print('\n# duplicate entries in {} = {}'.format(agency, len(agency_duplicate)))
    agency_df = agency_df.drop_duplicates()

    agency_dic = agency_df.groupby(['station'])\
        .apply(lambda x: x.groupby('CharacteristicName')\
        .apply(lambda x: x.set_index('datetime').to_dict())\
        .to_dict())\
        .to_dict()
    
    parameter_map = {'Dissolved oxygen (DO)': 'Dissolved Oxygen',         
                    'Salinity':'Salinity',
                    'Temperature, water': 'Water Temperature',
                    'Ammonia': 'Dissolved Ammonia',
                    'Total Particulate Nitrogen': 'Total Particulate Organic Nitrogen',
                    'Phosphorus, Particulate Organic': 'Total Particulate Organic Phosphorus',
                    'Inorganic nitrogen (nitrate and nitrite)': 'Nitrate and Nitrite',
                    'Total Particulate Carbon': 'Total Particulate Organic Carbon',
                    'Organic carbon': 'Total Dissolved Organic Carbon',
                    'Phosphorus': 'Total Phosphorus',
                    'Biogenic Silica': 'Biogenic Silica',
                    'Nitrogen': 'Total Nitrogen',
                    'Silica': 'Dissolved Silica', 
                    'Orthophosphate': 'Dissolved Phosphate',
                    'Total suspended solids': 'Total Suspended Solids',
                    'Chlorophyll a': 'Total Chlorophyll'}    
    
    csvoutdir = f'3 Data CSV Split\{agency}'
    if not os.path.isdir(csvoutdir):
        os.makedirs(csvoutdir)
        
    reference_dict = {}       
    for s, stn in enumerate(agency_dic.keys()):
        station_folder = os.path.join(csvoutdir,stn)
        if not os.path.isdir(station_folder):
            os.makedirs(station_folder)
        reference_dict[stn] = {}
        for v, var in enumerate(agency_dic[stn].keys()):
            if var in parameter_map.keys():
                df = pd.DataFrame.from_dict(agency_dic[stn][var])           
                df.index = pd.to_datetime(df.index)
                df = df.sort_index() 
                # Add three hours to index where 'timezone' is 'PDT'
                df.index = df.index + pd.to_timedelta(df['timezone'].eq('PDT') * 3, unit='h')
                df = df.drop(['timezone', 'CharacteristicName'], axis=1)

                rename_dict = {'depth': 'Depth',
                               'resultunit': 'Unit', 
                               'result': parameter_map[var]}                
                # Use the rename function to rename the columns
                df.rename(columns=rename_dict, inplace=True)
                df.index.name = 'Datetime'
                
                var_df_len = len(df)
                df = df.dropna(how='all').drop_duplicates() 
                df = df.loc[~df.index.duplicated(keep='first')]
                if len(df)-var_df_len > 0:                         
                    print('\n# duplicate entries in {} = {}'.format(var, len(df)-var_df_len))                               
                
                for col in [parameter_map[var], 'Depth']:                
                    outdf = df[col].dropna()                    
                    outdf = outdf.replace('<', '', regex=True)
                    outdf = outdf.replace('>', '', regex=True)                    
                    outdf = outdf.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(how='any')      
                    outdf  = outdf[~outdf.index.isnull()]
                    if col == parameter_map[var]: vardf_len = len(outdf)
                    
                    tag = [' Depth' if col == 'Depth' else ''][0]                    
                    if len(outdf) >0:
                        filename = (parameter_map[var] + f'{tag}.csv')
                        filename = filename.replace('/', '').replace('(', '').replace(')', '').replace(",", '')
                        filepath = os.path.join(station_folder, filename)
                        outdf.to_csv(filepath, sep=',', date_format = "%m/%d/%Y %H:%M:%S")
                        
                        if col == parameter_map[var]:
                            reference_dict[stn][parameter_map[var]] = {}
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'False'
                            reference_dict[stn][parameter_map[var]]['Unit'] = df['Unit'][0]
                        if (vardf_len>0) & (col == 'Depth'):
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'True'                                          
                    # else:
                    #     print(f'No data for {stn}, {var}: {col}')
                    
    # Reformatting the nested dictionary
    formatted_dict = []
    for station, measures in reference_dict.items():
        for measure, data in measures.items():
            formatted_dict.append({
                'Station': station,
                'Parameter': measure,
                **data })    
    # Convert to dataframe
    datasummary = pd.json_normalize(formatted_dict)
    datasummary['Source'] = 'CTDEEP Summer'
    datasummary = datasummary[['Source','Station', 'Parameter', 'Depth', 'Unit']]
    datasummary.to_csv(datasummary_csv , mode='a', index=False) 

                
def get_CTDEEPYear():         
    agency = 'CTDEEP Year Round'  
    agency_path = r"2 Data\CTDEEP Year Round\CTDEEP YearRound Stations Data.csv"
    agency_df = pd.read_csv(agency_path,                        
                            parse_dates = ['date'],
                            index_col = 'station',
                            usecols = ['station', 'date',
                                       'NB_depth','B_depth','M_depth', 'M2_depth', 'M3_depth', 'S_depth',
                                       'NB_temp','B_temp','M_temp', 'M2_temp', 'M3_temp', 'S_temp',
                                       'NB_sal','B_sal', 'M_sal','M2_sal', 'M3_sal', 'S_sal',
                                       'NB_dow','B_dow','M_dow', 'M2_dow','M3_dow', 'S_dow'],
                            skipinitialspace = True,
                            skip_blank_lines = True,
                            on_bad_lines = 'warn',
                            memory_map = True)    
    # wildcards = {'_temp': 'Water Temperature',
    #              '_sal': 'Salinity',
    #              '_dow': 'Dissolved Oxygen'}    
    
    # for wildcard, descrption in wildcards.items():
    #     # Extract columns with "_temp" in their names
    #     wildcard_columns = [col for col in agency_df.columns if wildcard in col]
    #     # Combine columns and keep the first value from each row
    #     agency_df[descrption] = agency_df[wildcard_columns].apply(lambda row: row[row.first_valid_index()] if not row.isnull().all() else np.nan, axis=1)
    #     # Remove the individual temperature columns
    #     agency_df.drop(wildcard_columns, axis=1, inplace=True) 
    
    # parameter_map = {'Dissolved Oxygen': 'Dissolved Oxygen',         
    #                 'Salinity':'Salinity',
    #                 'Water Temperature': 'Water Temperature'}  
    
    parameter_map = {   'NB_temp': 'NB Water Temperature',
                        'B_temp': 'B Water Temperature',
                        'M_temp': 'M Water Temperature',
                        'M2_temp': 'M2 Water Temperature',
                        'M3_temp': 'M3 Water Temperature',
                        'S_temp': 'S Water Temperature',
                        'NB_sal': 'NB Salinity',
                        'B_sal': 'B Salinity',
                        'M_sal': 'M Salinity',
                        'M2_sal': 'M2 Salinity',
                        'M3_sal': 'M3 Salinity',
                        'S_sal': 'S Salinity',
                        'NB_dow': 'NB Dissolved Oxygen',
                        'B_dow': 'B Dissolved Oxygen',
                        'M_dow': 'M Dissolved Oxygen',
                        'M2_dow': 'M2 Dissolved Oxygen',
                        'M3_dow': 'M3 Dissolved Oxygen',
                        'S_dow': 'S Dissolved Oxygen'}    
    
    agency_duplicate = agency_df[agency_df.duplicated()]
    print('\n# duplicate entries in {} = {}'.format(agency, len(agency_duplicate)))
    agency_df = agency_df.drop_duplicates()
    agency_dict = {station: station_df.set_index('date')  for station, station_df in agency_df.groupby('station')}
           
    csvoutdir = f'3 Data CSV Split\{agency}'
    if not os.path.isdir(csvoutdir):
        os.makedirs(csvoutdir)
    
    reference_dict = {}  
    for s, stn in enumerate(agency_dict.keys()):
        station_folder = os.path.join(csvoutdir,stn)
        if not os.path.isdir(station_folder):
            os.makedirs(station_folder)
        reference_dict[stn] = {}
        for v, var in enumerate(agency_dict[stn].columns):
            if var in parameter_map.keys():
                depthcol = re.sub(r'_.*', '_depth', var )
                df = agency_dict[stn][[var, depthcol]]            
                df.index = pd.to_datetime(df.index)
                df = df.sort_index() 
                depthcol_mappedname = re.sub(r'_.*', ' Depth', var)
                df.columns = [parameter_map[var], depthcol_mappedname]
                df.index.name = 'Datetime'
                
                var_df_len = len(df)
                df = df.dropna(how='all')
                df = df.loc[~df.index.duplicated(keep='first')]
                if len(df)-var_df_len > 0:                         
                    print('\n# duplicate entries in {} = {}'.format(var, len(df)-var_df_len))
                df = df.drop_duplicates()
               
                for col in df.columns:    
                    outdf = df[col].dropna()
                    outdf = outdf.replace('<', '', regex=True)
                    outdf = outdf.replace('>', '', regex=True)
                    outdf = outdf.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(how='any')
                    outdf  = outdf[~outdf.index.isnull()]
                    if col == parameter_map[var]: vardf_len = len(outdf)
                    
                    tag = [' Depth' if col == 'Depth' else ''][0]                    
                    if len(outdf) >0:
                        filename = (parameter_map[var] + f'{tag}.csv')
                        filename = filename.replace('/', '').replace('(', '').replace(')', '').replace(",", '')
                        filepath = os.path.join(station_folder, filename)
                        outdf.to_csv(filepath, sep=',', date_format = "%m/%d/%Y %H:%M:%S")                               
                        
                        if col == parameter_map[var]:
                            reference_dict[stn][parameter_map[var]] = {}
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'False'
                            reference_dict[stn][parameter_map[var]]['Unit'] = unit_lookup[parameter_map[var].split(' ', 1)[1]]
                        if (vardf_len>0) & (col == depthcol_mappedname):
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'True'   

                    # if len(outdf) ==0:
                    #     print(f'No data for {stn}, {var}: {col}')

    # Reformatting the nested dictionary
    formatted_dict = []
    for station, measures in reference_dict.items():
        for measure, data in measures.items():
            formatted_dict.append({
                'Station': station,
                'Parameter': measure,
                **data })    
    # Convert to dataframe
    datasummary = pd.json_normalize(formatted_dict)
    datasummary['Source'] = 'CTDEEP Year Round'
    datasummary = datasummary[['Source', 'Station', 'Parameter', 'Depth', 'Unit']]
    datasummary.to_csv(datasummary_csv , mode='a', index=False, header = False) 
                      
                                                                                   
def get_IEC():  
    agency = 'IEC'  
    agency_path = r"2 Data\Interstate Environmental Commission\IEC_Water_Quality_and_Nutrient_Data.csv"
    agency_df = pd.read_csv(agency_path,                        
                            parse_dates = {'datetime': ['DATE','TIME_24H']},
                            index_col = 'STATION_ID',
                            usecols = ['DATE','TIME_24H','STATION_ID', 'DEPTH_M',
                                       'DISSOLVED_OXYGEN_MG_L', 'TEMPERATURE_C',
                                       'SALINTY_PSU', 'CHLA_UG_L','SECCHI_DEPTH_M', 
                                       'BODs_MG_L', 'TSS_MG_L', 'AMMONIA-AMMONIUM_MG_L',
                                       'NITRITE_NITRATE_MG_L', 'PARTICULATE_N_MG_L',
                                       'ORTHOPHOSPHATE_MG_L', 'TOTAL_DISSOLVED_ N_MG_L',
                                       'TOTAL_DISSOLVED_P_MG_L', 'PARTICULATE_P_MG_L', 'DOC_MG_L',
                                       'PARTICULATE_C_MG_L', 'DISSOLVED_SILICA_MG_L',
                                       'BIOGENIC_SILICA_MG_L'],                                       
                            skipinitialspace = True,
                            skip_blank_lines = True,
                            on_bad_lines = 'warn',
                            memory_map = True,
                            encoding='latin-1')    
    
    agency_duplicate = agency_df[agency_df.duplicated()]
    print('\n# duplicate entries in {} = {}'.format(agency, len(agency_duplicate)))
    agency_df = agency_df.drop_duplicates()
    agency_dict = {station: station_df.set_index('datetime')  for station, station_df in agency_df.groupby('STATION_ID')}
    
    parameter_map = {'DISSOLVED_OXYGEN_MG_L': 'Dissolved Oxygen',         
                   'SALINTY_PSU':'Salinity',
                   'TEMPERATURE_C': 'Water Temperature',
                   'AMMONIA-AMMONIUM_MG_L': 'Dissolved Ammonia',
                   'PARTICULATE_N_MG_L': 'Total Particulate Organic Nitrogen',
                   'PARTICULATE_P_MG_L': 'Total Particulate Organic Phosphorus',
                   'NITRITE_NITRATE_MG_L': 'Nitrate and Nitrite',
                   'PARTICULATE_C_MG_L': 'Total Particulate Organic Carbon',
                   'DOC_MG_L': 'Total Organic Carbon',
                   'Phosphorus': 'Total Phosphorus',
                   'BIOGENIC_SILICA_MG_L': 'Biogenic Silica',
                   'Nitrogen': 'Total Nitrogen',
                   'DISSOLVED_SILICA_MG_L': 'Dissolved Silica', 
                   'ORTHOPHOSPHATE_MG_L': 'Dissolved Phosphate',
                   'TSS_MG_L': 'Total Suspended Solids',
                   'CHLA_UG_L': 'Total Chlorophyll',
                   'BODs_MG_L': 'Biochemical Oxygen Demand (5-day)'}                   
      
    csvoutdir = f'3 Data CSV Split\{agency}'
    if not os.path.isdir(csvoutdir):
        os.makedirs(csvoutdir)        
    reference_dict = {}  
    for s, stn in enumerate(agency_dict.keys()):
        station_folder = os.path.join(csvoutdir,stn)
        if not os.path.isdir(station_folder):
            os.makedirs(station_folder)
        reference_dict[stn] = {}
        for v, var in enumerate(agency_dict[stn].columns):
            if var in parameter_map.keys():
                df = agency_dict[stn][[var, 'DEPTH_M']]            
                df.index = pd.to_datetime(df.index)
                df = df.sort_index().dropna()
                df.columns = [parameter_map[var],'Depth']
                df.index.name = 'Datetime'    
                
                var_df_len = len(df)
                df = df.dropna(how='all')
                df = df.loc[~df.index.duplicated(keep='first')]
                if len(df)-var_df_len > 0:                         
                    print('\n# duplicate entries in {} = {}'.format(var, len(df)-var_df_len))
                df = df.drop_duplicates()
                    
                for col in df.columns:                
                    outdf = df[col].dropna()
                    outdf = outdf.replace('<', '', regex=True)
                    outdf = outdf.replace('>', '', regex=True)
                    outdf = outdf.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(how='any')
                    outdf  = outdf[~outdf.index.isnull()]
                    if col == parameter_map[var]: vardf_len = len(outdf)
                    
                    tag = [' Depth' if col == 'Depth' else ''][0]                    
                    if len(outdf) >0:
                        filename = (parameter_map[var] + f'{tag}.csv')
                        filename = filename.replace('/', '').replace('(', '').replace(')', '').replace(",", '')
                        filepath = os.path.join(station_folder, filename)
                        outdf.to_csv(filepath, sep=',', date_format = "%m/%d/%Y %H:%M:%S")

                        if col == parameter_map[var]:
                            reference_dict[stn][parameter_map[var]] = {}
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'False'
                            reference_dict[stn][parameter_map[var]]['Unit'] = unit_lookup[parameter_map[var]]
                        if (vardf_len>0) & (col == 'Depth'):
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'True'
                            
                    # if len(outdf) ==0:
                    #     print(f'No data for {stn}, {var}: {col}')

    # Reformatting the nested dictionary
    formatted_dict = []
    for station, measures in reference_dict.items():
        for measure, data in measures.items():
            formatted_dict.append({
                'Station': station,
                'Parameter': measure,
                **data })    
    # Convert to dataframe
    datasummary = pd.json_normalize(formatted_dict)
    datasummary['Source'] = 'IEC'
    datasummary = datasummary[['Source','Station', 'Parameter', 'Depth', 'Unit']]
    datasummary.to_csv(datasummary_csv , mode='a', index=False, header = False) 
                        
def get_NJHDG():
         
    agency = r'NJHDG'  
    agency_path = r"2 Data\New Jersey Harbor Discharge\NJHDG Data 2000-2018.csv"
    agency_df = pd.read_csv(agency_path,                        
                            parse_dates = {'datetime': ['Njdate','Time']},
                            index_col = 'NJsite',
                            usecols = ['NJsite', 'Njdate', 'Time', 'pH', 'pHq', 'TempC',
                                       'Tempq', 'DOmg/L', 'Doq', 'Dopct', 'Dopctq', 'Salinity', 'Salq',
                                       'Secchi depth (ft)', 'Secchiq', 'Fecal Coliform', 'Fecalq',
                                       'Enterococcus', 'Enteroq', 'E.coli', 'E Coliq', 'TSS', 'TSSq', 'TKN',
                                       'TKNq', 'CBOD5', 'CBOD5q', 'NH3', 'NH3q', 'NO2', 'NO2q', 'NO3', 'NO3q',
                                       'Total P', 'Total Pq', 'Ortho P', 'Ortho Pq', 'Chlor-a', 'Chlor-aq',
                                       'DOC', 'DOCq'],                                       
                            skipinitialspace = True,
                            skip_blank_lines = True,
                            on_bad_lines = 'warn',
                            memory_map = True)
    
    agency_df["NO2+NO3"] = agency_df["NO2"] + agency_df["NO3"]
    
    parameter_map = {'TempC': 'Water Temperature',
                     'Salinity':'Salinity',
                     'Chlor-a': 'Total Chlorophyll',
                     'CBOD5': 'Biochemical Oxygen Demand (5-day)',
                     'Ortho P': 'Dissolved Phosphate',
                     'NH3': 'Dissolved Ammonia',
                     "NO2+NO3": 'Nitrate and Nitrite',
                     'DOmg/L': 'Dissolved Oxygen',
                     'DOC': 'Total Dissolved Organic Carbon',
                     'TSS': 'Total Suspended Solids',
                     'Total P': 'Total Phosphorus',
                     'TKN': 'Total Kjeldahl Nitrogen'} 
    
    agency_duplicate = agency_df[agency_df.duplicated()]
    print('\n# duplicate entries in {} = {}'.format(agency, len(agency_duplicate)))
    agency_df = agency_df.drop_duplicates()    
    agency_dict = {station: station_df.set_index('datetime')  for station, station_df in agency_df.groupby('NJsite')}
    
    csvoutdir = f'3 Data CSV Split\{agency}'
    if not os.path.isdir(csvoutdir):
        os.makedirs(csvoutdir)        
    reference_dict = {}  
    for s, stn in enumerate(agency_dict.keys()):
        station_folder = os.path.join(csvoutdir,str(stn))
        if not os.path.isdir(station_folder):
            os.makedirs(station_folder)
        reference_dict[stn] = {}
        for v, var in enumerate(agency_dict[stn].columns):
            if var in parameter_map.keys():
                df = agency_dict[stn][[var, 'Secchi depth (ft)']]           
                df.index = pd.to_datetime(df.index)
                df = df.sort_index().dropna()
                df.columns = [parameter_map[var],'Depth']
                df.index.name = 'Datetime'    
                
                var_df_len = len(df)
                df = df.dropna(how='all')
                df = df.loc[~df.index.duplicated(keep='first')]
                if len(df)-var_df_len > 0:                         
                    print('\n# duplicate entries in {} = {}'.format(var, len(df)-var_df_len))
                df = df.drop_duplicates()
                
                for col in df.columns:                
                    outdf = df[col].dropna()
                    outdf = outdf.replace('<', '', regex=True)
                    outdf = outdf.replace('>', '', regex=True)
                    outdf = outdf.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(how='any')
                    outdf  = outdf[~outdf.index.isnull()]
                    if col == 'Depth':
                        tag = ' Depth'
                        outdf = outdf*0.3048 #Convert depth to m
                    else:
                        tag = ''
                        
                    if col == parameter_map[var]: vardf_len = len(outdf)                 
                    if len(outdf) >0:
                        filename = (parameter_map[var] + f'{tag}.csv')
                        filename = filename.replace('/', '').replace('(', '').replace(')', '').replace(",", '')
                        filepath = os.path.join(station_folder, filename)
                        outdf.to_csv(filepath, sep=',', date_format = "%m/%d/%Y %H:%M:%S")
                        
                        if col == parameter_map[var]:
                            reference_dict[stn][parameter_map[var]] = {}
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'False'
                            reference_dict[stn][parameter_map[var]]['Unit'] = unit_lookup[parameter_map[var]]
                        if (vardf_len>0) & (col == 'Depth'):
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'True'

                    # if len(outdf) ==0:
                    #     print(f'No data for {stn}, {var}: {col}')

    # Reformatting the nested dictionary
    formatted_dict = []
    for station, measures in reference_dict.items():
        for measure, data in measures.items():
            formatted_dict.append({
                'Station': station,
                'Parameter': measure,
                **data})    
    # Convert to dataframe
    datasummary = pd.json_normalize(formatted_dict)
    datasummary['Source'] = 'NJHDG'
    datasummary = datasummary[['Source','Station', 'Parameter', 'Depth', 'Unit']]
    datasummary.to_csv(datasummary_csv , mode='a', index=False, header = False) 
                
def get_NYCDEP():    
        
    agency = r'NYCDEP HS'  
    agency_path = r"2 Data\NYCDEP Harbor Survey\NYCDEP Harbor Water Quality Data.csv"
    agency_df = pd.read_csv(agency_path,                        
                            index_col = 'Sampling Location',
                            usecols = ['Sampling Location','Sample Date', 'Sample Time',
                                   'Top Sample Temperature (ºC)',
                                   'Bottom Sample Temperature (ºC)',
                                   'Site Actual Depth (ft)',
                                   'Top Salinity  (psu)', 
                                   'Bottom Salinity  (psu)',
                                   'CTD (conductivity, temperature, depth profiler) Top Dissolved Oxygen (mg/L)',
                                   'CTD (conductivity, temperature, depth profiler) Bottom Dissolved Oxygen (mg/L)',                                                     
                                   'Winkler Method Top Dissolved Oxygen (mg/L)',
                                   'Winkler Method Bottom Dissolved Oxygen (mg/L)', 'Secchi Depth (ft)',                                   
                                   'Top Nitrate/Nitrite (mg/L)', 'Bottom Nitrate/Nitrite (mg/L)',
                                   'Top Ammonium (mg/L)', 'Bottom Ammonium (mg/L)',
                                   'Top Ortho-Phosphorus (mg/L)', 'Bottom Ortho-Phosphorus (mg/L)',
                                   'Top Total Kjeldhal Nitrogen (mg/L)',
                                   'Bottom Total Kjeldhal Nitrogen (mg/L)',
                                   'Top Silica (mg/L)','Bottom Silica (mg/L)',
                                   'Total Phosphorus(mg/L)',
                                   'Bottom Total Phosphorus (mg/L)', 'Top Total Suspended Solid (mg/L)',
                                   'Bottom Total Suspended Solid (mg/L)',
                                   "Top Active Chlorophyll 'A' (µg/L)",
                                   "Bottom Active Chlorophyll 'A' (µg/L)",
                                   'Chlorophyll Top Sample Field (u/L (YSI)',
                                   'Chlorophyll Bottom Sample Field (u/L (YSI)',
                                   'TOP Total Organic Carbon (mg/L)',
                                   'Top Dissolved Organic Carbon (mg/L)',
                                   'Bottom Dissolved Organic Carbon (mg/L)',
                                   'Top Dissolved Organic Carbon YSI (mg/L)',
                                   'Bottom Dissolved Organic Carbon YSI (mg/L)',
                                   'Top Sample Salinity YSI (psu)', 'Bottom Sample Salinity YSI (psu)',
                                   'Top Sample Temperature  YSI (ºC)',
                                   'Bottom Sample Temperature YSI (ºC)',
                                   'Top Five-Day Biochemical Oxygen Demand(mg/L)',
                                   'Bottom Five-Day Biochemical Oxygen Demand(mg/L)',                                   
                                   'Oakwood BOD Top Sample (mg/L) ',
                                   'Oakwood BOD Bottom Sample(mg/L)',
                                   'Oakwood  Total Suspended Solid Top Sample  (mg/L)',
                                   'Oakwood Total Suspended Solid Bottom Sample (mg/L)'],                                       
                            skipinitialspace = True,
                            skip_blank_lines = True,
                            on_bad_lines = 'warn',
                            memory_map = True)
    
    agency_df['datetime'] = pd.to_datetime(agency_df['Sample Date'] + ' ' + agency_df['Sample Time'])
    agency_df = agency_df.drop(columns=['Sample Date', 'Sample Time'])
    
    agency_duplicate = agency_df[agency_df.duplicated()]
    print('\n# duplicate entries in {} = {}'.format(agency, len(agency_duplicate)))
    agency_df = agency_df.drop_duplicates()        
        
    wildcards_dic = {'Water Temperature': ["Sample Temperature (ºC)", "Sample Temperature  YSI (ºC)"],
                     'Total Chlorophyll': [ "Chlorophyll"],
                     'Biochemical Oxygen Demand (5-day)': ['Five-Day Biochemical Oxygen Demand(mg/L)'],
                     'Dissolved Phosphate': ['Ortho-Phosphorus (mg/L)'],
                     'Dissolved Ammonia' : ['Ammonium (mg/L)'],
                     'Nitrate and Nitrite': ['Nitrate/Nitrite (mg/L)'],
                     'Dissolved Silica': ['Silica (mg/L)'],
                     'Dissolved Oxygen': ["Dissolved Oxygen (mg/L)"],
                     'Total Suspended Solids':  ['Total Suspended Solid (mg/L)'],
                     'Total Organic Carbon': ['Total Organic Carbon (mg/L)'],
                     'Total Phosphorus': ['Total Phosphorus(mg/L)'],
                     'Total Kjeldahl Nitrogen': ['Total Kjeldhal Nitrogen (mg/L)']}           
    
    for description, wildcards in wildcards_dic.items():
        wildcard_columns = []
        for wildcard in wildcards:
            for col in agency_df.columns:
                if wildcard in col:
                    wildcard_columns.append(col)                  
            
        # Combine columns and keep the first value from each row
        agency_df[description] = agency_df[wildcard_columns].apply(lambda row: row[row.first_valid_index()] if not row.isnull().all() else np.nan, axis=1)
        # Remove the individual temperature columns
        agency_df.drop(wildcard_columns, axis=1, inplace=True)   
    agency_dict = {station: station_df.set_index('datetime')  for station, station_df in agency_df.groupby('Sampling Location')}
    parameter_map = {'Water Temperature': 'Water Temperature',
                     'Total Chlorophyll': 'Total Chlorophyll',
                     'Biochemical Oxygen Demand (5-day)': 'Biochemical Oxygen Demand (5-day)',
                     'Dissolved Phosphate': 'Dissolved Phosphate',
                     'Dissolved Ammonia' : 'Dissolved Ammonia',
                     'Nitrate and Nitrite': 'Nitrate and Nitrite',
                     'Dissolved Silica': 'Dissolved Silica',
                     'Dissolved Oxygen': 'Dissolved Oxygen',
                     'Total Suspended Solids':  'Total Suspended Solids',
                     'Total Organic Carbon': 'Total Organic Carbon',
                     'Total Phosphorus': 'Total Phosphorus',
                     'Total Kjeldahl Nitrogen': 'Total Kjeldahl Nitrogen'}            
        
    csvoutdir = f'3 Data CSV Split\{agency}'
    if not os.path.isdir(csvoutdir):
        os.makedirs(csvoutdir)
    
    reference_dict = {}      
    for s, stn in enumerate(agency_dict.keys()):
        station_folder = os.path.join(csvoutdir,stn)
        
        reference_dict[stn] = {}
        for v, var in enumerate(agency_dict[stn].columns):
            if var in parameter_map.keys():
                df = agency_dict[stn][[var, 'Site Actual Depth (ft)']]           
                df.index = pd.to_datetime(df.index)
                df = df.sort_index().dropna()
                df.columns = [parameter_map[var],'Depth']
                df.index.name = 'Datetime'
                
                var_df_len = len(df)
                df = df.dropna(how='all')
                df = df.loc[~df.index.duplicated(keep='first')]
                if len(df)-var_df_len > 0:                         
                    print('\n# duplicate entries in {} = {}'.format(var, len(df)-var_df_len))
                df = df.drop_duplicates()
                
                for col in df.columns:                
                    outdf = df[col].dropna()
                    outdf = outdf.replace('<', '', regex=True)
                    outdf = outdf.replace('>', '', regex=True)
                    outdf = outdf.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(how='any')
                    outdf  = outdf[~outdf.index.isnull()]
                    if col == 'Depth':
                        tag = ' Depth'
                        outdf = outdf*0.3048 #Convert depth to m
                    else:
                        tag = ''
                        
                    if col == parameter_map[var]: vardf_len = len(outdf)                
                    if len(outdf) >0:                            
                        if not os.path.isdir(station_folder):
                            os.makedirs(station_folder)
                        filename = (parameter_map[var] + f'{tag}.csv')
                        filename = filename.replace('/', '').replace('(', '').replace(')', '').replace(",", '')
                        filepath = os.path.join(station_folder, filename)
                        outdf.to_csv(filepath, sep=',', date_format = "%m/%d/%Y %H:%M:%S")
                        
                        if col == parameter_map[var]:
                            reference_dict[stn][parameter_map[var]] = {}
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'False'                            
                            reference_dict[stn][parameter_map[var]]['Unit'] = unit_lookup[parameter_map[var]]
                        if (vardf_len>0) & (col == 'Depth'):
                            reference_dict[stn][parameter_map[var]]['Depth'] = 'True'

                    # if len(outdf) ==0:
                    #     print(f'No data for {stn}, {var}: {col}')

    # Reformatting the nested dictionary
    formatted_dict = []
    for station, measures in reference_dict.items():
        for measure, data in measures.items():
            formatted_dict.append({
                'Station': station,
                'Parameter': measure,
                **data})    
    # Convert to dataframe
    datasummary = pd.json_normalize(formatted_dict)
    datasummary['Source'] = 'NYCDEP HS'
    datasummary = datasummary[['Source','Station', 'Parameter', 'Depth', 'Unit']]
    datasummary.to_csv(datasummary_csv , mode='a', index=False, header = False) 
    
#%% Task:     

get_CTDEEPSummer()
get_CTDEEPYear()
get_IEC()
get_NJHDG()
get_NYCDEP()

#%%
tool_endtime = time.time()
print('\n\n############\n')
print('\nTime taken: {}'.format(str(round(((tool_endtime - tool_starttime)),0))) + ' seconds')

log_file.write('\n\n############\n')
log_file.write('\nTime taken: {}'.format(str(int((tool_endtime - tool_starttime)))) + ' seconds')
log_file.close() 
