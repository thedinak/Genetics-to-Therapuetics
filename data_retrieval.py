import requests
import json
import numpy as np
import pandas as pd
import os
import regex as re
import tarfile
from fuzzywuzzy import process


def get_clinical_data_online(filters_table, fields_table, cases_endpoint):
    '''Fetches and formats clinical data from TCGA into dataframe'''

    params_table = {"filters": json.dumps(filters_table),
                    "fields": ",".join(fields_table),
                    "format": "TSV", "size": "5000"}

    response_table = requests.get(cases_endpoint, params=params_table)
    clinical_data = response_table.content.decode("utf-8")
    clinical_data_processed = clinical_data.replace("\t", ",")
    clinical_data_processed = clinical_data_processed.replace("\r\n", ",")
    cd = clinical_data_processed.split(",")

    # find index with first occurance of a number
    for i, value in enumerate(cd):
        if re.search(r"\d", value) is not None:
            reshape_value = i
            break

    cd_array = np.asarray(cd[0:-1])
    cd_array = cd_array.reshape(-1, reshape_value)
    cd_df = pd.DataFrame(cd_array[1:], columns=cd_array[0])
    column_renaming_dictionary = {value: value.split('.')[-1]
                                  for value in cd_df.columns}
    cd_df.rename(columns=column_renaming_dictionary, inplace=True)
    return cd_df


def get_clinical_data_files_locally(root_dir):
    '''After user has downloaded appropriate clinical data spreadsheets from
    TCGA, formats spreadsheets into dataframes'''
    clinical_dfs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt") and "MANIFEST" not in file:
                clinical_dfs.append(pd.read_csv(os.path.join(root, file),
                                    sep='\t', skiprows=[2], header=1))
    return clinical_dfs


def lowercase_dataframe(dfs):
    '''Changes all contents of clinical dataframes to lowercase, to avoid
    possible future processing issues'''
    lowercase_dfs = []
    for df in dfs:
        lowercase_dfs.append(df.applymap(lambda s: s.lower()
                             if type(s) == str else s))
    return lowercase_dfs


def uuid_index(lowercase_clinical_dfs):
    '''Sets the patient uuid as the index for all clinical data
    to facilitate joining dataframes'''
    for df in lowercase_clinical_dfs:
        if df.index.name != 'bcr_patient_uuid':
            try:
                df.set_index('bcr_patient_uuid', inplace=True)
            except KeyError:
                continue


def limit_to_select_columns(column_set, lowercase_dfs):
    '''Limits columns in clinical data to only relevant columns, drops
    any dataframes that are empty'''
    limited_dfs = []
    for df in lowercase_dfs:
        available_column = [column_name for column_name in column_set
                            if column_name in df.columns]
        limited_dfs.append(df.loc[:, available_column])
    limited_dfs = [df for df in limited_dfs if df.shape[1] != 0]
    return limited_dfs


def concat_dfs_on_patient_uuid(dfs):
    '''Combines all dataframes into one, using the patient uuids for joining'''
    df1 = dfs[0]
    i = 1
    for df in dfs[1:]:
        df1 = df1.join(df, on='bcr_patient_uuid', rsuffix=f'{i}', how='outer')
        i += 1
    df1.drop_duplicates(inplace=True)
    df1.set_index('bcr_patient_uuid', inplace=True)
    return df1


def push_columns_together(column_set, dataframe):
    '''Pushes all relevant information into one columns, ie all drug names
     into one column and clarify vital status'''
    for column_type in column_set:
        subset_group = [column_name for column_name in dataframe.columns
                        if column_type in column_name]
        if column_type != 'karnofsky_performance_score':
            dataframe[column_type] = (dataframe[subset_group].
                                      fillna('').agg(','.join, axis=1).
                                      str.strip(','))
        if column_type == 'karnofsky_performance_score':
            for column in subset_group:
                dataframe[column] = pd.to_numeric(dataframe[column],
                                                  errors='coerce')
            dataframe[column_type] = dataframe[subset_group].max(axis=1)
    # set all patients as dead if they have dead in the combined column
    dataframe['vital_status'] = ['dead' if 'dead' in value else 'alive'
                                 for value in dataframe['vital_status']]
    return dataframe[column_set]


def explode_drug_name_column(dataframe):
    ''' For further analysis, sets each distinct drug name
    as a seperate line in the dataframe '''
    df = dataframe.copy()
    df.drug_name = df.drug_name.str.split(r"\||,")
    exploded_dataframe = df.reset_index().explode('drug_name')
    exploded_dataframe.set_index('bcr_patient_uuid', inplace=True)
    # the explode function makes some accidental duplicates when there are
    # duplicate indices, hence the index reset
    return exploded_dataframe


def get_caseid_txt_file(dataframe, filename):
    '''Generates a text file with uuids of interest - this can
    then be used to easily pull RNA files from TCGA '''
    uuids = dataframe[(dataframe.drug_name != '')
                      & (dataframe.drug_name != '[not available]')
                      & (dataframe.drug_name != '[unknown]')].index
    with open(f'{filename}.txt', 'w') as f:
        f.write('\n'.join(uuids.unique()))
    return uuids


def get_all_clinical_files_for_disease(clinical_data_dataframe,
                                       root_dir_for_files):
    '''Pulls all clinical data files from TCGA that work for the project'''
    files_endpt = "https://api.gdc.cancer.gov/files"
    # only tcga has the correctly formatted clinical files
    tcga_project_ids = [project_id for project_id in
                        clinical_data_dataframe.project_id.value_counts().index
                        if "TCGA" in project_id]

    for project_id in tcga_project_ids:
        print(project_id)
        filters = {
            "op": "and",
            "content": [
                {
                 "op": "in",
                 "content": {
                    "field": "cases.project.program.name",
                    "value": [project_id.split("-")[0]]
                    }
                },
                {
                 "op": "in",
                 "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id]
                    }
                },
                {
                 "op": "in",
                 "content": {
                    "field": "files.data_category",
                    "value": ["clinical"]
                    }
                },
                {
                 "op": "in",
                 "content": {
                    "field": "files.data_format",
                    "value": ["bcr biotab"]
                    }}]
                }

        params = {
                  "filters": json.dumps(filters),
                  "fields": "file_id",
                  "format": "JSON",
                  "size": "20"
        }

        # Here a GET is used, so the filter parameters should
        # be passed as a JSON string.
        response = requests.get(files_endpt, params=params)
        file_uuid_list = []

        # This step populates the download list with the file_ids
        # from the previous query
        for file_entry in json.loads(response
                                     .content.decode("utf-8"))["data"]["hits"]:
            file_uuid_list.append(file_entry["file_id"])

        data_endpt = "https://api.gdc.cancer.gov/data"

        params = {"ids": file_uuid_list}

        response = requests.post(data_endpt, data=json.dumps(params),
                                 headers={"Content-Type": "application/json"})
        response_head_cd = response.headers["Content-Disposition"]
        print(response_head_cd)

        file_name = f'{project_id}.tar.gz'

        with open(os.path.join(root_dir_for_files, file_name),
                  "wb") as output_file:
            output_file.write(response.content)


def unzip_clinical_files(root_dir):
    ''' Downloaded clinical files are unzipped in place '''
    list_of_files_to_unpack = []
    for filename in os.listdir(root_dir):
        try:
            if ".gz" in filename:
                list_of_files_to_unpack.append(os.path.join(root_dir,
                                                            filename))
                full_path = root_dir+"/"+filename
                open_tar = tarfile.open(full_path)
                open_tar.extractall(f'{root_dir}/{filename.split(".")[0]}')
                open_tar.close()
        except FileNotFoundError:
            continue
    return list_of_files_to_unpack


def sort_drug_names(dataframe, drug_name_dictionary):
    ''' Cleans up the drug names based on a dictionary, so all names for a
     drug can be treated as one, ie brand name and generic.
      Also handles misspelled drug names '''
    # create reverse dictionary from given dictlist
    drug_dict = {}

    for key, value in drug_name_dictionary.items():
        for item in value:
            drug_dict[item] = key
    # explode out names in parenthesis
    df = dataframe.copy()
    df.drug_name = df.drug_name.str.split(r"\(")
    exploded_df = df.reset_index().explode('drug_name')
    exploded_df.drug_name = exploded_df.drug_name.str.strip(r'\)')

    # use lists and top drug names to correct for any spelling errors
    drug_name_value_counts = exploded_df[(exploded_df.drug_name != '')
                                         & (exploded_df.drug_name !=
                                         '[not available]')
                                         ].drug_name.value_counts()
    top_used_drugs = (drug_name_value_counts[drug_name_value_counts > 10]
                      .index.tolist())
    correctly_spelled_drug_names = set(top_used_drugs
                                       + list(drug_dict.keys())
                                       + list(drug_dict.values()))

    fuzzy_match_dict = {}
    fuzzywuzzy_threshold = 85
    for drug in exploded_df.drug_name:
        if drug not in correctly_spelled_drug_names and drug != '':
            new_name, score = process.extractOne(drug,
                                                 correctly_spelled_drug_names)
            if score > fuzzywuzzy_threshold:
                fuzzy_match_dict[drug] = new_name

    # use drug dictionary to replace drug names
    exploded_df['standard_drugs'] = (exploded_df.drug_name.
                                     map(fuzzy_match_dict).
                                     fillna(exploded_df['drug_name']))
    exploded_df['standard_drugs'] = (exploded_df.standard_drugs.
                                     map(drug_dict).
                                     fillna(exploded_df['standard_drugs']))
    exploded_df.drop_duplicates(inplace=True)

    return fuzzy_match_dict, exploded_df
