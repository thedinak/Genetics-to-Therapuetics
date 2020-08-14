import pandas as pd
import os
import tarfile
import glob
import json


def unzip_rna_seq_data(file_name, desired_folder_name):
    ''' Downloaded RNA files are tarfiles, this unzips them'''
    if 'tar' in file_name:
        open_tar = tarfile.open(file_name)
        open_tar.extractall(f'{desired_folder_name}')
        open_tar.close()
    else:
        print('Not a tarfile')


def unzip_individual_rna_seq_files(root_dir):
    ''' Tarfile unzip results in gz files, which need to be further unzipped'''
    files_to_unpack = []
    dfs = []
    meta_data_file = ''.join(glob.glob('**/**metadata.cart**', recursive=True))
    with open(meta_data_file, 'r') as f:
        meta_data = json.load(f)
    convert_filename_caseuuid = {meta_data[i]['file_id']:
                                 meta_data[i]['associated_entities'][0]
                                 ['case_id'] for i in range(0, len(meta_data))}
    # dictionary of file_id:case_id
    for directory in os.listdir(root_dir):
        try:
            for filename in os.listdir(os.path.join(root_dir, directory)):
                if ".gz" in filename:
                    files_to_unpack.append(os.path.join(root_dir,
                                                        directory, filename))
        except NotADirectoryError:
            continue
    for file in files_to_unpack:
        dfs.append(pd.read_csv
                   (file, compression='gzip', sep="\t", names=['gene',
                    convert_filename_caseuuid[os.path.split(os.path.dirname
                                                            (file)[1]),
                    index_col='gene'))
        # these dfs already have the correct case id name
    return files_to_unpack, dfs, convert_filename_caseuuid


def concat_all_rna_seq(dfs):
    ''' Takes each individual rna seq file and concatenates them into one '''
    rna_seq_data = pd.concat(dfs, join="outer", axis=1).T
    if type(rna_seq_data.index[0]) == str:
        rna_seq_data.reset_index(inplace=True)
    return rna_seq_data


def convert_ensg_to_gene_name(dataframe_with_genes):
    '''TCGA data is listed with ensemble names, this converts to gene
    names for greater readability '''
    change_name_file = 'mart_export.txt'
    gene_names = {}
    with open(change_name_file) as fh:
        for line in fh:
            ensg, gene_name = line.split(',', 1)
            gene_names[gene_name.split('.')[0]] = ensg
    dataframe = (dataframe_with_genes.rename
                 (columns=lambda x: x.split('.')[0]).rename(
                  columns=gene_names))
    genes = dataframe.columns[1:-1].tolist()
    return dataframe, genes, gene_names


def concat_rna_to_clinical_data(clinical_dataframe, rna_dataframe):
    ''' Combines clinical data and the rna seq data. Clinical dataframe should
    have bcr_patient_uuid as the index. '''
    full_data = pd.merge(rna_dataframe, clinical_dataframe,
                         how='right', left_on=['index'],
                         right_on=['bcr_patient_uuid'])
    return full_data


def limit_full_data_for_pca(full_data, genes):
    ''' Removes rna seq files where there is no drug name available and limits
    columns to rna seq data, drug name and vital status '''
    limit_full_data = (full_data.loc[(full_data.standard_drugs != '')
                       & (full_data.standard_drugs != '[not available]')
                       & (full_data.standard_drugs != '[unknown]')].copy())

    limit_full_data.dropna(subset=['index'], inplace=True)

    columns_needed = genes+['standard_drugs', 'vital_status']

    return limit_full_data.loc[:, columns_needed]
