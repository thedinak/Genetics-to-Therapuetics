import pandas as pd
import os
import pickle
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


def unzip_individual_rna_seq_file(root_dir):
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
            for filename in os.listdir(root_dir + '/' + directory):
                if ".gz" in filename:
                    files_to_unpack.append(os.path.join(root_dir,
                                                        directory, filename))

        except:
            continue
    for file in files_to_unpack:
        dfs.append(pd.read_csv(
                               file, compression='gzip', sep="\t", names=['gene',
                                      convert_filename_caseuuid[file.split('/')[-2]]],
                                index_col='gene'))
        # these dfs already have the correct case id name
    return files_to_unpack, dfs, convert_filename_caseuuid


# In[6]:


def concat_all_rna_seq(dfs):
    rna_seq_data = pd.concat(dfs, join = "outer", axis = 1).T
    if type(rna_seq_data.index[0])== str:
        rna_seq_data.reset_index(inplace=True)
    return rna_seq_data


# In[19]:


def convert_ensg_to_gene_name(dataframe_with_genes):
    change_name_file = 'mart_export.txt'
    gene_names = {}
    with open(change_name_file) as fh:
        for line in fh:
            engs, gene_name = line.split(',', 1)
            gene_names[gene_name.split('.')[0]] = engs
    dataframe = dataframe_with_genes.rename(columns = lambda x: x.split('.')[0]).rename(columns = gene_names)
    genes = dataframe.columns[1:-1].tolist()
    return dataframe, genes, gene_names



# In[8]:


def concat_rna_to_clinical_data(clinical_dataframe, rna_dataframe):
    full_data = pd.merge(rna_dataframe, clinical_dataframe,how = 'right', left_on =['index'], right_on=['bcr_patient_uuid'])
    return full_data


def limit_full_data_for_pca(full_data, genes):
    limit_full_data = full_data.loc[(full_data.standard_drugs!='')
                                &(full_data.standard_drugs!='[not available]')
                                &(full_data.standard_drugs!='[unknown]')].copy()

    limit_full_data.dropna(subset = ['index'], inplace=True)

    columns_needed = genes+['standard_drugs', 'vital_status']

    return limit_full_data.loc[:,columns_needed]


# In[10]:
unzip_rna_seq_data('/Users/dinakats/Desktop/SPICED/final_proj_git_renew/Genetics-to-Therapuetics/Data/kidney2test/kidney_v2_rna/gdc_download_20200529_001144.112503.tar.gz','spooky')


rootdir_test = '/Users/dinakats/Desktop/SPICED/final_proj_git_renew/Genetics-to-Therapuetics/Data/kidney2test/kidney_v2_rna/spooky'


# In[11]:


kidney_files, kidney_dfs, kidney_metadata_dict = unzip_individual_rna_seq_file(rootdir_test)


# In[12]:


kidney_dfs[0]


# In[13]:


len(kidney_files), len(kidney_dfs)


# In[14]:


kidney_rna_seq = concat_all_rna_seq(kidney_dfs)


# In[15]:


kidney_rna_seq.shape


# In[50]:


kidney_rna_seq.head()


# In[51]:


kidney_rna_rename, genes, gene_names_dict = convert_ensg_to_gene_name(kidney_rna_seq)


# In[52]:


genes = kidney_rna_rename.columns[1:-1].tolist()


# In[53]:


clinical_data_df6= pickle.load(open('clinical_data_df6.pickle','rb'))


# In[54]:


clinical_data_df6.head()


# In[55]:


full_data= concat_rna_to_clinical_data(clinical_data_df6, kidney_rna_rename)


# In[56]:


full_data


# In[57]:


limited_data = limit_full_data_for_pca(full_data, genes)


# In[58]:


limited_data.head()


# In[61]:


limited_data.shape


# In[62]:


limited_data.standard_drugs.value_counts()


# In[63]:


pickle.dump(limited_data, open( "limited_data_0624.pickle", "wb" ) )


# In[ ]:
