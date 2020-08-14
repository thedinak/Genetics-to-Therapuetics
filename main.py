import data_retrieval
import rna_format
import data_predictions
import example_variables
import os
import pickle


def run_clinical_data_collection():
    kidney_cd_df = (data_retrieval.get_clinical_data_online
                    (example_variables.kidney_filters_table,
                     example_variables.fields_table,
                     example_variables.cases_endpoint))
    print(f'The following TCGA projects have Kidney cancer related data:\n'
          f'{kidney_cd_df.project_id.value_counts()}')
    print("We will now download all appropriate clinical data,\n"
          "please create a directory")
    root_dir_clinical = input("print root directory for clinical files ")
    data_retrieval.get_all_clinical_files_for_disease(kidney_cd_df,
                                                      root_dir_clinical)
    data_retrieval.unzip_clinical_files(root_dir_clinical)
    df1 = data_retrieval.get_clinical_data_files_locally(root_dir_clinical)
    df1 = data_retrieval.lowercase_dataframe(df1)
    data_retrieval.uuid_index(df1)
    df2 = (data_retrieval.limit_to_select_columns
           (example_variables.column_set, df1))
    df3 = data_retrieval.concat_dfs_on_patient_uuid(df2)
    df4 = (data_retrieval.push_columns_together
           (example_variables.column_set, df3))
    df5 = data_retrieval.explode_drug_name_column(df4)
    misspell_dict, df6 = (data_retrieval.sort_drug_names
                          (df5, example_variables.alternative_drug_names))
    uuid_file_name = input("We will now generate a txt file with all "
                           "the uuids of relevant cases\n"
                           "Please enter your preferred file name ")
    data_retrieval.get_caseid_txt_file(df5, uuid_file_name)
    pickle_choice = input("Would you like to save the clinical data "
                          "dataframe for future use? ")
    if pickle_choice.lower() == 'yes':
        name = input("Choose a file name ")
        pickle.dump(df6, open(f'{name}.pickle', "wb"))

    return df6


def run_rna_formatting():
    rna_file = input("Print name of downloaded rna data file ")
    rna_folder_name = input("Print the folder name where unzipped rna seq data"
                            "should be saved ")
    rna_format.unzip_rna_seq_data(rna_file, rna_folder_name)
    rna_data_folder = os.path.join(os.getcwd(), rna_folder_name)
    print(rna_data_folder)
    kidney_files, kidney_dfs, kidney_metadata_dict = (
                              rna_format.unzip_individual_rna_seq_files
                              (rna_data_folder))
    kidney_rna_seq = rna_format.concat_all_rna_seq(kidney_dfs)
    print(f'You have {kidney_rna_seq.shape[0]} '
          f'samples with {kidney_rna_seq.shape[1]} genes')
    kidney_rna_rename, genes, gene_names_dict = (rna_format.
                                                 convert_ensg_to_gene_name
                                                 (kidney_rna_seq))
    pickle_choice = input("Would you like to save the rna data "
                          "dataframe for future use? ")
    if pickle_choice.lower() == 'yes':
        name = input("Choose a file name ")
        pickle.dump(kidney_rna_rename, open(f'{name}.pickle', "wb"))
    return kidney_rna_rename


def combine_rna_clinical(clinical_dataframe, rna_dataframe):
    genes = rna_data.columns[1:-1].tolist()
    full_data = rna_format.concat_rna_to_clinical_data(clinical_data, rna_data)
    limited_data = rna_format.limit_full_data_for_pca(full_data, genes)
    print("Sample counts by drug (top 10) ")
    print(limited_data.standard_drugs.value_counts()[0:10].to_markdown())
    pickle_choice = input("Would you like to save the combined clinical and "
                          "rna dataframe for future use? ")
    if pickle_choice.lower() == 'yes':
        name = input("Choose a file name ")
        pickle.dump(limited_data, open(f'{name}.pickle', "wb"))
    return limited_data


def run_data_predictions(limited_data):
    if limited_data == 0:
        name = input("Input limited data file name ")
        limited_data = pickle.load(open(f'{name}.pickle', 'rb'))
    elif limited_data != 0:
        pass
    df = data_predictions.one_hot_encode_vital_status(limited_data)
    drugs_of_interest = df.standard_drugs.value_counts()[0:3].index.tolist()
    df_drugs = data_predictions.split_dataframe_by_drug(drugs_of_interest, df)
    split_drug_dfs = data_predictions.batch_train_test_split(df_drugs)
    evals_stdev_no_pca, no_pca = (data_predictions.
                                  reduction_metric_optimization
                                  (split_drug_dfs,
                                   reduction_strategy='stdev',
                                   percent_remaining=[0.05, 0.1, 0.25,
                                                      0.5, 0.75]))
    print(evals_stdev_no_pca)
    corr_coeff = (data_predictions.determine_correlation_coefficients
                  (split_drug_dfs))
    evals_corr_no_pca, no_pca = (data_predictions.
                                 reduction_metric_optimization
                                 (split_drug_dfs,
                                  reduction_strategy='corr',
                                  percent_remaining=[0.05, 0.1, 0.25,
                                                     0.5, 0.75],
                                  correlation_coefficients=corr_coeff))
    print(evals_corr_no_pca.keys())
    corr_pca_evals, corr_pca_stats = (data_predictions.
                                      reduction_metric_optimization
                                      (split_drug_dfs,
                                       reduction_strategy='corr',
                                       percent_remaining=[0.05, 0.1, 0.25,
                                                          0.5, 0.75],
                                       correlation_coefficients=corr_coeff,
                                       scale='standard', pca='pca',
                                       threshold=[0.01, 0.2, 0.5, 0.9]))
    print(corr_pca_evals)
    minibatch_evals, minibatch_pca_stats = (data_predictions.
                                            reduction_metric_optimization
                                            (split_drug_dfs,
                                             reduction_strategy='nan',
                                             scale='standard',
                                             pca='sparse',
                                             n_components=[15, 10, 5, 2]))
    print(minibatch_evals)
    eval_list = [evals_stdev_no_pca, evals_corr_no_pca,
                 corr_pca_evals, minibatch_evals]
    run_summary = data_predictions.present_run_summary(eval_list)
    print(run_summary)


if __name__ == "__main__":
    limited_data = 0
    selection = input("Would you like to run the kidney cancer example? ")
    download_status = input("Did you already download the RNA data? ")
    if selection.lower() == 'yes':
        if download_status.lower() == 'no':
            clinical_data = run_clinical_data_collection()
            download_status = input("Have you downloaded the rna seq data? ")
            if download_status.lower() == 'yes':
                rna_data = run_rna_formatting()
                limited_data = combine_rna_clinical(clinical_data, rna_data)
                run_summary = run_data_predictions(limited_data)
            elif download_status.lower() != 'yes':
                with open("data_download_instructions.txt") as instructions:
                    print(instructions.read())
        if download_status.lower() == 'yes':
            clinical_data_status = input("Did you save the clinical "
                                         "dataframe? ")
            rna_data_status = input("Did you save the rna dataframe? ")
            combo_data_status = input("Did you save the combined dataframe? ")
            if clinical_data_status.lower() == 'yes':
                if rna_data_status.lower() == 'yes':
                    if combo_data_status.lower() != 'yes':
                        name = input("Input clinical file name ").split(".")[0]
                        clinical_data = pickle.load(open(f'{name}.pickle',
                                                    'rb'))
                        name = input("Input rna file name ").split(".")[0]
                        rna_data = pickle.load(open(f'{name}.pickle', 'rb'))
                        limited_data = combine_rna_clinical(clinical_data,
                                                            rna_data)
                        run_summary = run_data_predictions(limited_data)
                    elif combo_data_status.lower() == 'yes':
                        run_summary = run_data_predictions(limited_data)
                elif rna_data_status.lower() != 'yes':
                    name = input("Input clinical file name ").split(".")[0]
                    clinical_data = pickle.load(open(f'{name}.pickle',
                                                'rb'))
                    rna_data = run_rna_formatting()
                    limited_data = combine_rna_clinical(clinical_data,
                                                        rna_data)
                    run_summary = run_data_predictions(limited_data)
            elif clinical_data_status.lower != 'yes':
                clinical_data = run_clinical_data_collection()
                rna_data = run_rna_formatting()
                limited_data = combine_rna_clinical(clinical_data, rna_data)
                run_summary = run_data_predictions(limited_data)
