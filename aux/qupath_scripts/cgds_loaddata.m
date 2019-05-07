clear
clc

%% Get genetic sdata
addpath /Applications/MATLAB_R2017a.app/toolbox/cgds
cgdsURL = 'http://www.cbioportal.org/public-portal';
genes_of_interest = {'PTEN'};

cancer_studies = getcancerstudies(cgdsURL);
tcga_idx = 179;
prostate_cancer = cancer_studies.name{tcga_idx,1};
display(prostate_cancer)
% See also 180: (TCGA, Provisional) and 177: (MSKCC/DFCI, Nature Genetics 2018)
study_id = cancer_studies.cancerTypeId{strcmp(cancer_studies.name, prostate_cancer)};
case_list = getcaselists(cgdsURL, study_id);
genetic_profiles = getgeneticprofiles(cgdsURL, study_id);
profile_data = getprofiledata(cgdsURL, case_list.caseListId{1}, ...
                genetic_profiles.geneticProfileId(1:2), ...
                genes_of_interest, true);    
gene_patients_ids = sort(profile_data.caseId);


%% Check matching histology slide data
samples_file = "/Users/andreachatrian/Desktop/tcga_data_info/biospecimen.project-TCGA-PRAD.2018-10-05/sample.tsv";
samples_data = tdfread(samples_file);
slide_patients_ids = sort(cellstr(samples_data.sample_submitter_id));
remove_final_el = @(A) A(1:(end-1));
slide_patients_ids = cellfun(remove_final_el, slide_patients_ids, 'UniformOutput', false);
sample_and_gene_ids = intersect(gene_patients_ids, slide_patients_ids);
overlap = length(sample_and_gene_ids) / length(union(gene_patients_ids, sample_and_gene_ids));
display("Overlap = " + num2str(overlap))



