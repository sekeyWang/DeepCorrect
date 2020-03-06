# DeepCorrect

## Preprocess
Raw data: PXD005573 C_D160304_S251-Hela-2ug-2h_MSG_R01_T0 

The link is here: http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD005573

Raw data -> .mgf data: MSconvert(Windows software)

## GroundTruth from database search result:

Database: swiss_prot https://www.uniprot.org/downloads

Searching tool: Comet http://comet-ms.sourceforge.net/

Method: Target-decoy search strategy by filtering FDR 1%. Target-decoy method: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2922680/

Original data is Converters/feature_files/S01.mgf, containing features, peptide sequence and spectrum.

## Convert
Change old data format(features and spectrum in mgf file)to new format(features in features.csv and spectrum in spectrum.mgf) by running Converters/data_format_converter.py, so that the data can be implemented on deepnovo.

## Denovo peptide sequecing
### Novor
An open source denovo peptide sequencing based on machine learning. https://www.rapidnovor.com/download/

Run Novor on Converters/feature_files/spectrum.mgf. Result is in Converters/denovo_result/spectrum.mgf.csv

### Deepnovo2
DeepNovo2 is a denovo peptide sequencing method based on Deep Learning. 
This is a improved version of deepnovo in 2017.
https://arxiv.org/abs/1904.08514

Code is here: https://github.com/volpato30/DeepNovoV2. The model is trained with ABRF DDA spectrums (the data for Table 1 in the original paper) and default knapsack file.

Run Deepnovo on Converters/feature_files/spectrum.mgf and Converters/feature_files/features.csv. The result is at Converters/denovo_result/features.csv.deepnovo_denovo.

## Prepare for input data
Merging the result from Novor and Deepnovo by running Converters/denovo_result/merge_result. 
The merged result is at Converters/denovo_result/result.csv, which supply all features for my model. My model can trained either Novor's result or Deepnovo's result(Here Novor).

If you want to compare the accuracy of two methods, you can run analyse_result.py

For merged result, you can use divide_dataset.py to divide Converters/denovo_result/result.csv into training, validation and testing files in Converters/training_features.

## Load data
DeepCorrect.py is the main file. Config.py is configuration file.
For loading data, it first uses data_reader.DenovoDataset(feature_filename, spectrum_filename) to read the features and spectrum from files.
Then it uses construct_model_input to extract the model features from DenovoDataset. 
The extracted file should be a size 68*L matrix, where 68 is number of features and L is the length of denovo sequence. It can be the input of my CNN model.

## Train and validate model
Training is at train_model.py. It contains training, testing and other analysis functions. 
After training, the model will be saved at model folder. Currently good model is model3-3(The accuracy should be around 86%). Other model may not be useful since I changed some feature. 
You can change the configurations and train new models. It will automatically use the GPU.

## Local Search
After training a good model to give scores for a sequence, my ultimate goal is to modify the wrong sequence and to increase the accuracy.
The code modify_peptide.py is to modify the sequence, some of the functions are at LocalSearch.py.

I tried to modify the sequences in Converters/training_features/test.csv, but the result is not very good. 
