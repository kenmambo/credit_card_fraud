import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path='data', unzip=True)

# kaggle.api.dataset_metadata('mlg-ulb/creditcardfraud', path='data')

print("Dataset downloaded to data/creditcard.csv")