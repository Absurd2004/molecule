import pandas as pd 

# dataset_all = pd.read_csv("./fragment_utils/transformer_train_data_final.csv",header=None).values.flatten().tolist()
# dataset_error = pd.read_csv("./fragment_utils/error_smiles_of_transformer_train_data_final.csv",header=None).values.flatten().tolist()
# dataset = list(set(dataset_all)-set(dataset_error))
# print(len(dataset))
# pd_dataset = pd.DataFrame(dataset)
# pd_dataset.to_csv("./fragment_utils/dataset.csv",index=None,header=None)


# dataset_all = pd.read_csv("./fragment_utils/train_data_by_transformer.csv",header=None).values.flatten().tolist()
# dataset_error = pd.read_csv("./fragment_utils/error_smiles_of_train_data_by_transformer.csv",header=None).values.flatten().tolist()
# dataset = list(set(dataset_all)-set(dataset_error))
# print(len(dataset))
# pd_dataset = pd.DataFrame(dataset)
# pd_dataset.to_csv("./fragment_utils/dataset_gen.csv",index=None,header=None)

Voc1 = pd.read_csv("./fragment_utils/fragments_Voc.csv",header=None).values.flatten().tolist()

Voc2 = pd.read_csv("./fragment_utils/fragments_Voc2.csv",header=None).values.flatten().tolist()

print(len(Voc1))
print(len(Voc2))

dataset1 = list(set(Voc1)-set(Voc2))
print(len(dataset1))


dataset2 = list(set(Voc2)-set(Voc1))
print(len(dataset2))