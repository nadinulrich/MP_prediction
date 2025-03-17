from datetime import datetime
import math
import os
import deepchem as dc
import numpy as np 
import csv
import shutil
from deepchem.feat.mol_graphs import ConvMol
import pandas as pd
import sklearn
from mygraphconvmodel import MyGraphConvModel
from deepchem.models import KerasModel

#functions
def datasets(Extdataset_csv,ExtPredicts_csv, result_dir):

    time_now = datetime.now()                                             # aktuelle Zeit bestimmen

    extdatefile = os.path.basename(Extdataset_csv)                          #dateinamen extrahieren
    expredictfile = os.path.basename(ExtPredicts_csv)                       #dateinamen extrahieren 

    dateiname = extdatefile.split(".")[0]+"-"+expredictfile.split(".")[0]+"_"+time_now.strftime("%d%m%Y%H%M%S")   # Ergebnisdateinamen zusammensetzen

    #------------Excel Datei aus CSV/Dataframes erzeugen---------------------------#

    df1      = pd.DataFrame
    df2      = pd.DataFrame
    ergebnis = pd.DataFrame

    df1 = pd.read_csv(Extdataset_csv, encoding='utf-8', header=None)
    df2 = pd.read_csv(ExtPredicts_csv, encoding='utf-8', header=None)
    ergebnis = pd.concat([df1, df2], axis=1)

    try:
      ergebnis.to_excel(result_dir+dateiname+'.xlsx', index = False, header = False)
         # Excel schreiben
    except:
        print("ERROR: Excel konnte nicht geschrieben werden")

    #------------Magic lineare Regression Stuff-------------------------------------#

    extdataset_liste = df1.iloc[:,1].tolist()              # Colum 1 (Zähler beginnt bei 0) aus DF1 in eigene Liste laden
    extpredicts_liste =df2.iloc[:,0].tolist()              # Colum 0 (Zähler beginnt bei 0) aus DF2 in eigene Liste laden

    if len(extdataset_liste) == len(extpredicts_liste):                        
              
        mse = sklearn.metrics.mean_squared_error(extdataset_liste,extpredicts_liste)   
        rmse = math.sqrt(mse)                                                          
        r2 = sklearn.metrics.r2_score(extdataset_liste,extpredicts_liste)              

        with open(result_dir+dateiname+'_r2_wert.txt','w') as f:                             # Magic Stuff in Datei schreiben
            f.write('mse: '+str(mse)+"\n")
            f.write('rmse: '+str(rmse)+"\n")
            f.write('r2: '+str(r2)+"\n")
    return rmse

    #------------Seaborn Grafik Zeuchs-------------------------------------------------------#

def read_data(input_file_path, output_file_path):
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=prediction_tasks, feature_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(input_file_path, shard_size=8192)
    
    trainarray=[]
    smiles = dataset.ids
    values = dataset.y    
    
    for i in range(dataset.ids.size):
      entry=[str(smiles[i]),str(values[i][0])]
      trainarray.append(entry)

    with open(output_dir+output_file_path,mode='w') as csv_file:
        wr=csv.writer(csv_file,quoting=csv.QUOTE_NONE)
        wr.writerows(trainarray)

    return dataset

def make_predictions(model,dataset,batch_size, output_dir, result_dir, read_out_string ):
  predictions = model.predict_on_generator(data_generator(dataset,batch_size=batch_size))
  predictions = reshape_y_pred(dataset.y, predictions)
  write_predictions(predictions, 'Auslesen'+read_out_string+'Prediction.csv')
  rmse = datasets(output_dir+'Auslesen'+read_out_string+'.csv',output_dir+'Auslesen'+read_out_string+'Prediction.csv',result_dir)
  return rmse
 
def data_generator(dataset,batch_size,epochs=1,modelX = None):
  best_rmse = float("inf")
  for epoch in range(epochs):
    epoch_num = epoch+1
    if modelX is not None:
      print("start epoch "+ str(epoch+1))
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size,deterministic=True, pad_batches=True)):
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
      labels = [y_b]
      weights = [w_b]
      yield (inputs, labels, weights)
    if modelX is not None:
      print("#####Making step predictions.")
      modelX.model_dir = stepmodell_directory
      modelX.fit_generator(data_generator(train_dataset,batch_size,epochs=1))
      modelX.save_checkpoint()
      new_checkpoint_path = stepmodell_directory+'epoch'+str(epoch_num)
      isExist = os.path.exists(new_checkpoint_path)
      if not isExist:
        os.makedirs(new_checkpoint_path)   

      files_in_directory = os.listdir(stepmodell_directory)
      for file in files_in_directory:
        if file.startswith('c'):
          file_path = os.path.join(modelX.model_dir, file)
          new_file_path = os.path.join(new_checkpoint_path, file)
          shutil.move(file_path, new_file_path)
      #print(rmse_train)      
      #print("#####Making ModelX Valset predictions.")
      rmse_val = make_predictions(model=modelX,dataset= test_dataset,batch_size=batch_size, output_dir=output_dir, result_dir=result_dir, read_out_string='Valset')
      print(rmse_val)
      print("end epoch "+ str(epoch_num))

      #implement early stopping
      if rmse_val < best_rmse:
        best_rmse = rmse_val
        patience_counter = 0
      else:
        patience_counter += 1
      if patience_counter == patience:
        print("Early stopping triggered. Training stopped.")
        break

        
 

def reshape_y_pred(y_true, y_pred):
 
  n_samples = len(y_true)
  retval = np.vstack(y_pred)
  return retval[:n_samples]

def write_predictions(dataset, file_name):
  with open(output_dir+file_name,mode='w') as csv_file:
    wr=csv.writer(csv_file,quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)

#Constants
prediction_tasks = ['MP']
basis='./'
model_dir = basis+'CheckPoints/'
output_dir = basis+'Auslesen/'
result_dir = basis+'Results/'
stepmodell_directory = basis+'StepModelle/'
stop_epoch =  len(os.listdir(stepmodell_directory))
train_dataset = read_data('INPUTDATA', 'OUTPUTDATA')
test_dataset = read_data('INPUTDATA_TEST', 'OUTPUTDATA_TEST')
ext_dataset = read_data('INPUTDATA_EXT', 'OUTPUTDATA_EXT')

#model parameters
batch_size = 50
num_epochs = 2000
neuronslayer1=64
neuronslayer2=128
dropout=0.1
learning_rate=0.0005
patience = 20

#training model
train= MyGraphConvModel(batch_size=batch_size,neuronslayer1=neuronslayer1,neuronslayer2=neuronslayer2,dropout=dropout)
keras_model = KerasModel(train, loss=dc.models.losses.L1Loss() ,learning_rate=learning_rate,  model_dir=model_dir)
print('---start training ---')
#if no step predictions needed, modelX => None
loss = keras_model.fit_generator(data_generator(train_dataset,batch_size=batch_size,epochs=num_epochs,modelX=keras_model)) 
print('TRAINED MODEL Loss: %f' % loss)
keras_model.save_checkpoint()
print('SAVED MODEL')   

#loading model
#count epochs
num_epoch = 1
stepmodell_directory = os.listdir(basis+"StepModelle/")
for file in stepmodell_directory:
  if file.startswith('epoch'):
    num_epoch+=1
print("num epoch "+ str(num_epoch))
best_epoch = num_epoch-patience-1
print("best_epoch "+ str(best_epoch))
restore_dir = stepmodell_directory+'epoch'+str(best_epoch)
print("restore_dir"+ str(restore_dir))
model2 = KerasModel(MyGraphConvModel(batch_size=batch_size,neuronslayer1=neuronslayer1,neuronslayer2=neuronslayer2,dropout=dropout), loss=dc.models.losses.L1Loss(), model_dir=restore_dir)
model2.restore()

#test predictions
#print('---start test predictions---')
make_predictions(model=model2,dataset= test_dataset,batch_size=batch_size, output_dir=output_dir, result_dir=result_dir, read_out_string='Valset')
#ext predictions
#print('---start ext predictions---')
make_predictions(model=model2,dataset= ext_dataset,batch_size=batch_size, output_dir=output_dir, result_dir=result_dir, read_out_string='Extdataset')
print('---END!---')
