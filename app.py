from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import collections
from model_def import REG
from seq_dataloader import _get_test_data_loader
import os

app = Flask(__name__)

def read_fasta_file(fasta_path):
    f = open(fasta_path, "r")
    seq = collections.OrderedDict()
    for line in f:
        if line.startswith(">"):
            name = line.split()[0]
            seq[name] = ''
        else:
            seq[name] += line.replace("\n", '').strip()
    f.close()
    seq_df = pd.DataFrame(seq.items(), columns=['id', 'sequence'])
    seq_df["sequence_space"] = [" ".join(ele) for ele in seq_df["sequence"]]
    return seq_df

def predict(ec_model, sa_model, fasta_path, task_id):
    # csv_path = fasta_path.replace(".fasta", ".csv")
    csv_path = "result/{id}.csv".format(id=task_id)
    batch_size = 500
    seq = read_fasta_file(fasta_path)
    seq.to_csv(csv_path)
    test_loader = _get_test_data_loader(batch_size, csv_path)
    
    ec_predict_list, sa_predict_list = [], []
    ec_model.eval()
    sa_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids']
            b_input_mask = batch['attention_mask']
            ec_predict_pMIC, _ = ec_model(b_input_ids, attention_mask=b_input_mask)
            ec_predict_list.extend(ec_predict_pMIC.data.numpy())
            sa_predict_pMIC, _ = sa_model(b_input_ids, attention_mask=b_input_mask)
            sa_predict_list.extend(sa_predict_pMIC.data.numpy())
            
    ec_predict_list = [item for sublist in ec_predict_list for item in sublist]
    sa_predict_list = [item for sublist in sa_predict_list for item in sublist]
    
    seq["ec_predicted_pmic"] = ec_predict_list
    seq["sa_predicted_pmic"] = sa_predict_list
    seq.to_csv(csv_path, index=False)
    return csv_path

@app.route('/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST' and 'task_id' in request.form:
        # fasta_content = request.form.get('fasta_content')
        # fasta_path = request.form.get('fasta_path')
        task_id = request.form.get('task_id')
        fasta_path = "./fasta/{id}.fasta".format(id=task_id)
        # if not fasta_path or not os.path.exists(fasta_path):
        #     return jsonify({'error': 'fasta_path is required and should be a valid path to a fasta file.'}), 400
        
        # Load your models here
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ec_model_path = "./model/ec_prot_bert_finetune_reproduce.pkl"
        ec_model = REG()
        ec_model.load_state_dict(torch.load(ec_model_path, map_location=device))



        sa_model_path = "./model/sa_prot_bert_finetune_reproduce.pkl"
        sa_model = REG()
        sa_model.load_state_dict(torch.load(sa_model_path, map_location=device))
        
        print("here!")

        csv_path = predict(ec_model, sa_model, fasta_path, task_id)
        
        # You might want to return a URL to the CSV file instead of the path
        return jsonify({'status': True, 'csv_path': csv_path, 'task_id': task_id}), 200
    else:
        return jsonify({'status': False, 'error': 'task_id is required.'}), 400
if __name__ == '__main__':
    app.run(debug=True)
