from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import collections
from model_def import REG
from seq_dataloader import _get_test_data_loader
import os
import tempfile
import json

app = Flask(__name__)

# 全局模型緩存
_ec_model = None
_sa_model = None
_device = None

def load_models():
    """載入模型到全局緩存"""
    global _ec_model, _sa_model, _device
    
    if _ec_model is None or _sa_model is None:
        print("Loading models...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 載入EC模型
        ec_model_path = "./model/ec_prot_bert_finetune_reproduce.pkl"
        _ec_model = REG()
        _ec_model.load_state_dict(torch.load(ec_model_path, map_location=_device))
        _ec_model.eval()
        
        # 載入SA模型
        sa_model_path = "./model/sa_prot_bert_finetune_reproduce.pkl"
        _sa_model = REG()
        _sa_model.load_state_dict(torch.load(sa_model_path, map_location=_device))
        _sa_model.eval()
        
        print("Models loaded successfully!")
    
    return _ec_model, _sa_model, _device


@app.route("/health", methods=["GET"])
def health_check():
    """健康檢查端點"""
    return jsonify({
        "status": "healthy",
        "service": "AMP Regression EC SA Predict",
        "version": "1.0.0"
    })


def read_fasta_file(fasta_path):
    f = open(fasta_path, "r")
    seq = collections.OrderedDict()
    for line in f:
        if line.startswith(">"):
            name = line.split()[0]
            seq[name] = ""
        else:
            seq[name] += line.replace("\n", "").strip()
    f.close()
    seq_df = pd.DataFrame(seq.items(), columns=["id", "sequence"])
    seq_df["sequence_space"] = [" ".join(ele) for ele in seq_df["sequence"]]
    return seq_df


def sequences_to_dataframe(sequences):
    """將JSON序列數據轉換為DataFrame"""
    seq_data = []
    for seq in sequences:
        seq_data.append({
            "id": seq.get("id", ""),
            "sequence": seq.get("sequence", "")
        })
    
    seq_df = pd.DataFrame(seq_data)
    seq_df["sequence_space"] = [" ".join(ele) for ele in seq_df["sequence"]]
    return seq_df


def predict_from_dataframe(ec_model, sa_model, seq_df, task_id):
    """從DataFrame直接進行預測，無需文件依賴"""
    # 創建臨時CSV文件用於DataLoader
    temp_csv_path = f"result/{task_id}_temp.csv"
    os.makedirs("result", exist_ok=True)
    seq_df.to_csv(temp_csv_path, index=False)
    
    batch_size = 500
    test_loader = _get_test_data_loader(batch_size, temp_csv_path)

    ec_predict_list, sa_predict_list = [], []
    ec_model.eval()
    sa_model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch["input_ids"]
            b_input_mask = batch["attention_mask"]
            ec_predict_pMIC, _ = ec_model(b_input_ids, attention_mask=b_input_mask)
            ec_predict_list.extend(ec_predict_pMIC.data.numpy())
            sa_predict_pMIC, _ = sa_model(b_input_ids, attention_mask=b_input_mask)
            sa_predict_list.extend(sa_predict_pMIC.data.numpy())

    ec_predict_list = [item for sublist in ec_predict_list for item in sublist]
    sa_predict_list = [item for sublist in sa_predict_list for item in sublist]

    # 添加預測結果到DataFrame
    seq_df["ec_predicted_MIC_μM"] = [10 ** (-item) for item in ec_predict_list]
    seq_df["sa_predicted_MIC_μM"] = [10 ** (-item) for item in sa_predict_list]
    seq_df = seq_df.drop(columns=["sequence_space"])
    
    # 清理臨時文件
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)
    
    return seq_df


# def predict(ec_model, sa_model, fasta_path, task_id):
#     csv_path = "result/{id}.csv".format(id=task_id)
#     batch_size = 500
#     seq = read_fasta_file(fasta_path)
#     seq.to_csv(csv_path)
#     test_loader = _get_test_data_loader(batch_size, csv_path)

#     ec_predict_list, sa_predict_list = [], []
#     ec_model.eval()
#     sa_model.eval()
#     with torch.no_grad():
#         for batch in test_loader:
#             b_input_ids = batch["input_ids"]
#             b_input_mask = batch["attention_mask"]
#             ec_predict_pMIC, _ = ec_model(b_input_ids, attention_mask=b_input_mask)
#             ec_predict_list.extend(ec_predict_pMIC.data.numpy())
#             sa_predict_pMIC, _ = sa_model(b_input_ids, attention_mask=b_input_mask)
#             sa_predict_list.extend(sa_predict_pMIC.data.numpy())

#     ec_predict_list = [item for sublist in ec_predict_list for item in sublist]
#     sa_predict_list = [item for sublist in sa_predict_list for item in sublist]

#     seq["ec_predicted_pmic"] = ec_predict_list
#     seq["sa_predicted_pmic"] = sa_predict_list
#     seq.to_csv(csv_path, index=False)
#     return csv_path


def predict(ec_model, sa_model, fasta_path, task_id):
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
            b_input_ids = batch["input_ids"]
            b_input_mask = batch["attention_mask"]
            ec_predict_pMIC, _ = ec_model(b_input_ids, attention_mask=b_input_mask)
            ec_predict_list.extend(ec_predict_pMIC.data.numpy())
            sa_predict_pMIC, _ = sa_model(b_input_ids, attention_mask=b_input_mask)
            sa_predict_list.extend(sa_predict_pMIC.data.numpy())

    ec_predict_list = [item for sublist in ec_predict_list for item in sublist]
    sa_predict_list = [item for sublist in sa_predict_list for item in sublist]

    seq["ec_predicted_MIC_μM"] = [10 ** (-item) for item in ec_predict_list]
    seq["sa_predicted_MIC_μM"] = [10 ** (-item) for item in sa_predict_list]
    seq = seq.drop(columns=["sequence_space"])
    seq.to_csv(csv_path, index=False)
    return csv_path


@app.route("/predict/sequences", methods=["POST"])
def api_predict_sequences():
    """
    新的API端點：基於JSON序列數據進行預測
    完全解耦，無文件系統依賴
    """
    print("here!")
    try:
        # 解析JSON輸入
        data = request.get_json()
        if not data:
            return jsonify({
                "status": False,
                "error": "Invalid JSON data"
            }), 400
        
        # 驗證必要字段
        if 'task_id' not in data or 'sequences' not in data:
            return jsonify({
                "status": False,
                "error": "task_id and sequences are required"
            }), 400
        
        task_id = data['task_id']
        sequences = data['sequences']
        
        # 驗證sequences格式
        if not isinstance(sequences, list) or len(sequences) == 0:
            return jsonify({
                "status": False,
                "error": "sequences must be a non-empty array"
            }), 400
        
        # 驗證每個序列的格式
        for seq in sequences:
            if not isinstance(seq, dict) or 'id' not in seq or 'sequence' not in seq:
                return jsonify({
                    "status": False,
                    "error": "Each sequence must have 'id' and 'sequence' fields"
                }), 400
        
        # 載入模型
        ec_model, sa_model, device = load_models()
        
        # 轉換序列數據為DataFrame
        seq_df = sequences_to_dataframe(sequences)
        
        # 執行預測
        result_df = predict_from_dataframe(ec_model, sa_model, seq_df, task_id)
        
        # 轉換結果為JSON格式
        predictions = result_df.to_dict('records')
        
        return jsonify({
            "status": True,
            "task_id": task_id,
            "predictions": predictions,
            "total_sequences": len(predictions)
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/predict", methods=["POST"])
def api_predict():
    if request.method == "POST" and "task_id" in request.form:
        # fasta_content = request.form.get('fasta_content')
        # fasta_path = request.form.get('fasta_path')
        task_id = request.form.get("task_id")
        fasta_path = "./fasta/{id}.fasta".format(id=task_id)
        # if not fasta_path or not os.path.exists(fasta_path):
        #     return jsonify({'error': 'fasta_path is required and should be a valid path to a fasta file.'}), 400

        # Load your models here
        ec_model, sa_model, device = load_models()

        print("here!")

        csv_path = predict(ec_model, sa_model, fasta_path, task_id)

        # You might want to return a URL to the CSV file instead of the path
        return jsonify({"status": True, "csv_path": csv_path, "task_id": task_id}), 200
    else:
        return jsonify({"status": False, "error": "task_id is required."}), 400


if __name__ == "__main__":
    app.run(debug=True)
