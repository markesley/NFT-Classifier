# app.py
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, 'data')

MODEL_PATH = os.path.join(DATA_DIR, 'model.joblib')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.joblib')
LE_PATH = os.path.join(DATA_DIR, 'label_encoder.joblib')
META_PATH = os.path.join(DATA_DIR, 'meta.json')

# checagem de arquivos
for p in [MODEL_PATH, SCALER_PATH, LE_PATH, META_PATH]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Arquivo não encontrado: {p} — coloque os artefatos em ./data/")

# carrega artefatos
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)
with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)

EMBED_MODEL_NAME = meta.get('embedding_model_name', None)
if EMBED_MODEL_NAME is None:
    raise ValueError("meta.json não contém 'embedding_model_name' — verifique meta salvo no Colab")

print(f"[app] usando embedding model: {EMBED_MODEL_NAME}")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# tentar inferir dimensão esperada
expected_dim = None
if hasattr(scaler, 'mean_'):
    expected_dim = scaler.mean_.shape[0]
elif hasattr(model, 'n_features_in_'):
    expected_dim = model.n_features_in_
print(f"[app] dimensão esperada (scaler/model): {expected_dim}")

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    descricao = payload.get('descricao', '')
    top_k = int(payload.get('top_k', 3))
    if not descricao or not descricao.strip():
        return jsonify({'error': 'Descrição vazia'}), 400

    # gera embedding
    emb = embed_model.encode([descricao])[0]
    emb = np.asarray(emb, dtype=float).reshape(1, -1)

    # checagem de dimensão
    emb_dim = emb.shape[1]
    if expected_dim is not None and emb_dim != expected_dim:
        return jsonify({
            'error': 'Dimensão do embedding incompatível.',
            'details': {
                'embedding_dim': int(emb_dim),
                'expected_dim': int(expected_dim),
                'message': 'Gere embeddings com o mesmo modelo usado no treino (veja meta.json).'
            }
        }), 500

    # transforma e prediz
    try:
        Xs = scaler.transform(emb)
        probs = model.predict_proba(Xs)[0]
    except Exception as e:
        return jsonify({'error': 'Falha interna ao preparar/predizer', 'exc': str(e)}), 500

    idxs = np.argsort(probs)[::-1][:top_k]
    results = []
    for i in idxs:
        results.append({
            'colecao': str(le.inverse_transform([i])[0]),
            'proba': float(probs[i]),
            'proba_pct': round(float(probs[i]) * 100, 2)
        })

    return jsonify({'predicao': results[0], 'top_k': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
