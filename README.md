# 🔮 NFT Classifier

Este projeto implementa um **classificador de coleções de NFTs** usando *embeddings de linguagem* e um modelo de machine learning treinado previamente no Google Colab.  
A aplicação final é servida em um **servidor Flask** com interface web simples.

---

## 🚀 Pipeline do Projeto

### 1. Treinamento no Google Colab
No Colab foi feito o **treinamento e avaliação de vários classificadores** a partir das descrições das coleções de NFT.  

1. **Preparação dos dados**
   - Cada NFT possuía *badges* e descrições textuais.
   - Os textos foram processados (remoção de stopwords, lematização etc.).
   - As classes alvo são **7 categorias**:  
     ```
     ['art', 'gaming', 'memberships', 'music', 'pfps', 'photography', 'virtual-worlds']
     ```

2. **Geração dos embeddings**
   - Foi utilizado o modelo de linguagem **`paraphrase-multilingual-MiniLM-L12-v2`** (da biblioteca `sentence-transformers`).
   - Esse modelo converte cada descrição em um vetor de **384 dimensões**.
   - Esses vetores numéricos representam o "significado" do texto em um espaço vetorial.

3. **Treinamento dos classificadores**
   - Vários modelos foram testados (SVM, Logistic Regression, Random Forest, etc.).
   - O **Random Forest** obteve a **melhor acurácia (~75%)** nos dados de validação.
   - Para manter consistência, os embeddings foram **normalizados** com `StandardScaler`.

4. **Exportação dos artefatos**
   - O modelo final (`RandomForest`), o scaler, o label encoder e as informações de configuração foram salvos em arquivos:
     ```
     model.joblib
     scaler.joblib
     label_encoder.joblib
     meta.json
     ```

---

### 2. Servidor Flask

A aplicação Flask carrega os artefatos salvos e expõe um endpoint para classificação.

1. **Carregamento dos artefatos**
   - O `RandomForest`, `StandardScaler` e `LabelEncoder` são carregados com `joblib`.
   - O `meta.json` indica qual modelo de embedding deve ser utilizado (garante compatibilidade).

2. **Fluxo de predição**
   - O usuário envia uma descrição de NFT (`descricao`).
   - Essa descrição é convertida em embedding com o mesmo modelo usado no treino.
   - O vetor é transformado pelo `StandardScaler` (normalização).
   - O `RandomForest` gera probabilidades (`predict_proba`) para cada uma das 7 classes.
   - As classes são ordenadas da maior para a menor probabilidade.
   - O resultado retorna:
     - A **classe mais provável** (`predicao`).
     - As **Top K classes mais próximas** com suas probabilidades (`top_k`).

3. **Endpoint principal**
   - `GET /` → retorna a página web (`index.html`).
   - `POST /predict` → recebe `{ descricao, top_k }` e retorna JSON com predição.

---

### 3. Interface Web

A página web foi feita com **Bootstrap** para melhor visualização:

- Campo de texto para inserir a descrição da coleção.
- Campo numérico para escolher o `Top K`.
- Resultado principal destacado com **badge verde**.
- Lista das Top K classes com probabilidades exibidas em forma de **lista estilizada**.

Exemplo de saída:

---

## 📊 Como as métricas de aproximação são calculadas

O classificador usa o método `predict_proba` do `RandomForest`, que funciona assim:

1. Cada árvore de decisão do Random Forest "vota" em uma classe.
2. A probabilidade final é a proporção de votos de todas as árvores.
   - Exemplo: em 100 árvores, se 78 votarem em "pfps" → probabilidade = 0.78 (78%).
3. O sistema retorna as probabilidades normalizadas (que somam 1.0).
4. O `np.argsort(probs)[::-1]` organiza da maior para a menor probabilidade.
5. O campo `proba_pct` é a probabilidade multiplicada por 100 e arredondada.

Ou seja:
- **Classe predita** = a de maior probabilidade.
- **Top K** = as K classes mais prováveis (ordenadas).

---

## ▶️ Como rodar localmente

1. Clone este repositório e entre na pasta:
   ```bash
   git clone <repo>
   cd nft-classifier
2. pip install -r requirements.txt

3. Coloque os arquivos do modelo dentro da pasta data/:

4. python app.py
