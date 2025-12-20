# Analisi degli Embedding ATC tramite Graph Neural Networks (GAT)

Questo repository contiene il codice e il report per apprendere **embedding densi (128D)** dei codici **ATC** sfruttando direttamente la **struttura gerarchica parent-child** come grafo, tramite **Graph Attention Networks (GAT)** in **PyTorch Geometric**.

## Contenuti principali
- **Pre-processing ATC**: pulizia/normalizzazione dei codici, rimozione duplicati, gestione NaN, creazione mapping `code -> node_idx`.
- **Costruzione del grafo**: archi **child → parent** + archi inversi per message passing bidirezionale.
- **Feature dei nodi**: livello (one-hot) + feature topologiche (in-degree, out-degree, profondità normalizzata) → vettore **8D**.
- **Modello GNN**: encoder **GAT a 4 layer** + 2 teste di classificazione:
  - macro-categoria anatomica (prima lettera ATC, ~14 classi)
  - livello gerarchico (5 classi)
- **Training multi-task** con loss totale:
  - CrossEntropy categoria
  - CrossEntropy livello (pesata)
  - **hierarchy pull loss** per avvicinare embedding parent-child (pesata)
- **Export risultati**: salvataggio embedding e metadati in `atc_embeddings_trained.pt`.

## Valutazione e visualizzazioni
- **Preservazione gerarchica**: confronto distanze child-parent vs random-random.
- **Coerenza locale**: purezza **k-NN** per macro-categoria.
- **Metriche di clustering**: Silhouette, Davies–Bouldin, Calinski–Harabasz.
- **Plot**:
  - t-SNE 2D (categoria / livello)
  - PCA 2D globale (categoria / livello, con filtro opzionale per livello massimo)
  - heatmap distanze tra centroidi delle categorie (t-SNE e PCA)
  - statistiche dataset (conteggi per livello/categoria) e sottografi per categorie selezionate.
