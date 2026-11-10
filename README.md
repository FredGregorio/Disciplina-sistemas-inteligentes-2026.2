# Sistemas Inteligentes - 2026.2

Repositório com os projetos da disciplina **PAM0466 - Sistemas Inteligentes** (UFERSA - Centro Multidisciplinar
de Pau dos Ferros), semestre **2026.2**.

**Discente:** Frederico Gregório Emidio Santos Ferreira
**Docente:** Pedro Thiago Valério de Souza

## Projetos - Unidade 1

Cada projeto tem um notebook com o código executado e um relatório (`relatorio.md`) com as análises pedidas.

| # | Problema | Técnica | Dataset | Notebook | Relatório | Colab |
|---|---|---|---|---|---|---|
| 1 | Previsão de produção de energia em usina de ciclo combinado | Regressão linear múltipla | UCI CCPP | [problema1.ipynb](Primeira%20Unidade/Projetos/01-regressao-linear-ccpp/src/problema1.ipynb) | [relatório](Primeira%20Unidade/Projetos/01-regressao-linear-ccpp/relatorio.md) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredGregorio/Disciplina-sistemas-inteligentes-2026.2/blob/main/Primeira%20Unidade/Projetos/01-regressao-linear-ccpp/src/problema1.ipynb) |
| 2 | Previsão de doenças cardíacas | Regressão logística | UCI Heart Disease (Cleveland) | [problema2.ipynb](Primeira%20Unidade/Projetos/02-regressao-logistica-heart-disease/src/problema2.ipynb) | [relatório](Primeira%20Unidade/Projetos/02-regressao-logistica-heart-disease/relatorio.md) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredGregorio/Disciplina-sistemas-inteligentes-2026.2/blob/main/Primeira%20Unidade/Projetos/02-regressao-logistica-heart-disease/src/problema2.ipynb) |
| 3 | Classificação da qualidade de vinhos | k-NN | UCI Wine Quality | [problema3.ipynb](Primeira%20Unidade/Projetos/03-knn-wine-quality/src/problema3.ipynb) | [relatório](Primeira%20Unidade/Projetos/03-knn-wine-quality/relatorio.md) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredGregorio/Disciplina-sistemas-inteligentes-2026.2/blob/main/Primeira%20Unidade/Projetos/03-knn-wine-quality/src/problema3.ipynb) |
| 4 | Classificação de risco de crédito | Árvore de decisão | UCI Statlog German Credit | em andamento | | |
| 5 | Segmentação de clientes | k-Means | UCI Wholesale Customers | em andamento | | |

Enunciados: [Lista de Projetos](Primeira%20Unidade/Listas/Lista%20de%20Projetos.pdf) ·
[Lista de Exercícios](Primeira%20Unidade/Listas/Lista%20de%20Exercicios.pdf)

## Estrutura

```
Primeira Unidade/
├── Listas/                      enunciados (PDF)
└── Projetos/
    └── 0N-<nome-do-projeto>/
        ├── data/                dataset usado
        ├── src/                 notebook (.ipynb)
        ├── resultados/          gráficos gerados pelo notebook
        └── relatorio.md         análises e respostas
```

## Como executar

### Opção 1 - Google Colab (sem instalar nada)

Clique no botão **Abrir no Colab** do projeto e execute todas as células (*Ambiente de execução → Executar tudo*).
O notebook baixa o dataset diretamente deste repositório.

### Opção 2 - No próprio computador

Requer Python 3.11 ou superior.

```bash
git clone https://github.com/FredGregorio/Disciplina-sistemas-inteligentes-2026.2.git
cd Disciplina-sistemas-inteligentes-2026.2
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install notebook
jupyter notebook
```

Depois, abra o notebook do projeto em `Primeira Unidade/Projetos/.../src/` e execute as células. Rodando no
computador, o notebook usa o dataset da pasta `data/` e salva os gráficos em `resultados/`.

## Datasets

Todos os datasets vêm do [UCI Machine Learning Repository](https://archive.ics.uci.edu/), conforme indicado no
enunciado de cada problema.
