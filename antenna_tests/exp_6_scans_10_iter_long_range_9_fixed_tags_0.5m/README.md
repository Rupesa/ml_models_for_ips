Descrição de teste

O objetivo deste teste é fixar a posição de várias tags ao longo de um corredor (metro a metro) e fazer variar a posição de outra tag paralela equidistante à antena.
As posições da tag dinâmica variara dos 0.5m aos 10m com 10 iterações e potência 280, 290 e 300 mW.

tags fixas:
1m - e28068940000400386621d4f
2m - e2806894000050038661fd4f
3m - e2806894000050038662154f
4m - e2806894000050038662194f
5m - e2806894000040038662114f
6m - e2806894000040038662054f
7m - e2806894000040038661f94f
8m - e2806894000040038662214f
9m - e2806894000050038662014f

tag dinâmica:
- e28068940000500386620d4f


Análise:

Erros:
Treinei e testei os modelos com vários tipos de features:
- Usando apenas o número de ativações, o menor erro foi de 1.05m
- Usando o tempo médio entre ativações, o menor erro foi de 0.75m
- Usando a mediana do tempo entre ativações, o menor erro foi de 0.98m
- Usando o número de ativações e o tempo médio entre ativações, o menor erro foi de 0.84m
- Usando o número de ativações, o tempo médio entre ativações e a diferença do numero de ativações relativamente a cada tag fixa, o menor erro foi de 0.72m

Pré-processamento:
Foram utilizada diversas técnicas de pré-processamento para a remoção dos outliers relativos aos tempos entre ativações antes de fazer a média, resultando nos seguintes erros:
- No Outlier: 0.75m
- Isolation Forest: 0.92m
- Minimum Covariance Determinant (Elliptic Envelope) with contamination = 0.05: 0.88m
- Minimum Covariance Determinant (Elliptic Envelope) with contamination = 0.01: 0.88m
- Minimum Covariance Determinant (Elliptic Envelope) with contamination = 0.1: 0.91m
- Minimum Covariance Determinant (Elliptic Envelope) with contamination = 0.4: 1.01m
- Local Outlier Factor: 0.80m
- One-Class SVM: 1.68m
- Z-score with mean: 0.88m
- Z-score with median: 0.98m
- Interquartile range method with mean: 0.84m
- Interquartile range method with median: 1.04m

Foi também variado o cálculo das diferenças entre timestamps para gaps variáveis:


References:
https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
https://www.statology.org/remove-outliers-python/