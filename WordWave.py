# Importar bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Passo 1: Ler o arquivo e processar os dados
termos = []
contagens = []

with open('termos.txt', 'r', encoding='utf-8') as file:
    for linha in file:
        partes = linha.strip().split()
        if len(partes) < 2:
            continue  # Ignora linhas inválidas
        # Combina os termos (podem ter espaços) e separa a contagem
        try:
            contagem = int(partes[-1])  # Último elemento é a contagem
            termo = ' '.join(partes[:-1])  # Termo é o restante
            termos.append(termo)
            contagens.append(contagem)
        except ValueError:
            continue

# Passo 2: Análise estatística básica
if not termos:
    print("Nenhum dado válido encontrado.")
    exit()

# Criar DataFrame
df = pd.DataFrame({'Termo': termos, 'Contagem': contagens})

# Estatísticas descritivas
print("\nEstatísticas das Contagens:")
print(f"Média: {df['Contagem'].mean():.2f}")
print(f"Desvio Padrão: {df['Contagem'].std():.2f}")
print(f"Máximo: {df['Contagem'].max()}")
print(f"Mínimo: {df['Contagem'].min()}\n")

# Passo 3: Análise de correlação de termos (co-ocorrência)
# Dividir termos individuais (ex: "Mimosa AND Tenuiflora" → ["Mimosa", "Tenuiflora"])
termos_split = [termo.replace(' AND ', ' ').split() for termo in df['Termo']]

# Coletar todos os termos únicos
todos_termos = list(set([termo for sublist in termos_split for termo in sublist]))

# Criar matriz de co-ocorrência
matriz_coocorrencia = pd.DataFrame(0, index=todos_termos, columns=todos_termos)

for i, termos_combinacao in enumerate(termos_split):
    contagem = df['Contagem'].iloc[i]
    for termo1 in termos_combinacao:
        for termo2 in termos_combinacao:
            if termo1 != termo2:
                matriz_coocorrencia.loc[termo1, termo2] += contagem

# Calcular correlações (similaridade de Jaccard)
def jaccard_similarity(termo1, termo2):
    total = matriz_coocorrencia.loc[termo1].sum() + matriz_coocorrencia.loc[termo2].sum() - matriz_coocorrencia.loc[termo1, termo2]
    if total == 0:
        return 0
    return matriz_coocorrencia.loc[termo1, termo2] / total

# Exemplo: Correlação média entre todos os pares
pares = [(t1, t2) for t1 in todos_termos for t2 in todos_termos if t1 < t2]
correlacoes = [jaccard_similarity(t1, t2) for t1, t2 in pares]
print(f"Correlação Média (Jaccard): {np.mean(correlacoes):.2f}\n")

# Passo 4: Nuvem de palavras
# Calcular frequência total de cada termo
frequencias = {}
for termo in todos_termos:
    frequencias[termo] = matriz_coocorrencia.loc[termo].sum()

# Gerar nuvem de palavras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencias)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nuvem de Termos Mais Frequentes", fontsize=16, pad=20)
plt.savefig('nuvem_termos.png', dpi=300, bbox_inches='tight')
plt.show()