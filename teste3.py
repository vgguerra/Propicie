import pandas as pd



# Criar tabela

'''
colunas = ["Distância real", "Distância calculada", "Erro"]

df = pd.DataFrame(columns=colunas)

df.to_excel("dados.xlsx", index=False, engine="openpyxl")

print("Arquivo Excel criado com sucesso!")'''





# Ler da tabela
"""
arquivo = "dados.xlsx"
df = pd.read_excel(arquivo)

coluna = df["Erro"].values
tamanho = df["Erro"].size + 1
soma = 0

for col in coluna:
    soma += col 


print(f"O erro aproximado é de {soma/tamanho:.3f}")"""

teste = [12,45]

opa = max(teste)

print(opa)


