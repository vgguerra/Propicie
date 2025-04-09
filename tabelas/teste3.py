import os
import pandas as pd

# Criar tabela

os.makedirs("./tabelas", exist_ok=True)

colunas = ["Idade","Altura","Peso","Gênero","Distância real", "Distância calculada"]
df = pd.DataFrame(columns=colunas)

df.to_excel("./tabelas/dados.xlsx", index=False, engine="openpyxl")

print("Arquivo Excel criado com sucesso!")




                        # caminho_arquivo = "./tabelas/dados.xlsx"
                        # df = pd.read_excel(caminho_arquivo, engine="openpyxl")
                        
                        # real = input("Qual a distância real: ")
                        # erro = np.abs(float(real) - float(final_distance))
                        # nova_linha = {
                        #             "Distância real": real,
                        #             "Distância calculada": final_distance,
                        #             "Erro": erro
                        #         }
                        # df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
                        # df.to_excel(caminho_arquivo, index=False, engine="openpyxl")


# Ler da tabela

"""arquivo = "./tabelas/dados.xlsx"
df = pd.read_excel(arquivo)

coluna = df["Erro"].values
tamanho = df["Erro"].size + 1
soma = 0

for col in coluna:
    soma += col 


print(f"O erro aproximado é de {soma/tamanho:.3f}")"""


