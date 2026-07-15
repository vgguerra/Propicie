# Propicie: Avaliação Automatizada da Aptidão Física

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.8+-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema para avaliação automatizada de testes de aptidão física da *Bateria de Fullerton*, projetado para apoiar o envelhecimento ativo, auxiliando profissionais da área da saúde e de cuidados com idosos. Este projeto utiliza visão computacional com um sensor Kinect V2 e a biblioteca MediaPipe Holistic do Google para fornecer medições precisas e em tempo real.

## 📋 Sobre o Projeto

Este projeto, desenvolvido como parte da colaboração **PROPICIE - IPBEJA & IFSC** e contribuindo para a iniciativa **CAPACITA**, continuado por meio das **Bolsas TRUST** visa automatizar a avaliação da aptidão física de idosos. Ao automatizar os testes da Bateria de Testes Funcionais de Fullerton, podemos coletar dados objetivos sobre flexibilidade e força, o que é crucial para monitorar o declínio físico e promover programas de envelhecimento ativo personalizados.

O sistema foca em duas avaliações principais:
* **Teste de Sentar e Alcançar** (`Sentado e alcançar os pés com as mãos`): Mede a flexibilidade dos membros inferiores.
* **Teste de Alcançar atrás das Costas** (`Alcançar as mãos atrás das costas`): Mede a flexibilidade dos membros superiores (ombros).

O núcleo do projeto é uma aplicação em Python que utiliza um sensor Kinect V2 para capturar os movimentos do usuário e o framework MediaPipe Holistic para realizar a detecção de marcos corporais em tempo real. Essa abordagem permite o cálculo preciso dos ângulos corporais para validação da postura e das distâncias-chave para a pontuação dos testes.

Posteriormente foi utilizado a camera Orbbec Femto Mega, para comparar os resultados entre as duas cameras.

### Principais Descobertas
A pesquisa conduzida por Artem Bukhantsev e aprofundada por <a href="https://github.com/vgguerra"> Victor Guerra <a/> e cotinuada por <a href="https://github.com/JuliaKoene"> Julia Koene </a> concluiu que:
* A implementação com **MediaPipe** demonstrou uma precisão superior para o teste de Sentar e Alcançar, com um Erro Médio Absoluto (MAE) de aproximadamente **2.25 cm**.
* Esta abordagem foi significativamente mais precisa do que uma implementação nativa com PyKinect, que apresentou um MAE de 8.65 cm, devido a desafios como a instabilidade do esqueleto virtual ("jittering").
* A camera Orbbec pode se provar mais precisa que o Kinect por sua maior qualidade.

---

## ✨ Funcionalidades

* **Avaliação em Tempo Real**: Análise automatizada dos exercícios Sentar e Alcançar e Alcançar atrás das Costas.
* **Rastreamento de Alta Precisão**: Utiliza o MediaPipe Holistic para um rastreamento robusto e em tempo real de 33 marcos de pose, além de marcos detalhados das mãos.
* **Validação de Postura**: Calcula ângulos articulares (joelho, quadril, cotovelo) para garantir que o usuário esteja executando o exercício corretamente antes de realizar a medição.
* **Cadastro de Usuário**: Uma interface simples para registrar os dados do participante (idade, altura, peso, gênero) antes de iniciar os testes.
* **Registro de Dados**: Salva automaticamente os resultados dos testes, incluindo a distância calculada, a distância real (para validação) e o erro de medição, em arquivos Excel (`.xlsx`) para análise posterior.
* **Análise Estatística**: Inclui scripts em Python para analisar os dados coletados e calcular estatísticas-chave sobre o erro de medição.
* **Feedback em Tempo Real**: Fornece visualizações na tela do esqueleto, métricas-chave e instruções para guiar o usuário.
* **Visualização dos Dados**: Utilizando o Matplotlib foram criados gráficos para visualização dos testes realizados.
* **Interface Clara**: Adicionadas posteriormente, foram adicionadas mais instruções, de forma que alguém que desconhece o sistema o usuaria sem grandes dificuldades.

---

## 🛠️ Como Funciona

O sistema segue um fluxo de trabalho claro para cada avaliação:
1.  **Tela de Menu**: Seleção do exercício desejado ou visualização dos dados, bem como a seleção da linguagem, português ou inglês.
2.  **Cadastro do Usuário**: O usuário insere seus dados demográficos.
3.  **Captura de Vídeo**: Um Kinect V2 captura o feed de vídeo do usuário.
4.  **Detecção de Marcos**: O vídeo é processado quadro a quadro. O MediaPipe Holistic detecta os marcos do corpo, mãos e face do usuário.
5.  **Calibração e Verificação da Postura**:
    * Para o teste de **Sentar e Alcançar**, o sistema valida a postura verificando se os ângulos do joelho, quadril e cotovelo estão dentro de limites pré-definidos (por exemplo, o joelho deve estar estendido). Uma vez que o usuário mantém uma pose de calibração válida, a posição do pé é fixada como referência.
    * Para o teste de **Alcançar atrás das Costas**, o sistema aguarda o usuário manter uma pose estável com as mãos atrás das costas.
6.  **Medição da Distância**: A distância euclidiana entre os marcos-chave (por exemplo, pontas dos dedos até a posição calibrada do pé, ou pontas dos dedos de uma mão para a outra) é calculada em pixels e convertida para centímetros. Um fator de correção de erro, derivado de testes empíricos, é aplicado para aumentar a precisão.
7.  **Exibição e Registro dos Resultados**: A distância final calculada é exibida na tela, junto com a distância real medida e o cálculo do erro, e os resultados completos são salvos em um arquivo de log e em uma planilha Excel para o grupo de usuários.

---

## 🚀 Tecnologias Utilizadas

* **Linguagem**: **Python 3.8+**
* **Visão Computacional**: **OpenCV**, **MediaPipe Holistic**
* **Hardware**: **Microsoft Kinect for Windows v2**, **Orbbec Femto Mega**
* **Wrappers**: **SDK do Kinect: PyKinect2**, **SDK Orbbec: OrbbecSDK_v2**
* **Manipulação e Análise de Dados**: **Pandas**, **NumPy**, **Matplotlib**
* **Orquestração**: Os scripts podem ser executados diretamente com Python (`runner.py`).

---

## ⚙️ Configuração e Instalação

Para executar este projeto, siga os passos abaixo.

### Pré-requisitos
* Um computador com **Windows 10/11** (necessário para o SDK do Kinect).
* Um sensor **Microsoft Kinect v2** ou uma camera **Orbbec Femto Mega** com seus respectivos adaptadores de energia e cabos USB 3.0.
* Uma porta **USB 3.0** livre.
* **Python 3.8** (a distribuição Anaconda é recomendada).

### Passos de Instalação para Kinect

1.  **Instale o SDK do Kinect para Windows 2.0**:
    * Baixe e instale o SDK do site oficial da Microsoft: [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).
    * Conecte seu sensor Kinect ao PC via USB 3.0 e a uma fonte de energia. Verifique seu campo de visão e funcionamento pelo Kinect Studio presente na instalação do Kinect SDK.

2.  **Configure o Ambiente Python**:
    * É altamente recomendável usar um ambiente virtual. Para criar com o Anaconda:
        ```bash
        conda create -n propicie_env python=3.8
        conda activate propicie_env
        ```

3.  **Instale as Bibliotecas Necessárias**:
    * Instale as dependências principais usando pip e utilize as versões recomendadas para evitar erros:
        ```bash
        pip install numpy==1.23.5
        pip install comtypes==1.1.14
        pip install pykinect2==0.1.0
        pip install mediapipe==0.10.9
        pip install opencv-python pandas openpyxl
        ```

4.  **Instale o PyKinect2**:
    * `PyKinect2` requer uma instalação manual. Clone o repositório oficial e execute o script de setup.
        ```bash
        git clone [https://github.com/Kinect/PyKinect2.git](https://github.com/Kinect/PyKinect2.git)
        cd PyKinect2
        python setup.py install
        ```

5.  **Para alterações nos textos do programa**:
    * Após alterar os arquivos .po em locale compile-os para .mo.
   ```bash
   pybabel compile -i locale/en_US/LC_MESSAGES/messages.po -o locale/en_US/LC_MESSAGES/messages.mo
   pybabel compile -i locale/pt_PT/LC_MESSAGES/messages.po -o locale/pt_PT/LC_MESSAGES/messages.mo
   ```
---

### Passos de Instalação para Orbbec

#### Se a versão Kinect já foi rodada: 

1.  **Instale o SDK da Orbbec para Windows v2**:
    * Baixe e instale o SDK do github open source oficial da Orbbec: [Orbbec for Windows SDK 2.0](https://github.com/orbbec/OrbbecSDK/releases).
    * Conecte sua camera Orbbec ao PC via USB 3.0 e a uma fonte de energia. Verifique seu campo de visão e funcionamento pelo Orbbec Viewer presente na instalação do Orbbec SDK.

2.  **Para alterações nos textos do programa**:
    * Após alterar os arquivos .po em locale compile-os para .mo.
   ```bash
   pybabel compile -i locale/en_US/LC_MESSAGES/messages.po -o locale/en_US/LC_MESSAGES/messages.mo
   pybabel compile -i locale/pt_PT/LC_MESSAGES/messages.po -o locale/pt_PT/LC_MESSAGES/messages.mo
   ```
---

#### Se é a primeira vez que o sistema rodará: 

1.  **Instale o SDK da Orbbec para Windows v2**:
    * Baixe e instale o SDK do github open source oficial da Orbbec: [Orbbec for Windows SDK 2.0](https://github.com/orbbec/OrbbecSDK/releases).
    * Conecte sua camera Orbbec ao PC via USB 3.0 e a uma fonte de energia. Verifique seu campo de visão e funcionamento pelo Orbbec Viewer presente na instalação do Orbbec SDK.

2.  **Configure o Ambiente Python**:
    * É altamente recomendável usar um ambiente virtual. Para criar com o Anaconda:
        ```bash
        conda create -n propicie_env python=3.8
        conda activate propicie_env
        ```

3.  **Instale as Bibliotecas Necessárias**:
    * Instale as dependências principais usando pip e utilize as versões recomendadas para evitar erros:
        ```bash
        pip install numpy==1.23.5
        pip install comtypes==1.1.14
        pip install pykinect2==0.1.0
        pip install mediapipe==0.10.9
        pip install opencv-python pandas openpyxl
        ```

4.  **Para alterações nos textos do programa**:
    * Após alterar os arquivos .po em locale compile-os para .mo.
   ```bash
   pybabel compile -i locale/en_US/LC_MESSAGES/messages.po -o locale/en_US/LC_MESSAGES/messages.mo
   pybabel compile -i locale/pt_PT/LC_MESSAGES/messages.po -o locale/pt_PT/LC_MESSAGES/messages.mo
   ```
---


## ▶️ Uso

Após a conclusão da configuração, você pode executar as avaliações.

### Executando a Suíte de Testes Completa
Para executar o programa:

```bash
python runner.py
```

No canto inferior esquerdo da tela é possível escolher  a linguagem desejada, também é possível selecionar entre as opções:
* **Automático**: Inicia-se com o Sentar e Alcançar, repete 2 vezes para cada lado, e depois segue para o Alcançar atrás das Costas, também repete 2 vezes para cada lado, ao final retorna para o menu.
* **Sentar e Alncançar**: Repete 2 vezes para cada lado, ao final retorna para o menu.
* **Alcançar atrás das Costas**: Repete 2 vezes para cada lado, ao final retorna para o menu.
* **Visualizar Dados**: Mostra os dados de acordo com as opções selecionadas em forma de gráfico.
* **Encerrar Sessão**: Finaliza o programa.

### O Processo
1.  Quando um script é iniciado, uma janela aparecerá solicitando as informações do usuário (Idade, Altura, Peso, Gênero). Preencha os campos e pressione `Enter`.
2.  Uma repetição do exercício será executado.
3.  A janela principal da aplicação será aberta, mostrando o feed da câmera do Kinect com a sobreposição do esqueleto do MediaPipe.
4.  Siga as instruções no Guia do Utilizador para se posicionar corretamente e observe o texto no topo da tela.
5.  O sistema detectará automaticamente quando você estiver na postura correta, manterá a pose e, em seguida, calculará o resultado.
6.  Em seguida, uma janela solicitará a distância real medida. Isso é usado para validação e cálculo de erro. Insira o valor e pressione `Enter`.
7.  O resultado será exibido, e você será solicitado a continuar (`Enter`) ou sair (`ESC`).

---

## 📁 Estrutura do Projeto

```
.
├── /analises/                 # Scripts e resultados para análise estatística dos dados.
├── /arquivos/
      ├── relatorios           # Relatórios de progresso e finais detalhados.
      ├── tabelas_testes       # Planilhas de dados de teste.
      └── tabelas_utentes      # Planilhas com dados coletados dos testes com usuários.
├── /exercicios/
      ├── back-scratch.py      # Script Python para o teste de Alcançar atrás das Costas.
      └── sit-and-reach.py     # Scripts Python para o teste de Sentar e Alcançar.
├── /locale/                   # Contém os arquivos de texto com suporte de linguagem.
├── /ui/
      ├── draw.py              # Script para criação de telas padronizadas.
      ├── exercise_intro.py    # Script para mostrar qual o próximo exercício e quais foram feitos.
      ├── language_select.py   # Script para primeira tela e seleção de linguagem.
      ├── menu.py              # Script da tela de Menu: Automático, Sentar e Alcançar, Alcançar atrás das Costas, Visualizar Dados e Terminar Sessão.
      ├── theme.py             # Registra as cores e padrões.
      └── view_data.py         # Cria a tela de visualização dos dados.
├── .gitignore                 # Especifica arquivos a serem ignorados pelo Git.
├── camera.py                  # SOMENTE NO FORK ORBBEC - Recebe os dados da camera e os converte.
├── config.py                  # Dados fixos para os cálculos do sistema.
├── locale_setup.py            # Aplica a linguagem utilizando os arquivos em locale.
├── runner.py                  # Um script Python simples para executar todos os testes.
├── utils.py                   # Script com todas as funções comuns, que antes eram duplicadas.
└── README.md                  # Este arquivo.
```

---

## Agradecimentos

* Este trabalho faz parte de uma colaboração de pesquisa entre o **Instituto Politécnico de Beja (IPBeja)** e o **Instituto Federal de Santa Catarina (IFSC)**.
* Este projeto contribui para o projeto mais amplo **CAPACITA**, que visa desenvolver ferramentas digitais para avaliar e melhorar as capacidades físicas da população idosa.
