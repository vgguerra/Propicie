import subprocess

print("Executando Sit and Reach...")
subprocess.run(["python", "./exercicios/sit_and_reach_holistic_2.py"], check=True)

print("Executando Back Scratch...")
subprocess.run(["python", "./exercicios/back_scratch.py"], check=True)

print("Testes concluídos com sucesso.")