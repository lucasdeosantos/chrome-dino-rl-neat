# Chrome Dino: Treinamento e Avaliação de agentes com RL e NEAT

Uma adaptação do jogo do dinossauro do chrome para treinar e avaliar agentes com PPO, DQN e NEAT.

## Instalação

```bash
git clone https://github.com/lucasdeosantos/chrome-dino-rl-neat.git
cd chrome-dino-rl-neat

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Jogar

```bash
python play.py
python play.py --scenario 2
python play.py --versus models/neat/s3/seed42/best.pkl --scenario 3
```

| Opção | Descrição |
|---|---|
| `--scenario [1, 2, 3, 4]` | Cenário (padrão: 4) |
| `--versus ARQUIVO` | Joga contra um agente |

### Cenários

| Cenário | Obstáculos | Velocidade |
|---|---|---|
| 1 | Apenas cactos | 360 px/s |
| 2 | Apenas pterodáctilos | 360 px/s |
| 3 | Cactos e pterodáctilos | 360 px/s |
| 4 | Cactos e pterodáctilos (a partir de 510 px/s) | Cresce de 360 até 780 px/s |

Os cactos podem ser pequenos ou grandes, aparecendo em grupos de 1 a 3 unidades. Já os pterodáctilos podem surgir em quatro alturas diferentes (y = 100, 75, 50 ou 25), todas com a mesma probabilidade.

### Controles

| Tecla | Efeito |
|---|---|
| Cima / Espaço | Saltar |
| Baixo | Agachar |
| Enter / Espaço | Iniciar / Reiniciar |
| Esc | Sair |

## Ambiente de treinamento

O ambiente oferece três ações distintas:

| Valor | Ação |
|---|---|
| 0 | Correr |
| 1 | Pular |
| 2 | Agachar |

Para padronizar o tamanho do pulo, foram removidas do jogo a descida rápida e a variação de altura do salto conforme o tempo em que o botão fica pressionado.

### Estrutura da observação

A observação é um vetor de **19 números**, composto pelo estado do dinossauro, pela velocidade do cenário e pelos dois próximos obstáculos.

| Índice | Informação / Escala |
|---|---|
| 0 | Altura do dinossauro acima do chão / 150 |
| 1 | Velocidade vertical / 1000 |
| 2 | Está no chão?  (0 ou 1) |
| 3 | Está agachado? (0 ou 1) |
| 4 | Velocidade do cenário / 780 |
| 5–11 | Primeiro obstáculo |
| 12–18 | Segundo obstáculo |

A velocidade vertical é negativa quando o dinossauro está caindo.

### Informações de cada obstáculo

Cada obstáculo ocupa um bloco com a seguinte estrutura:

| Posição no bloco | Informação / Escala |
|---|---|
| 0 | Existe?  (0 ou 1) |
| 1 | Distância horizontal em relação ao dinossauro / 600 |
| 2 | Coordenada vertical do topo / 150 |
| 3 | Largura / 100 |
| 4 | Altura / 150 |
| 5 | Velocidade relativa / 780 |
| 6 | É pterodáctilo? (0 ou 1) |

## Treino e Avaliação
 
O treinamento e a avaliação utilizam mapas diferentes, definidos por sementes aleatórias:
 
| Etapa | Sementes | O que faz |
|---|---|---|
| Treino | 101 em diante | PPO/DQN trocam de mapa a cada partida, NEAT usa um mapa por geração |
| Avaliação | 1–100 | Avaliação final, sem aprendizado nem exploração |
 
Cada partida termina por colisão ou ao atingir o limite de passos.

### Escolha do melhor modelo

O melhor modelo (`best.zip`/`best.pkl`) é o de maior pontuação no treino. Se mais de um modelo atingirem a pontuação máxima, o mais recente vira o melhor.

No PPO e no DQN, `last.zip` guarda a rede ao final do treinamento. No NEAT, `last.pkl` guarda o melhor indivíduo da última geração.

## Treino (`train.py`)

```bash
python train.py --agent ppo  --scenario 1 --output models/ppo-new-model
python train.py --agent dqn  --scenario 1 --output models/dqn-new-model
python train.py --agent neat --scenario 1 --output models/neat-new-model
```

| Opção | Descrição |
|---|---|
| `--agent [ppo, dqn, neat]` | Algoritmo do modelo |
| `--scenario [1, 2, 3, 4]` | Cenário |
| `--output PASTA` | Pasta de saída do modelo |
| `--seed N` | Semente aleatória do algoritmo (padrão: 42) |
| `--timesteps N` | Passos do PPO/DQN (padrão: 10 milhões) |
| `--reward [r1..r9]` | Função de recompensa do PPO/DQN (padrão: r1) |
| `--generations N` | Gerações do NEAT (padrão: 100) |
| `--population N` | Tamanho da população do NEAT (padrão: 100) |
| `--max-steps N` | Limite de passos por partida (padrão: 36 mil passos = 10 minutos simulados) |
| `--stop-at-max-steps` | Encerra o treino quando um modelo atinge o limite definido em --max-steps. No NEAT, conclui a geração antes de encerrar |
| `--resume PASTA` | Continua um treino interrompido com os mesmos parâmetros |
| `--pretrained PASTA` | Novo treino a partir de um modelo do mesmo algoritmo, exige uma nova pasta de saída |
| `--render [normal\|2x\|4x\|8x]` | Visualização gráfica do treino (velocidade padrão: 8x) |

Hiperparâmetros do RL ficam em `src/training/config.py`, a configuração do NEAT fica em `neat_config.txt`.

### Funções de Recompensa

As funções de recompensa são definidas em `src/training/rewards.py`. O NEAT usa a pontuação da partida como *fitness* e não é afetado por `--reward`.

A função de recompensa padrão $r_1$ é calculada por:

$$
r_1 = 0{,}1 \cdot s
$$

Onde $s$ é o número de passos sobrevividos, que são convertidos na pontuação conforme a velocidade do cenário.

### Retomada e pré-treinamento

```bash
python train.py --resume models/dqn/s2/seed42
python train.py --agent dqn --pretrained models/dqn/s2/seed42/best.zip --scenario 4 --timesteps 1000000 --output models/dqn/s4/from-s2
```
- **Retomada** (`--resume`): permite continuar um treinamento interrompido a partir do checkpoint.
- **Pré-treinamento** (`--pretrained`): permite utilizar um modelo previamente treinado como ponto de partida para o treinamento em outro cenário.

## Avaliação (`evaluate.py`)

```bash
python evaluate.py --agent ppo  --model models/ppo-new-model/best.zip --scenario 1
python evaluate.py --agent neat --model models/neat-new-model/best.pkl --scenario 1 --render 2x
```

| Opção | Descrição |
|---|---|
| `--agent [ppo, dqn, neat]` | Algoritmo do modelo |
| `--scenario [1, 2, 3, 4]` | Cenário |
| `--model ARQUIVO` | Modelo (`best.zip`/`best.pkl` ou `last.zip`/`last.pkl`) |
| `--max-steps` | Limite de passos por partida (padrão: 36000 = 10 min simulados) |
| `--render [normal\|2x\|4x\|8x]` | Visualização gráfica da avaliação |

## Experimentos

```bash
python scripts/run_experiments.py
python scripts/run_rewards.py
```

- `run_experiments.py`: executa o treinamento e a avaliação dos algoritmos NEAT, DQN e PPO nos quatro cenários, utilizando as configurações padrão.

- `run_rewards.py`: executa o treinamento e a avaliação dos algoritmos DQN e PPO nos quatro cenários, utilizando cada uma das funções de recompensa definidas em `src/training/rewards.py` e as configurações padrão.

## Resultados e arquivos gerados

Em cada pasta de modelo:

| Arquivo | Conteúdo |
|---|---|
| `best.zip` / `best.pkl` | Melhor modelo do treino |
| `last.zip` / `last.pkl` | Modelo final de PPO/DQN ou melhor indivíduo da última geração do NEAT |
| `training_episodes.json` | Partidas de treinamento |
| `best_training.json` | Recorde de pontuação no treino |
| `training_result.json` | Resultado do treinamento |
| `run.json` | Configuração, sementes e versões |
| `effective_neat_config.txt` | Configuração do NEAT efetivamente usada |
| `checkpoint/` | Snapshots usados por `--resume` |
| `replay_buffer.pkl` | Checkpoints do RL |
| `neat-checkpoint-*` | Checkpoints do NEAT |

### Logs

Os resultados e registros de execução dos processos de treinamento e avaliação são armazenados em `results/logs/`. Durante a execução, o terminal exibe apenas o caminho do arquivo de log correspondente. Para acompanhar os logs em tempo real, utilize:

```bash
tail -f results/logs/<arquivo>.log
```
