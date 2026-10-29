# Árvores, Random Forest e Boosting — quem sobreviveu ao Titanic?

Prática de **IA Clássica** que compara três famílias de modelos baseados em árvores para prever a sobrevivência dos passageiros do Titanic: **Árvore de Decisão**, **Random Forest** e **Gradient Boosting**.

## Objetivo

Entender como cada modelo funciona, como ajustar seus hiperparâmetros principais e como avaliar o resultado comparando o desempenho no **treino** com o desempenho em **dados que o modelo não viu** (validação), para identificar overfitting.

## Dados

Dataset **Titanic** (Kaggle / seaborn-data): 891 passageiros, dos quais **38% sobreviveram**.

Variáveis usadas: `pclass` (classe do bilhete), `sex`, `age`, `sibsp` e `parch` (familiares a bordo), `fare` (valor da passagem) e `embarked` (porto de embarque).

Tratamentos aplicados:
- idades vazias preenchidas com a mediana;
- portos de embarque vazios preenchidos com `"S"`;
- `sex` convertido para 0/1 e `embarked` transformado em colunas 0/1 (one-hot encoding).

## Metodologia

- Validação cruzada estratificada com 5 partes (`StratifiedKFold`, `random_state=42`);
- Métrica: acurácia no treino e na validação;
- Modelos: `DecisionTreeClassifier`, `RandomForestClassifier` e `GradientBoostingClassifier`;
- Hiperparâmetros testados: `max_depth`, `n_estimators` e `learning_rate`.

## Resultados

| Modelo | Treino | Validação |
|---|---|---|
| Árvore sem limite | 98,2% | 78,9% |
| Árvore `max_depth=2` | 79,8% | 77,7% |
| Árvore `max_depth=4` | 83,9% | 81,8% |
| Árvore `max_depth=8` | 90,3% | 82,9% |
| Random Forest, 100 árvores | 98,2% | 82,5% |
| Random Forest, 300 árvores, `max_depth=8` | 92,1% | **84,3%** |
| Boosting `learning_rate=0,01` | 83,5% | 81,5% |
| Boosting `learning_rate=0,1` | 90,1% | 82,7% |
| Boosting `learning_rate=1,0` | 98,2% | 81,0% |

Os valores podem variar um pouco entre execuções e ambientes.

## Principais conclusões

- **Árvore sem limite decora o treino:** 98,2% de acerto no treino contra 78,9% na validação. Limitar a profundidade reduz o overfitting.
- **Random Forest foi a melhor:** 84,3% de validação, cerca de 2,5 pontos acima da melhor árvore podada (`max_depth=4`) e mais estável. Mais árvores não causam overfitting, só aumentam o tempo de treino.
- **Boosting precisa de ajuste:** com `learning_rate=1,0` o treino chegou a 98,2% e a validação caiu para 81,0%, sinal de overfitting.
- **Modelo escolhido para produção:** Random Forest com 300 árvores e `max_depth=8`, pela maior validação e por uma distância razoável entre treino e validação.

## Variáveis mais importantes

Segundo a Random Forest, as mais usadas foram `sex` (cerca de 42%), `fare` (18%), `age` (14%) e `pclass` (13%). Isso faz sentido historicamente: a regra "mulheres e crianças primeiro" e o acesso aos botes, que favoreceu a primeira classe, aparecem nos dados.

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/titanic-arvores-rf-boosting.git
   cd titanic-arvores-rf-boosting
   ```
2. Instale as dependências:
   ```bash
   pip install pandas matplotlib scikit-learn jupyter
   ```
3. Abra o notebook `pratica_alunos_resolvido.ipynb` e execute todas as células. O dataset é baixado automaticamente da URL indicada no código, então é preciso ter conexão com a internet.

## Tecnologias

Python, pandas, scikit-learn, matplotlib e Jupyter Notebook.

## Autor

Joao Pedro Lima Barbosa 
