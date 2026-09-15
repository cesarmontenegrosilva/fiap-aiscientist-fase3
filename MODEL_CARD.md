# Model Card — Risco de Não Alfabetização

## 1. Identificação

**Projeto:** FIAP AI Scientist — Tech Challenge Fase 3  
**Problema:** classificação supervisionada de risco de não alfabetização  
**Modelo final:** Logistic Regression  
**Biblioteca principal:** Scikit-learn  
**Versão do documento:** 1.0  
**Status:** prova de conceito acadêmica

---

## 2. Resumo do modelo

O modelo estima a probabilidade de um aluno pertencer à classe:

```text
risco_nao_alfabetizado = 1
```

A classe positiva representa alunos classificados como não alfabetizados na Avaliação da Alfabetização.

O modelo foi desenvolvido com finalidade acadêmica e analítica. Ele não foi projetado para substituir avaliação pedagógica, julgamento profissional ou processos formais de decisão pública.

---

## 3. Uso pretendido

### Adequado para

- análise exploratória;
- comparação de fatores associados ao risco;
- investigação de padrões territoriais;
- construção de indicadores agregados;
- apoio à geração de hipóteses;
- demonstração de pipeline de Machine Learning;
- apoio analítico a gestores e pesquisadores.

### Não adequado para

- decisão automática sobre um aluno;
- classificação oficial de alfabetização;
- definição de sanções, benefícios ou direitos;
- diagnóstico pedagógico individual;
- inferência causal;
- ranking oficial de municípios;
- implantação em produção sem validações adicionais.

---

## 4. Dados utilizados

### Fonte principal

Avaliação da Alfabetização — INEP.

### Enriquecimentos

- Censo Escolar 2023 — INEP;
- IBGE 2022.

### Unidade de análise

Uma linha por aluno.

### Tamanho da amostra

```text
5.000 alunos
```

### Distribuição do target

| Classe | Quantidade | Percentual |
|---|---:|---:|
| 0 — sem risco | 2.618 | 52,36% |
| 1 — risco | 2.382 | 47,64% |

---

## 5. Target

```text
risco_nao_alfabetizado
```

Definição:

```text
1 = não alfabetizado
0 = alfabetizado
```

O target é derivado de `alfabetizado_oficial`.

---

## 6. Features

O dataset curado possui 31 features efetivas.

Principais grupos:

### Educacionais

```text
tipo_dependencia
pct_alfabetizados_2023
meta_2024
```

### Infraestrutura educacional municipal

Inclui variáveis de:

- número de escolas;
- localização urbana;
- dependência administrativa;
- conectividade;
- biblioteca/sala de leitura;
- laboratório de informática;
- água;
- energia;
- esgoto;
- coleta de lixo;
- quadra;
- pátio;
- refeitório;
- auditório;
- salas e equipamentos.

### Socioeconômicas e territoriais

```text
sigla_uf
populacao_2022
pct_urbana_2022
pib_per_capita_2022
```

---

## 7. Variáveis explicitamente excluídas

### Data leakage

```text
proficiencia_lp
alfabetizado_oficial
alfabetizado_calculado_743
divergencia_indicador_743
pct_alfabetizados_2024
variacao_2023_2024_pp
gap_meta_2024_pp
atingiu_meta_2024
rankings de 2024
nivel_alfabetizacao
```

### Variáveis operacionais ou identificadores

```text
id_aluno
id_escola
codigo_municipio
nome_municipio
peso_aluno_lp
presenca_lp
preenchimento_lp
codigo_caderno_lp
```

`id_escola` é utilizado somente como grupo para validação.

---

## 8. Pré-processamento

O pré-processamento faz parte da `Pipeline` do Scikit-learn.

### Numéricas

```text
SimpleImputer(strategy="median")
StandardScaler()
```

### Categóricas

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

Essa integração reduz o risco de leakage durante imputação e transformação.

---

## 9. Estratégia de validação

Foi utilizado:

```text
StratifiedGroupKFold
```

com:

```text
group = id_escola
```

Objetivos:

- preservar aproximadamente a distribuição do target;
- reduzir sobreposição de contexto escolar entre folds;
- evitar que `id_escola` seja usado como feature.

Um dos folds foi reservado como holdout final, correspondendo aproximadamente a 20% da amostra.

---

## 10. Algoritmos avaliados

Foram comparados:

```text
Logistic Regression
Random Forest
HistGradientBoosting
```

### Cross-validation

| Modelo | Balanced Accuracy | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,609 | 0,602 | 0,595 | 0,647 | **0,609** |
| Random Forest | 0,583 | 0,562 | 0,563 | 0,606 | 0,564 |
| HistGradientBoosting | 0,573 | 0,517 | 0,537 | 0,601 | 0,562 |

A Logistic Regression foi selecionada por apresentar melhor desempenho de validação, especialmente em PR-AUC.

---

## 11. Hiperparâmetros finais

```text
model__C = 0.18329807108324356
```

Demais parâmetros seguem a configuração definida no pipeline de treinamento.

---

## 12. Performance no holdout

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,566 |
| Balanced Accuracy | 0,565 |
| Precision — risco | 0,545 |
| Recall — risco | 0,547 |
| F1 — risco | 0,546 |
| ROC-AUC | 0,611 |
| PR-AUC | 0,565 |

### Interpretação

A capacidade preditiva é moderada.

O modelo identifica parte relevante da classe de risco, mas ainda apresenta número significativo de falsos positivos e falsos negativos.

As métricas não sustentam uso autônomo para decisão individual.

---

## 13. Interpretabilidade

Foram aplicadas:

- Permutation Importance;
- SHAP Values.

### Top 10 SHAP

| Feature | Mean \|SHAP\| |
|---|---:|
| `sigla_uf` | 0,627 |
| `pct_alfabetizados_2023` | 0,427 |
| `tipo_dependencia` | 0,130 |
| `pct_escolas_lab_informatica` | 0,121 |
| `meta_2024` | 0,098 |
| `pct_escolas_coleta_lixo` | 0,096 |
| `media_salas_utilizadas` | 0,085 |
| `pct_escolas_urbanas` | 0,078 |
| `populacao_2022` | 0,073 |
| `pct_urbana_2022` | 0,069 |

### Observação

SHAP explica o comportamento do modelo. Não demonstra causalidade.

Uma feature importante para o modelo não deve ser automaticamente interpretada como intervenção de política pública eficaz.

---

## 14. Teste de robustez territorial

Como `sigla_uf` apareceu como principal variável no SHAP, foi realizado um teste removendo essa feature.

| Métrica | Com UF | Sem UF |
|---|---:|---:|
| Accuracy | 0,566 | 0,579 |
| Balanced Accuracy | 0,565 | 0,580 |
| Precision — risco | 0,545 | 0,554 |
| Recall — risco | 0,547 | 0,597 |
| F1 — risco | 0,546 | 0,575 |
| ROC-AUC | 0,611 | 0,603 |
| PR-AUC | 0,565 | 0,547 |

### Conclusão do teste

A remoção da UF:

- aumentou accuracy;
- aumentou balanced accuracy;
- aumentou recall;
- aumentou F1;
- reduziu pouco ROC-AUC;
- reduziu pouco PR-AUC.

Portanto, o modelo não depende exclusivamente da identidade da UF para manter capacidade preditiva. O sinal parece também estar distribuído entre variáveis educacionais, estruturais e socioeconômicas.

---

## 15. Risco municipal

As probabilidades individuais do conjunto de teste foram agregadas por município.

O resultado é destinado a análise exploratória.

O ranking municipal:

- utiliza somente a amostra;
- utiliza somente o holdout;
- impõe quantidade mínima de observações;
- não deve ser chamado de ranking oficial.

---

## 16. Limitações

### 16.1 Amostra

Foram utilizados 5.000 registros individuais, e não o universo completo.

### 16.2 Granularidade do Censo Escolar

O enriquecimento do Censo Escolar é municipal. Não representa necessariamente a infraestrutura da escola individual de cada aluno.

### 16.3 Desempenho moderado

ROC-AUC e PR-AUC indicam sinal preditivo, mas não performance suficiente para automação de decisões sensíveis.

### 16.4 Pouco histórico temporal

O dataset não possui série temporal longa suficiente para previsões robustas de metas futuras.

### 16.5 Generalização

O teste por grupos reduz leakage, mas uma validação geográfica ainda mais rigorosa, como leave-one-state-out, seria desejável antes de qualquer uso real.

### 16.6 Causalidade

As relações aprendidas são associativas.

Não há desenho causal.

---

## 17. Fairness e riscos de uso

O projeto envolve contexto educacional e diferenças territoriais.

Possíveis riscos incluem:

- reprodução de desigualdades históricas;
- associação excessiva entre localização e risco;
- interpretação indevida de diferenças territoriais;
- uso do score individual fora do contexto acadêmico;
- estigmatização de alunos ou municípios.

### Mitigações aplicadas

- exclusão de identificadores do modelo;
- análise de leakage;
- teste específico sem `sigla_uf`;
- interpretação agregada;
- documentação explícita das limitações.

### Recomendações adicionais

Antes de uso real:

- avaliar métricas por UF e outros grupos;
- investigar calibração;
- testar fairness;
- realizar validação temporal;
- validar em dados externos;
- envolver especialistas em educação.

---

## 18. Monitoramento recomendado

Caso o modelo fosse futuramente colocado em produção, deveriam ser acompanhados:

### Data drift

Mudanças na distribuição de:

- desempenho histórico;
- infraestrutura;
- população;
- urbanização;
- contexto econômico.

### Performance drift

Monitorar:

- recall;
- precision;
- PR-AUC;
- ROC-AUC;
- calibração.

### Fairness

Comparar desempenho entre:

- UFs;
- regiões;
- dependências administrativas;
- diferentes contextos territoriais.

---

## 19. Artefatos

### Modelo

```text
Data/models/best_model.joblib
```

### Metadados

```text
Data/models/model_metadata.json
```

### Métricas

```text
Data/reports/metricas_modelos_cv.csv
Data/reports/metricas_teste_final.csv
Data/reports/robustez_sem_sigla_uf.csv
```

### Interpretabilidade

```text
Data/reports/permutation_importance.csv
Data/reports/shap_importance.csv
```

### Predições

```text
Data/reports/predictions_test.csv
```

---

## 20. Reprodutibilidade

Seed:

```text
42
```

A reprodutibilidade depende de:

- mesmas versões das bibliotecas;
- mesmos arquivos de origem;
- mesmo seed;
- mesma lógica de preparação e split.

O pipeline preserva pré-processamento e estimador em um único objeto serializado.

---

## 21. Aprovação para uso

### Status atual

```text
APROVADO PARA:
- demonstração acadêmica
- análise exploratória
- discussão de políticas públicas em nível agregado
- geração de hipóteses

NÃO APROVADO PARA:
- decisão automática individual
- classificação oficial
- produção sem validação adicional
```

---

## 22. Próximas evoluções

- aumentar a amostra individual;
- usar o universo de microdados;
- adicionar histórico temporal;
- avaliar calibração;
- implementar validação geográfica;
- testar fairness;
- testar modelos adicionais;
- aprofundar análise de efeitos municipais;
- incorporar informações escolares diretamente caso uma chave compatível se torne disponível.
