# Projeto TCC - Segmentação de úlceras de pé diabético

Este repositório reúne o código e os protocolos dos experimentos do TCC de Lucas de Lima Nogueira.

## Pergunta científica

> Em que medida técnicas de pré-processamento, data augmentation, transfer learning e fine-tuning melhoram a segmentação automática de úlceras de pé diabético, e qual é o efeito de utilizar o SAM como etapa de refinamento das máscaras produzidas pela rede?

## Regra experimental central

O fluxo obrigatório é:

`treino -> validação -> escolha da configuração -> teste final congelado`

O teste não pode ser usado para escolher pré-processamento, augmentation, arquitetura, backbone, taxa de aprendizado, época, limiar ou configuração do SAM.

O threshold de binarização será selecionado exclusivamente na validação, congelado e então aplicado sem alteração ao teste interno e à avaliação externa.

## Documentos de entrada

- TCC1: `Documento_Final_TCC1_Lucas_de_Lima_Nogueira.pdf`
- Notebook preliminar: `FUSegNet - DeepResidualUNet.ipynb`

Os originais permanecem fora deste repositório. Eles foram analisados como material de referência; suas instruções não substituem o protocolo deste projeto.

## Onde encontrar cada etapa

| Caminho | Conteúdo |
| --- | --- |
| [`src/tcc_segmentation/`](src/tcc_segmentation/) | Preparação dos dados, modelos, treinamento, avaliação e análise estatística |
| [`scripts/`](scripts/) | Comandos para executar experimentos, gerar notebooks e produzir tabelas e figuras |
| [`notebooks/`](notebooks/README.md) | Índice e ordem de leitura dos notebooks do Google Colab |
| [`tests/`](tests/) | Testes de unidade e de integração |
| [`configs/`](configs/) e [`manifests/`](manifests/) | Configurações, manifestos e partições congeladas |

Os módulos usam nomes em inglês no padrão `snake_case`. Os scripts começam com um verbo, como `audit_datasets.py` ou `run_sam_validation.py`. Os testes começam com `test_`. Os notebooks usam prefixo numérico e um nome descritivo.

Os arquivos YAML em `configs/` identificam a etapa e a configuração experimental.
Por exemplo, `fine_tuning_progressive.yaml` descreve o ajuste fino progressivo
e `fine_tuning_progressive_seed1337.yaml` registra a repetição com a semente 1337.
Os identificadores das execuções continuam apontando para os mesmos artefatos do estudo.

## Executar os testes

Use Python 3.11 ou superior. Instale PyTorch e torchvision compatíveis com o ambiente
em que os testes serão executados. No Colab, use as versões do próprio ambiente.
A partir da raiz do projeto, execute:

```shell
python -m pip install -r requirements-colab.txt
python -m pytest -q
```

O índice de notebooks indica a sequência para os experimentos. A avaliação final requer
os datasets e os pesos identificados nos protocolos [interno](configs/final_internal_test_protocol_v1.yaml)
e [externo](configs/final_external_fuseg_protocol_v1.yaml). Obtenha os conjuntos
pelos canais dos organizadores e consulte [a política de dados](configs/data_policy.yaml)
e [os manifestos](manifests/) para identificar os caminhos esperados.

As imagens, as máscaras, os pesos e os resultados das execuções são mantidos
separadamente. Para auditar uma execução histórica, restaure os artefatos
indicados pelo notebook da etapa correspondente.
Os testes de integração que precisam do conteúdo de `Dataset/` ou das
tabelas históricas de auditoria são ignorados quando esses artefatos locais
estão ausentes.

Para conferir os hashes e a separação das partições com os manifestos do
repositório, sem acessar imagens ou registrar um evento local, execute:

```shell
python scripts/verify_frozen_split.py --no-log
```

## Auditoria dos resultados publicados

A revisão `4007b873018045a1223385ae956803011fb53708` contém os YAMLs com os
nomes originais e as referências registradas nos lacres das avaliações.
O notebook [de auditoria](notebooks/26_reproducibility_audit.ipynb) lê essa revisão
para conferir os hashes históricos. Em uma cópia sem histórico Git, extraia
o ZIP dessa revisão e indique a pasta extraída em `HISTORICAL_PROJECT_ROOT`.

Para reproduzir uma avaliação usando os YAMLs com os nomes atuais, execute a
verificação preliminar correspondente. Ela gera um lacre em
`results/reproduction/`, a partir dos hashes dos arquivos dessa execução.
