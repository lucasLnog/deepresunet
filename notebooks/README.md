# Notebooks do estudo

Os nomes seguem `NN_etapa.ipynb`. A numeração indica a ordem de leitura. Abra o notebook da etapa desejada no Colab, confira `PROJECT_ROOT` na primeira célula de configuração e siga as instruções da etapa. Os geradores ficam em [scripts](../scripts/).

| Etapa | Notebooks |
| --- | --- |
| Ambiente e verificações iniciais | [Ambiente Colab](00_colab_setup.ipynb), [overfit em amostra reduzida](01_training_overfit_smoke.ipynb), [migração da DeepResUNet](02_deep_res_unet_migration.ipynb), [baselines](03_development_baselines.ipynb), [encoder ResNet-34](04_deep_res_unet_resnet34_check.ipynb) |
| Dados e baseline D0 | [União de máscaras duplicadas](05_duplicate_mask_union.ipynb), [sanidade do pipeline](06_pipeline_sanity.ipynb), [treinamento D0](07_baseline_d0_training.ipynb) |
| Pré-processamento e aumento de dados | [Auditoria do pré-processamento](08_preprocessing_audit.ipynb), [treinamento P1/P2](09_preprocessing_training.ipynb), [auditoria do aumento de dados](10_augmentation_audit.ipynb), [treinamento A1/A2](11_augmentation_training.ipynb), [combinação P1+A1](12_combined_preprocessing_augmentation.ipynb) |
| Transferência e ajuste fino | [DR-RT e DR-IT](13_transfer_learning.ipynb), [DR-IF e DR-IFT](14_fine_tuning.ipynb), [confirmação com múltiplas sementes](15_multiseed_confirmation.ipynb) |
| Refinamento e métricas de contorno | [Integração SAM](16_sam_integration_smoke.ipynb), [SAM na validação](17_sam_validation.ipynb), [MedSAM na validação](18_medsam_validation.ipynb), [Boundary Loss](19_boundary_loss_ablation.ipynb), [Boundary IoU](20_boundary_iou_validation.ipynb) |
| Avaliação final | [Verificação interna](21_internal_dfuc2022_preflight.ipynb), [teste interno DFUC2022](22_internal_dfuc2022_evaluation.ipynb), [verificação externa](23_external_fuseg_preflight.ipynb), [avaliação externa FUSeg](24_external_fuseg_evaluation.ipynb) |
| Análise e rastreabilidade | [Análise estatística e de erros](25_statistical_error_analysis.ipynb), [auditoria de reprodutibilidade](26_reproducibility_audit.ipynb) |

As etapas de verificação interna e externa conferem os arquivos e seus hashes antes da inferência. A auditoria de reprodutibilidade lê as tabelas e os registros de execuções já concluídas.
