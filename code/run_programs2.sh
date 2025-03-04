#!/bin/bash

# Define o caminho do Python se necessário
PYTHON="python"

# Define variables for program options
SD="data/img_031027/"                   # Save directory
TD="tif/reg_30d/"                       # Tif directory
Q=4                                     # Quadrant
LD="data/img_031027/logs/"              # Logs directory
I="031027"                              # Image ID/Name
PD="data/img_031027/"                   # Processed time directory
DSI="data/img_031027/spark_pca_images/" # Directory for Spark PCA images
SP=0                                    # Show prints (debug)
RF=1                                    # Read df_with fetures to pca it
# Lista de programas a serem executados
programs=(
    "spk_gen_df_ToPCA_nb_term_save_sdft.py -sd $SD -td $TD -q $Q -ld $LD -i $I -pd $PD -sp $SP"
    "spk_gen_df_ToPCA_nb_term_from_sdft.py -sd $SD -q $Q -ld $LD -i $I -pd $PD -sp $SP"
    "spk_gen_pca.py -sd $SD -q $Q -ld $LD -i $I -pd $PD -sp $SP"
    "spk_gen_pca.py -sd $SD -q $Q -ld $LD -i $I -pd $PD -sp $SP -rf $RF"
    "spk_gen_imgPCA.py -sd $SD -q $Q -ld $LD -i $I -pd $PD -sp $SP"
    "spk_gen_imgPCA_segmentation.py -sd $SD -q $Q -ld $LD -i $I -pd $PD -dsi $DSI -sp $SP"
    "spk_gen_1cluster_snic_pca.py -sd $SD -q $Q -ld $LD -i $I -pd $PD -dsi $DSI -sp $SP"
    "spk_gen_2cluster_snic_pca.py -sd $SD -q $Q -ld $LD -pd $PD -sp $SP"
    "spk_gen_2_1cluster_snic_pca.py -sd $SD -q $Q -ld $LD -pd $PD -sp $SP"
    "spk_gen_2_2cluster_snic_pca.py -sd $SD -q $Q -ld $LD -pd $PD -sp $SP"
    "spk_gen_2_3cluster_snic_pca.py -sd $SD -q $Q -ld $LD -pd $PD -sp $SP"
)

# Executa cada programa na ordem
for program in "${programs[@]}"; do
    echo "Executing $program..."
    $PYTHON $program "$@"
    if [ $? -ne 0 ]; then
        echo "Error occurred while running $program. Exiting."
        exit 1
    fi
    echo "$program executed successfully."
done

echo "All programs executed successfully."