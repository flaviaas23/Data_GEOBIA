#!/bin/bash

# Define o caminho do Python se necessário
PYTHON="python"

# Lista de programas a serem executados
programs=(
    "spk_gen_df_ToPCA_nb_term_save_sdft.py -sd data/img_031027/ -td tif/reg_30d/ -q 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    "spk_gen_df_ToPCA_nb_term_from_sdft.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    "spk_gen_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    "spk_gen_pca.py -sd data/img_031027/ -q 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -sp 0"
    "spk_gen_imgPCA.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    "spk_gen_imgPCA_segmentation.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/ -sp 0"
    "spk_gen_1cluster_snic_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/ -sp 0"
    "spk_gen_2cluster_snic_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0"
    "spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0"
    "spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0"
    "spk_gen_2_3cluster_snic_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0"
    "spk_gen_2_4cluster_snic_pca.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 0"
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