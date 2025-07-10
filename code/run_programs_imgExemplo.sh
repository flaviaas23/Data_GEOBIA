#!/bin/bash

# Define o caminho do Python se necessário
PYTHON="python"

# Lista de programas a serem executados
programs=(
    # para imagem exemplo
    "python spk_gen_df_ToPCA_nb_term_save_sdft.py -sd data/img_20LMR/ -td tif/Rondonia-20LMR/ -q 9 -ld data/img_20LMR/logs/ -i 20LMR -pd data/img_20LMR/ -sp 0"
    "python spk_gen_df_ToPCA_nb_term_from_sdft.py -sd data/img_20LMR/ -q 9 -ld data/img_20LMR/logs/ -i 20LMR -pd data/img_20LMR/ -sp 1"
    #20250618: rodando no meu lap: para salvar o df with features 
    "python spk_gen_pca.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_20LMR/ -q 9 -pi 2 -ld data/img_20LMR/logs/ -i 20LMR -pd data/img_20LMR/ -rf 0 -sp 1"
    #20250618: rodando no meu lap: para salvar o df with features scaled e gen pca 
    "python spk_gen_pca.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_20LMR/ -q 9 -pi 2 -ld data/img_20LMR/logs/ -i 20LMR -pd data/img_20LMR/ -rf 1 -nc 6 -sp 1"
    #20250618: rodando no meu lap: para gerar as imagens PCA da img full
    "python spk_gen_imgPCA.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_20LMR/ -q 9 -pfi 1 -ld data/img_20LMR/logs/ -i 20LMR -pd data/img_20LMR/ -nc 6 -pfi 1 -isz 1200 -sp 1"

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
    
    #para executar com a SM2
    "spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0 -md 1 -sm 2"
    "spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0 -md 1 -sm 2"
    "spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 2 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0 -md 1 -sm 2"
    "spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 2 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0 -md 1 -sm 2"
    "spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 3 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0 -md 1 -sm 2"
    "spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 3 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 0 -md 1 -sm 2"

    #para executar com a imagem completa
    # para gerar um sdft por data da img completa tive que rodar para cada data para gerar um sdft por data
    "spk_gen_df_ToPCA_nb_term_save_sdft.py -sd data/img_031027/ -td tif/reg_30d/ -q 9 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    #para gerar um df_toPCA com todos os sdft's da img completa
    # executei 2 vezes primeiro com as dates[:6] e depois com dates[6:]
    # gerei 2 dfs to PCA
    # mudança feita no codigo direto
    "spk_gen_df_ToPCA_nb_term_from_sdft.py -sd data/img_031027/ -q 9 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    #criei o programa abaixo para juntar os 2 df_toPCA 's
    "spk_gen_df_ToPCA_imgfull.py -sd data/img_031027/ -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 0"
    # gerar o df with features 
    "spk_gen_pca.py -sd data/img_031027/ -q 9 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 0 -sp 1"
    
    #20250509: Gerando um df_toPCA de um quadrante  com linhas pares e impares
    # por linha da matriz no df_toPCA da matriz
    "spk_gen_df_ToPCA_pares_impares.py -sd data/img_031027/ -q 4 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"
    # gerar o df with features para pca com o df de pares e impares
    "spk_gen_pca.py -sd data/img_031027/ -q 4 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 0 -sp 1"
    # gerar o df with features para pca com o df de impares e pares 
    "spk_gen_pca.py -sd data/img_031027/ -q 4 -pi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 0 -sp 1"

    #20250516: 
    "spk_gen_pca.py -sd data/img_031027/ -q 4 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -sp 0"

    #20250519: rodar o pca para a imagem completa nao conseguiu salvar ...
    "spk_gen_pca.py -sd data/img_031027/ -q 9 -pi 0 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -sp 0"    

    #20250520: gerar o sdf_toPCA da img full com linhas pares e impares
    # por linha da matriz no df_toPCA da matriz , linha par pixels impares
    # e linha impar pixels pares
    "spk_gen_df_ToPCA_pares_impares.py -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"

    #no colab /content/drive/MyDrive/Colab
    #20250521: funcionou no colab
    "spk_gen_pca.py -bd /content/drive/MyDrive/Colab/ -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 0 -sp 1"    
    #20250522: rodar no meu lap: deu ero qdo estava salvando o df
    "spk_gen_pca.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -sp 1"    
    #rodando no Colab deu erro ao salvar o df
    "spk_gen_pca.py -bd /content/drive/MyDrive/Colab/ -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -sp 1"    
    #20250523: rodando no meu lap: para salvar o df with features scaled
    "python spk_gen_pca.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -sp 1"
    #20250523: prog alterado para ler o df with features scaled e fazer o pca partir daí
    "python spk_gen_pca.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -rf 1 -rs 1 -sp 1"
    #20250528: prog alterado para ler o df with pca for full img e gerar a img pca
    "python spk_gen_imgPCA.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"
    "python spk_gen_imgPCA.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"
    "python spk_gen_imgPCA.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"
    "python spk_gen_imgPCA.py -bd /Users/flaviaschneider/Documents/flavia/Data_GEOBIA/ -sd data/img_031027/ -q 9 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"
    #20250529: rodando o resto do processo com as imagens do quadrantes feitas do pca da imagem full
    # rodei para os 4 quadrantes
    "python spk_gen_imgPCA_segmentation.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    "python spk_gen_imgPCA_segmentation.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    "python spk_gen_imgPCA_segmentation.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    "python spk_gen_imgPCA_segmentation.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    
    "python spk_gen_1cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    "python spk_gen_1cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    "python spk_gen_1cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    "python spk_gen_1cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -dsi data/img_031027/spark_pca_images/FromFull/ -sp 1"
    
    "python spk_gen_2cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1"
    "python spk_gen_2cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1"
    "python spk_gen_2cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1"
    "python spk_gen_2cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1"
    # calcula o cluster com SM1 
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    
    # Calcula n_opt_ inter e intra cluster com SM1
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 1"

    #calcula a SM2 do cluster com SM1
    "python spk_gen_2_3cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_3cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_3cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_3cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"

    # para rodar com a SM2:     
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_1cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"

    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 1 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 2 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 3 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"
    "python spk_gen_2_2cluster_snic_pca.py -sd data/img_031027/ -q 4 -pfi 1 -ld data/img_031027/logs/ -pd data/img_031027/ -sp 1 -md 1 -sm 2"

    #    
    "spk_gen_df_ToPCA_pares_impares.py -bd /content/drive/MyDrive/Colab/ -sd data/img_031027/ -q 9 -pi 2 -ld data/img_031027/logs/ -i 031027 -pd data/img_031027/ -sp 1"

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