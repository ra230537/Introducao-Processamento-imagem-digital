import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
from skimage.filters import threshold_otsu
import argparse

# Constantes para classificação de área
SMALL_AREA_THRESHOLD = 1500
LARGE_AREA_THRESHOLD = 3000


def parse_args():
    parser = argparse.ArgumentParser(description="Medidas de objetos em imagem colorida.")
    parser.add_argument("-i", "--input", required=True, help="Caminho da imagem PNG de entrada")
    parser.add_argument("-o", "--output_dir", default="../output/", help="Pasta para salvar resultados")
    return parser.parse_args()


def load_and_convert_image(image_path):
    """Carrega a imagem e converte para tons de cinza."""
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray


def extract_regions(img_gray):
    """Extrai e rotula regiões da imagem."""
    thresh = threshold_otsu(img_gray)
    img_bin = (img_gray < thresh)
    labels = measure.label(img_bin)

    return img_bin, labels


def extract_and_save_properties(labels, img_gray, output_dir):
    """Extrai e salva propriedades das regiões em arquivo."""
    props = measure.regionprops(labels, intensity_image=img_gray)

    with open(os.path.join(output_dir, "propriedades_regioes.txt"), "w") as f:
        for i, reg in enumerate(props):
            area = reg.area
            perimeter = reg.perimeter
            eccentricity = reg.eccentricity
            solidity = reg.solidity
            f.write(f"Região {i}: área={area} perímetro={perimeter:.6f} "
                    f"excentricidade={eccentricity:.6f} solidez={solidity:.6f}\n")

    return props


def save_contours_image(img_color, img_bin, output_dir):
    """Gera e salva imagem com contornos destacados."""
    img_bin_uint8 = (img_bin * 255).astype(np.uint8)
    contours_cv, _ = cv2.findContours(img_bin_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_contours = np.ones_like(img_color) * 255
    cv2.drawContours(img_contours, contours_cv, -1, (0, 0, 255), 2)

    output_path = os.path.join(output_dir, "contornos.png")
    cv2.imwrite(output_path, img_contours)


def save_labeled_regions(labels, props, output_dir):
    """Gera e salva imagem com regiões rotuladas em cores e numeradas."""
    img_label_rgb = color.label2rgb(labels, bg_label=0, bg_color=(1, 1, 1))
    img_label_rgb_uint8 = (img_label_rgb * 255).astype(np.uint8)

    img_label_bgr = cv2.cvtColor(img_label_rgb_uint8, cv2.COLOR_RGB2BGR)

    # Adiciona números nas regiões
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    for i, reg in enumerate(props):
        centroid_y, centroid_x = reg.centroid

        text = str(i)

        # Calcular o tamanho do texto para centralizá-lo
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calcular posição para centralizar o texto no centróide
        text_x = int(centroid_x - text_width // 2)
        text_y = int(centroid_y + text_height // 2)

        # Adicionar texto centralizado
        cv2.putText(img_label_bgr, text, (text_x, text_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    output_path = os.path.join(output_dir, "regioes_rotuladas.png")
    cv2.imwrite(output_path, img_label_bgr)
    output_path = os.path.join(output_dir, "regioes_rotuladas.png")
    cv2.imwrite(output_path, img_label_bgr)


def classify_objects_by_area(props, output_dir):
    """Classifica objetos por área e salva contagens em arquivo."""
    cont_pequenas = cont_medias = cont_grandes = 0
    areas = []

    for reg in props:
        area = reg.area
        areas.append(area)
        if area < SMALL_AREA_THRESHOLD:
            cont_pequenas += 1
        elif area < LARGE_AREA_THRESHOLD:
            cont_medias += 1
        else:
            cont_grandes += 1

    # Salvar classificação em arquivo
    with open(os.path.join(output_dir, "classification_areas.txt"), "w") as f:
        f.write(f"Quantidade de regiões: {cont_grandes + cont_medias + cont_pequenas}\n")
        f.write(f"Regiões pequenas: {cont_pequenas}\n")
        f.write(f"Regiões médias: {cont_medias}\n")
        f.write(f"Regiões grandes: {cont_grandes}\n")

    return cont_pequenas, cont_medias, cont_grandes, areas


def generate_area_histogram(areas, output_dir):
    """Gera e salva histograma de áreas dos objetos."""
    plt.figure()
    plt.hist(areas, bins=10, edgecolor='black')
    plt.title("Histograma de Áreas dos Objetos")
    plt.xlabel("Área (pixels)")
    plt.ylabel("Quantidade de objetos")
    plt.axvline(x=SMALL_AREA_THRESHOLD, color='red', linestyle='--')
    plt.axvline(x=LARGE_AREA_THRESHOLD, color='red', linestyle='--')

    output_path = os.path.join(output_dir, "histograma_areas.png")
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Carregar e converter imagem
    img_color, img_gray = load_and_convert_image(args.input)

    # Extrair regiões conectadas
    img_bin, labels = extract_regions(img_gray)

    # Extrair e imprimir propriedades
    props = extract_and_save_properties(labels, img_gray, args.output_dir)

    # Salvar imagem com contornos
    save_contours_image(img_color, img_bin, args.output_dir)

    # Salvar regiões rotuladas
    save_labeled_regions(labels, props, args.output_dir)

    # Classificar objetos por área
    cont_pequenas, cont_medias, cont_grandes, areas = classify_objects_by_area(props, args.output_dir)

    # Gerar histograma de áreas
    generate_area_histogram(areas, args.output_dir)


if __name__ == "__main__":
    main()
