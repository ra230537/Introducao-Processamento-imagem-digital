import os
import sys
import argparse
import math
import numpy as np
import cv2


# test
def R(s):
    """Função kernel cúbica (B-spline) conforme fórmulas (5) e (6)"""
    s = abs(s)
    if s <= 1:
        return (1 / 6) * ((3 * s ** 3) - (6 * s ** 2) + 4)
    elif 1 < s < 2:
        return (1 / 6) * ((-s ** 3) + (6 * s ** 2) - (12 * s) + 8)
    else:
        return 0


def vizinho_mais_proximo(img_input, escala=None, angulo=None, largura=None, altura=None):
    """Implementa interpolação pelo vizinho mais próximo."""
    h_in, w_in = img_input.shape[:2]
    canais = img_input.shape[2] if img_input.ndim == 3 else 1

    # Inicializar imagem de saída
    if canais == 1:
        img_out = np.zeros((altura, largura), dtype=img_input.dtype)
    else:
        img_out = np.zeros((altura, largura, canais), dtype=img_input.dtype)

    # Pontos centrais
    cx_in = w_in / 2.0
    cy_in = h_in / 2.0
    cx_out = largura / 2.0
    cy_out = altura / 2.0

    # Laço sobre cada pixel de saída
    for y_out in range(altura):
        for x_out in range(largura):
            if escala is not None:
                x_in_f = x_out / escala
                y_in_f = y_out / escala
            else:  # rotação
                x0 = x_out - cx_out
                y0 = y_out - cy_out
                cos_t = math.cos(angulo)
                sin_t = math.sin(angulo)
                # rotação inversa
                x_rot = cos_t * x0 + sin_t * y0
                y_rot = -sin_t * x0 + cos_t * y0
                x_in_f = x_rot + cx_in
                y_in_f = y_rot + cy_in

            # Vizinho mais próximo: arredonda
            x_near = int(round(x_in_f))
            y_near = int(round(y_in_f))

            # Se dentro dos limites
            if 0 <= x_near < w_in and 0 <= y_near < h_in:
                img_out[y_out, x_out] = img_input[y_near, x_near]
            else:
                # fora, deixa preto
                if canais == 1:
                    img_out[y_out, x_out] = 0
                else:
                    img_out[y_out, x_out, :] = 0

    return img_out


def interp_bilinear(img_input, escala=None, angulo=None, largura=None, altura=None):
    """Implementa interpolação bilinear."""
    h_in, w_in = img_input.shape[:2]
    canais = img_input.shape[2] if img_input.ndim == 3 else 1

    # Inicializar imagem de saída
    if canais == 1:
        img_out = np.zeros((altura, largura), dtype=img_input.dtype)
    else:
        img_out = np.zeros((altura, largura, canais), dtype=img_input.dtype)

    # Pontos centrais
    cx_in = w_in / 2.0
    cy_in = h_in / 2.0
    cx_out = largura / 2.0
    cy_out = altura / 2.0

    for y_out in range(altura):
        for x_out in range(largura):
            # Calcular coordenadas na imagem de entrada
            if escala is not None:
                x_in_f = x_out / escala
                y_in_f = y_out / escala
            else:  # rotação
                x0 = x_out - cx_out
                y0 = y_out - cy_out
                cos_t = math.cos(angulo)
                sin_t = math.sin(angulo)
                x_rot = cos_t * x0 + sin_t * y0
                y_rot = -sin_t * x0 + cos_t * y0
                x_in_f = x_rot + cx_in
                y_in_f = y_rot + cy_in

            # Índices inteiros mais próximos à "cima e à esquerda"
            x0_int = int(math.floor(x_in_f))
            y0_int = int(math.floor(y_in_f))
            dx = x_in_f - x0_int
            dy = y_in_f - y0_int

            # Pegar quatro pixels vizinhos
            if (0 <= x0_int < w_in - 1) and (0 <= y0_int < h_in - 1):
                if canais == 1:
                    f00 = float(img_input[y0_int, x0_int])
                    f10 = float(img_input[y0_int, x0_int + 1])
                    f01 = float(img_input[y0_int + 1, x0_int])
                    f11 = float(img_input[y0_int + 1, x0_int + 1])
                    val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
                    img_out[y_out, x_out] = np.clip(val, 0, 255)
                else:
                    # Para cada canal
                    for c in range(canais):
                        f00 = float(img_input[y0_int, x0_int, c])
                        f10 = float(img_input[y0_int, x0_int + 1, c])
                        f01 = float(img_input[y0_int + 1, x0_int, c])
                        f11 = float(img_input[y0_int + 1, x0_int + 1, c])
                        val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
                        img_out[y_out, x_out, c] = np.clip(val, 0, 255)
            else:
                # Fora dos limites
                if canais == 1:
                    img_out[y_out, x_out] = 0
                else:
                    img_out[y_out, x_out, :] = 0

    return img_out


def interp_bicubica(img_input, escala=None, angulo=None, largura=None, altura=None):
    """Implementa interpolação bicúbica."""
    h_in, w_in = img_input.shape[:2]
    canais = img_input.shape[2] if img_input.ndim == 3 else 1

    # Inicializar imagem de saída
    if canais == 1:
        img_out = np.zeros((altura, largura), dtype=img_input.dtype)
    else:
        img_out = np.zeros((altura, largura, canais), dtype=img_input.dtype)

    # Pontos centrais
    cx_in = w_in / 2.0
    cy_in = h_in / 2.0
    cx_out = largura / 2.0
    cy_out = altura / 2.0

    for y_out in range(altura):
        for x_out in range(largura):
            # Calcular coordenadas na imagem de entrada
            if escala is not None:
                x_in_f = x_out / escala
                y_in_f = y_out / escala
            else:  # rotação
                x0 = x_out - cx_out
                y0 = y_out - cy_out
                cos_t = math.cos(angulo)
                sin_t = math.sin(angulo)
                x_rot = cos_t * x0 + sin_t * y0
                y_rot = -sin_t * x0 + cos_t * y0
                x_in_f = x_rot + cx_in
                y_in_f = y_rot + cy_in

            x0_int = int(math.floor(x_in_f))
            y0_int = int(math.floor(y_in_f))
            dx = x_in_f - x0_int
            dy = y_in_f - y0_int

            # Verificar se temos vizinhança 4x4 disponível
            if (1 <= x0_int < w_in - 2) and (1 <= y0_int < h_in - 2):
                if canais == 1:
                    val = 0.0
                    for j in range(-1, 3):
                        for i in range(-1, 3):
                            pixel_val = float(img_input[y0_int + j, x0_int + i])
                            peso = R(i - dx) * R(j - dy)
                            val += pixel_val * peso
                    img_out[y_out, x_out] = np.clip(val, 0, 255)
                else:
                    # Para cada canal
                    for c in range(canais):
                        val = 0.0
                        for j in range(-1, 3):
                            for i in range(-1, 3):
                                pixel_val = float(img_input[y0_int + j, x0_int + i, c])
                                peso = R(i - dx) * R(j - dy)
                                val += pixel_val * peso
                        img_out[y_out, x_out, c] = np.clip(val, 0, 255)
            else:
                # Fora dos limites para bicúbica, usar bilinear como fallback
                if (0 <= x0_int < w_in - 1) and (0 <= y0_int < h_in - 1):
                    if canais == 1:
                        f00 = float(img_input[y0_int, x0_int])
                        f10 = float(img_input[y0_int, x0_int + 1])
                        f01 = float(img_input[y0_int + 1, x0_int])
                        f11 = float(img_input[y0_int + 1, x0_int + 1])
                        val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
                        img_out[y_out, x_out] = np.clip(val, 0, 255)
                    else:
                        for c in range(canais):
                            f00 = float(img_input[y0_int, x0_int, c])
                            f10 = float(img_input[y0_int, x0_int + 1, c])
                            f01 = float(img_input[y0_int + 1, x0_int, c])
                            f11 = float(img_input[y0_int + 1, x0_int + 1, c])
                            val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
                            img_out[y_out, x_out, c] = np.clip(val, 0, 255)
                else:
                    if canais == 1:
                        img_out[y_out, x_out] = 0
                    else:
                        img_out[y_out, x_out, :] = 0

    return img_out


def interp_lagrange(img_input, escala=None, angulo=None, largura=None, altura=None):
    """Implementa interpolação por polinômios de Lagrange (usando 4 pontos)."""
    h_in, w_in = img_input.shape[:2]
    canais = img_input.shape[2] if img_input.ndim == 3 else 1

    # Inicializar imagem de saída
    if canais == 1:
        img_out = np.zeros((altura, largura), dtype=img_input.dtype)
    else:
        img_out = np.zeros((altura, largura, canais), dtype=img_input.dtype)

    # Pontos centrais
    cx_in = w_in / 2.0
    cy_in = h_in / 2.0
    cx_out = largura / 2.0
    cy_out = altura / 2.0

    def lagrange_weights(t):
        """Calcula os pesos de Lagrange para 4 pontos."""
        t0, t1, t2, t3 = -1, 0, 1, 2
        w0 = ((t - t1) * (t - t2) * (t - t3)) / ((t0 - t1) * (t0 - t2) * (t0 - t3))
        w1 = ((t - t0) * (t - t2) * (t - t3)) / ((t1 - t0) * (t1 - t2) * (t1 - t3))
        w2 = ((t - t0) * (t - t1) * (t - t3)) / ((t2 - t0) * (t2 - t1) * (t2 - t3))
        w3 = ((t - t0) * (t - t1) * (t - t2)) / ((t3 - t0) * (t3 - t1) * (t3 - t2))
        return w0, w1, w2, w3

    for y_out in range(altura):
        for x_out in range(largura):
            # Calcular coordenadas na imagem de entrada
            if escala is not None:
                x_in_f = x_out / escala
                y_in_f = y_out / escala
            else:  # rotação
                x0 = x_out - cx_out
                y0 = y_out - cy_out
                cos_t = math.cos(angulo)
                sin_t = math.sin(angulo)
                x_rot = cos_t * x0 + sin_t * y0
                y_rot = -sin_t * x0 + cos_t * y0
                x_in_f = x_rot + cx_in
                y_in_f = y_rot + cy_in

            x0_int = int(math.floor(x_in_f))
            y0_int = int(math.floor(y_in_f))
            dx = x_in_f - x0_int
            dy = y_in_f - y0_int

            # Verificar se temos vizinhança 4x4 disponível
            if (1 <= x0_int < w_in - 2) and (1 <= y0_int < h_in - 2):
                wx = lagrange_weights(dx)
                wy = lagrange_weights(dy)

                if canais == 1:
                    val = 0.0
                    for j in range(4):
                        for i in range(4):
                            pixel_val = float(img_input[y0_int - 1 + j, x0_int - 1 + i])
                            val += pixel_val * wx[i] * wy[j]
                    img_out[y_out, x_out] = np.clip(val, 0, 255)
                else:
                    # Para cada canal
                    for c in range(canais):
                        val = 0.0
                        for j in range(4):
                            for i in range(4):
                                pixel_val = float(img_input[y0_int - 1 + j, x0_int - 1 + i, c])
                                val += pixel_val * wx[i] * wy[j]
                        img_out[y_out, x_out, c] = np.clip(val, 0, 255)
            else:
                # Fora dos limites para Lagrange, usar bilinear como fallback
                if (0 <= x0_int < w_in - 1) and (0 <= y0_int < h_in - 1):
                    if canais == 1:
                        f00 = float(img_input[y0_int, x0_int])
                        f10 = float(img_input[y0_int, x0_int + 1])
                        f01 = float(img_input[y0_int + 1, x0_int])
                        f11 = float(img_input[y0_int + 1, x0_int + 1])
                        val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
                        img_out[y_out, x_out] = np.clip(val, 0, 255)
                    else:
                        for c in range(canais):
                            f00 = float(img_input[y0_int, x0_int, c])
                            f10 = float(img_input[y0_int, x0_int + 1, c])
                            f01 = float(img_input[y0_int + 1, x0_int, c])
                            f11 = float(img_input[y0_int + 1, x0_int + 1, c])
                            val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
                            img_out[y_out, x_out, c] = np.clip(val, 0, 255)
                else:
                    if canais == 1:
                        img_out[y_out, x_out] = 0
                    else:
                        img_out[y_out, x_out, :] = 0

    return img_out


def aplicar_transformacao(args):
    """
    De acordo com args, chama a função de interpolação correta e salva o resultado.
    """
    # 1) Carregar imagem de entrada (PNG)
    img_input = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img_input is None:
        print("Falha ao carregar a imagem de entrada.")
        exit(1)

    h_in, w_in = img_input.shape[:2]

    # 2) Determinar tipo de transformação e tamanho da saída
    if args.escala is not None:
        # ESCALA
        if args.dim:
            largura_out, altura_out = args.dim[0], args.dim[1]
            escala_calc = None  # Usar dimensões fixas
        else:
            escala_calc = args.escala
            largura_out = int(w_in * escala_calc)
            altura_out = int(h_in * escala_calc)

        angulo_calc = None
        nome_transformacao = f"escala_{args.escala}"

    else:
        # ROTAÇÃO
        theta = math.radians(args.angle)
        if args.dim:
            largura_out, altura_out = args.dim[0], args.dim[1]
        else:
            # Opção simples: manter mesmo tamanho
            largura_out = w_in
            altura_out = h_in
            # Alternativa: calcular caixa delimitadora
            # sin_t = abs(math.sin(theta))
            # cos_t = abs(math.cos(theta))
            # largura_out = int(h_in * sin_t + w_in * cos_t)
            # altura_out = int(h_in * cos_t + w_in * sin_t)

        escala_calc = None
        angulo_calc = theta
        nome_transformacao = f"rotacao_{args.angle}graus"

    # 3) Chamar função de interpolação escolhida
    if args.method == "nearest":
        img_result = vizinho_mais_proximo(img_input, escala_calc, angulo_calc, largura_out, altura_out)
    elif args.method == "bilinear":
        img_result = interp_bilinear(img_input, escala_calc, angulo_calc, largura_out, altura_out)
    elif args.method == "bicubic":
        img_result = interp_bicubica(img_input, escala_calc, angulo_calc, largura_out, altura_out)
    elif args.method == "lagrange":
        img_result = interp_lagrange(img_input, escala_calc, angulo_calc, largura_out, altura_out)

    # 4) Salvar resultado
    nome_arquivo = f"{args.method}_{nome_transformacao}.png"
    caminho_saida = os.path.join(args.output, nome_arquivo)
    cv2.imwrite(caminho_saida, img_result)
    print(f"Imagem salva em: {caminho_saida}")


def main():
    parser = argparse.ArgumentParser(description="Transformações Geométricas (Escala/Rotação).")
    parser.add_argument("-i", "--input", required=True, help="Arquivo PNG de entrada")
    parser.add_argument("-o", "--output", default="../output/", help="Pasta de output")
    parser.add_argument("-m", "--method", required=True,
                        choices=["nearest", "bilinear", "bicubic", "lagrange"],
                        help="Método de interpolação: nearest, bilinear, bicubic, lagrange")
    parser.add_argument("-e", "--escala", type=float, default=None,
                        help="Fator de escala (float). Se definido, aplica só ESCALA.")
    parser.add_argument("-a", "--angle", type=float, default=None,
                        help="Ângulo em graus anti-horário (float). Se definido, aplica só ROTAÇÃO.")
    parser.add_argument("-d", "--dim", nargs=2, type=int, metavar=("LARG", "ALT"),
                        help="Dimensões da imagem de saída (largura altura). Se omitido, calcular automaticamente.")
    args = parser.parse_args()

    if args.escala is None and args.angle is None:
        print("Defina um fator de escala (-e) ou um ângulo de rotação (-a).")
        exit(1)
    if args.escala is not None and args.angle is not None:
        print("Forneça apenas um: escala (-e) OU ângulo (-a).")
        exit(1)

    os.makedirs(args.output, exist_ok=True)
    aplicar_transformacao(args)


if __name__ == "__main__":
    main()
