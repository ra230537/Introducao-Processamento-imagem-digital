import os
import argparse
import math
import numpy as np
import cv2


def nearest_neighbor(img: np.ndarray, x: float, y: float):
    xi = int(round(x))
    yi = int(round(y))

    h, w = img.shape

    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))

    return float(img[yi, xi])


def bilinear(img: np.ndarray, x: float, y: float):
    # Coordenadas do pixel superior-esquerdo
    x0, y0 = math.floor(x), math.floor(y)

    # Frações que representam a distância do pixel atual para o pixel vizinho mais próximo.
    dx, dy = x - x0, y - y0

    # Clamping, w-2 e h-2 assegura que sempre haverá pixels vizinhos.
    h, w = img.shape
    x0_clamped = max(0, min(w - 2, x0))
    y0_clamped = max(0, min(h - 2, y0))

    # Obtém os valores dos pixels vizinhos mais próximos.
    i_00 = img[y0_clamped, x0_clamped]  # Superior-esquerdo
    i_10 = img[y0_clamped, x0_clamped + 1]  # Superior-direito
    i_01 = img[y0_clamped + 1, x0_clamped]  # Inferior-esquerdo
    i_11 = img[y0_clamped + 1, x0_clamped + 1]  # Inferior-direito

    # Calcula o valor interpolado do pixel atual.
    i_0 = i_00 * (1 - dx) + i_10 * dx
    i_1 = i_01 * (1 - dx) + i_11 * dx
    return float(i_0 * (1 - dy) + i_1 * dy)


def _p(t: float):
    return max(t, 0.0)


def _r(s: float):
    """
    Kernel R(s) = (1/6)[P(s+2)^3 - 4P(s+1)^3 + 6P(s)^3 - 4P(s-1)^3].
    B-spline cúbica.
    """
    return (1.0 / 6.0) * (
            _p(s + 2) ** 3
            - 4 * _p(s + 1) ** 3
            + 6 * _p(s) ** 3
            - 4 * _p(s - 1) ** 3
    )


def bicubic(img: np.ndarray, x: float, y: float):
    h, w = img.shape
    x0 = math.floor(x)
    y0 = math.floor(y)
    dx = x - x0
    dy = y - y0

    valor = 0.0
    # varre vizinhança 4×4: m,n E {-1,0,1,2}
    for m in range(-1, 3):
        for n in range(-1, 3):
            xm = x0 + m
            yn = y0 + n
            # clamp nos limites da imagem
            xm = min(max(xm, 0), w - 1)
            yn = min(max(yn, 0), h - 1)
            w_m = _r(m - dx)
            w_n = _r(n - dy)
            valor += img[yn, xm] * (w_m * w_n)

    return valor


def lagrange(img: np.ndarray, x: float, y: float):
    h, w = img.shape
    x0 = math.floor(x)
    y0 = math.floor(y)
    dx = x - x0
    dy = y - y0

    def sample(xx: int, yy: int):
        """Clampa e retorna img[yy,xx]."""
        xx = min(max(xx, 0), w - 1)
        yy = min(max(yy, 0), h - 1)
        return img[yy, xx]

    def _l(n: int):
        """
        Polinômio L(n), n=1..4, conforme slide:
        """
        yy = y0 + (n - 2)
        f_m1 = sample(x0 - 1, yy)
        f_0 = sample(x0, yy)
        f_p1 = sample(x0 + 1, yy)
        f_p2 = sample(x0 + 2, yy)
        # coeficientes em x:
        t1 = -dx * (dx - 1) * (dx - 2) / 6 * f_m1
        t2 = (dx + 1) * (dx - 1) * (dx - 2) / 2 * f_0
        t3 = -(dx + 1) * dx * (dx - 2) / 2 * f_p1
        t4 = (dx + 1) * dx * (dx - 1) / 6 * f_p2
        return t1 + t2 + t3 + t4

    # coeficientes em y:
    c1 = -dy * (dy - 1) * (dy - 2) / 6
    c2 = (dy + 1) * (dy - 1) * (dy - 2) / 2
    c3 = -(dy + 1) * dy * (dy - 2) / 2
    c4 = (dy + 1) * dy * (dy - 1) / 6

    return c1 * _l(1) + c2 * _l(2) + c3 * _l(3) + c4 * _l(4)


def interpolate(img: np.ndarray, x: float, y: float, method: str = 'bilinear'):
    h, w = img.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0.0
    if method == 'nearest':
        return nearest_neighbor(img, x, y)
    elif method == 'bilinear':
        return bilinear(img, x, y)
    elif method == 'bicubic':
        return bicubic(img, x, y)
    else:
        return lagrange(img, x, y)


def scale_image(img: np.ndarray, scale_x: float, scale_y: float, method: str):
    h_in, w_in = img.shape
    h_out, w_out = int(h_in * scale_y), int(w_in * scale_x)
    output = np.zeros((h_out, w_out), dtype=img.dtype)
    for y_out in range(h_out):
        for x_out in range(w_out):
            x_in = x_out / scale_x
            y_in = y_out / scale_y
            output[y_out, x_out] = interpolate(img, x_in, y_in, method)
    return output


def rotate_image(img: np.ndarray, angle_deg: float, method: str):
    h, w = img.shape
    center_x, center_y = w / 2.0, h / 2.0
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    output = np.zeros_like(img)
    for y_out in range(h):
        for x_out in range(w):
            # obtém as distâncias ao centro
            center_distance_x = x_out - center_x
            center_distance_y = y_out - center_y
            # Usando a Matriz de rotação
            x_rot = center_distance_x * cos_t - center_distance_y * sin_t + center_x
            y_rot = center_distance_x * sin_t + center_distance_y * cos_t + center_y
            output[y_out, x_out] = interpolate(img, x_rot, y_rot, method)
    return output


def apply_transformation(args):
    # Carrega como grayscale
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    h_in, w_in = img.shape

    # Determina tipo e parâmetros de saída
    if args.escala is not None:
        # Escala uniforme ou dimensões fixas
        if args.dim:
            w_out, h_out = args.dim
            scale_x = w_out / w_in
            scale_y = h_out / h_in
        else:
            scale_x = scale_y = args.escala
        result = scale_image(img, scale_x, scale_y, args.method)
        tag = f"scale_{scale_x:.2f}x{scale_y:.2f}"
    else:
        # Rotação
        result = rotate_image(img, args.angle, args.method)
        tag = f"rotate_{int(args.angle)}deg"

    # Prepara saída
    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    filename = f"{base}_{tag}_{args.method}.png"
    path_out = os.path.join(args.output, filename)
    cv2.imwrite(path_out, result)


def main():
    parser = argparse.ArgumentParser(
        description="Transformações Geométricas"
    )
    parser.add_argument("-i", "--input", required=True, help="PNG de entrada")
    parser.add_argument("-o", "--output", help="Pasta de saída")
    parser.add_argument(
        "-m", "--method", required=True,
        choices=["nearest", "bilinear", "bicubic", "lagrange"],
        help="Método de interpolação"
    )
    parser.add_argument(
        "-e", "--escala", type=float, default=None,
        help="Fator de escala"
    )
    parser.add_argument(
        "-a", "--angle", type=float, default=None,
        help="Ângulo de rotação"
    )
    parser.add_argument(
        "-d", "--dim", nargs=2, type=int, metavar=("W", "H"),
        help="Dimensões de saída"
    )
    args = parser.parse_args()

    apply_transformation(args)


if __name__ == "__main__":
    main()
