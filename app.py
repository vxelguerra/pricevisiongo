"""Análisis técnico didáctico de capturas de gráficos OHLC."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Colores de anotación expresados en BGR para OpenCV. Contorno oscuro para contraste.
COLOR_OUTLINE = (32, 24, 35)
COLOR_SUPPORT = (0, 230, 80)       # verde brillante
COLOR_RESISTANCE = (30, 45, 255)    # rojo brillante
COLOR_TREND = (255, 220, 0)         # cian eléctrico
COLOR_HIGH = (0, 235, 90)          # HH/LH
COLOR_LOW = (0, 155, 255)           # HL/LL
COLOR_BULL_PATTERN = (0, 235, 255)  # amarillo
COLOR_BEAR_PATTERN = (255, 0, 220) # magenta
COLOR_DEFAULT_PATTERN = (255, 220, 0)

# Pesos simples del sesgo. Tendencia y estructura pesan más; cada zona cercana
# y patrón reciente aporta una señal. La proximidad se mide en píxeles del gráfico.
BIAS_TREND_WEIGHT = 2
BIAS_STRUCTURE_WEIGHT = 2
BIAS_LEVEL_WEIGHT = 1
BIAS_PATTERN_WEIGHT = 1
BIAS_RECENT_PATTERNS = 5
BIAS_NEAR_LEVEL_SPANS = 2.0
BIAS_BUY_THRESHOLD = 2
BIAS_SELL_THRESHOLD = -2


@dataclass
class Candle:
    x: int
    high: int
    low: int
    open: int
    close: int
    bullish: bool
    width: int


def _masks(rgb: np.ndarray, scheme: str) -> tuple[np.ndarray, np.ndarray, bool]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if scheme in ("Auto", "Auto (verde/rojo)"):
        saturated = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([180, 255, 255]))
        if cv2.countNonZero(saturated) > max(25, rgb.shape[0] * rgb.shape[1] * .0002):
            scheme = "Verde / rojo"
        else:
            scheme = "Blanco / negro"
    if scheme == "Verde / rojo":
        green = cv2.inRange(hsv, np.array([30, 35, 35]), np.array([100, 255, 255]))
        red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 45, 35]), np.array([12, 255, 255])),
                             cv2.inRange(hsv, np.array([168, 45, 35]), np.array([180, 255, 255])))
        return cv2.bitwise_or(green, red), green, False
    # Usa la polaridad de primer plano opuesta al fondo, evitando que el fondo
    # claro/oscuro conecte todos los candidatos en una sola región.
    value = hsv[:, :, 2]
    if float(np.median(value)) < 128:
        foreground = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 110, 255]))
        return foreground, foreground, True
    foreground = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 130, 105]))
    return foreground, foreground, True


def detect_candles(rgb: np.ndarray, scheme: str, min_height: int, gap: int):
    mask, up_mask, monochrome = _masks(rgb, scheme)
    h, w = mask.shape
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, min_height // 3))))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    parts = []
    for i in range(1, count):
        x, y, ww, hh, area = map(int, stats[i])
        if hh >= min_height and 0 < ww <= max(24, int(w * .025)) and hh < h * .9 and area >= 3:
            parts.append((x, y, ww, hh))
    parts.sort(key=lambda p: p[0] + p[2] / 2)
    groups = []
    for part in parts:
        if groups and part[0] <= max(q[0] + q[2] for q in groups[-1]) + gap:
            groups[-1].append(part)
        else:
            groups.append([part])
    candles = []
    for group in groups:
        x0, x1 = min(p[0] for p in group), max(p[0] + p[2] for p in group)
        y0, y1 = min(p[1] for p in group), max(p[1] + p[3] for p in group)
        ww = x1 - x0
        if ww > max(24, int(w * .025)) or y1 - y0 < min_height:
            continue
        roi = mask[y0:y1, x0:x1]
        rows = np.count_nonzero(roi, axis=1)
        peak = int(rows.max(initial=0))
        body = np.flatnonzero(rows >= max(2, peak * .62)) if peak else np.array([], dtype=int)
        if len(body):
            runs = np.split(body, np.where(np.diff(body) > 1)[0] + 1)
            run = max(runs, key=len)
            by0, by1 = int(run[0]), int(run[-1])
        else:
            by0, by1 = 0, max(0, y1 - y0 - 1)
        x = (x0 + x1 - 1) // 2
        if monochrome:
            bullish = float(rgb[min(h - 1, y0 + by0), x].mean()) > 127
        else:
            body_mask = roi[by0:by1 + 1]
            green_pixels = np.count_nonzero(up_mask[y0 + by0:y0 + by1 + 1, x0:x1])
            bullish = green_pixels >= max(1, np.count_nonzero(body_mask) / 2)
        body_top, body_bottom = y0 + by0, y0 + by1
        # Screen y increases downward: bullish candles close lower; bearish candles close higher.
        open_y, close_y = (body_top, body_bottom) if bullish else (body_bottom, body_top)
        candles.append(Candle(x, y0, y1 - 1, open_y, close_y, bullish, ww))
    return candles, joined


def detect_pivots(candles: list[Candle], sensitivity: int):
    pivots = []
    n = len(candles)
    for i in range(sensitivity, n - sensitivity):
        window = candles[i - sensitivity:i + sensitivity + 1]
        if candles[i].high == max(c.high for c in window) and sum(c.high == candles[i].high for c in window) == 1:
            pivots.append({"kind": "high", "i": i, "x": candles[i].x, "y": candles[i].high})
        if candles[i].low == min(c.low for c in window) and sum(c.low == candles[i].low for c in window) == 1:
            pivots.append({"kind": "low", "i": i, "x": candles[i].x, "y": candles[i].low})
    # En imagen, y menor equivale a precio mayor.
    for kind in ("high", "low"):
        seq = [p for p in pivots if p["kind"] == kind]
        for j, p in enumerate(seq):
            if not j:
                p["label"] = "H" if kind == "high" else "L"
            else:
                prev = seq[j - 1]
                p["label"] = (("HH" if p["y"] < prev["y"] else "LH") if kind == "high"
                              else ("HL" if p["y"] < prev["y"] else "LL"))
    return sorted(pivots, key=lambda p: p["x"])


def trend_of(pivots):
    highs = [p["y"] for p in pivots if p["kind"] == "high"]
    lows = [p["y"] for p in pivots if p["kind"] == "low"]
    if min(len(highs), len(lows)) < 2:
        return "indeterminada"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "alcista"
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "bajista"
    return "lateral"


def detect_zones(pivots, tolerance):
    zones = []
    for pivot in sorted(pivots, key=lambda p: p["y"]):
        if zones and abs(pivot["y"] - np.mean([p["y"] for p in zones[-1]])) <= tolerance:
            zones[-1].append(pivot)
        else:
            zones.append([pivot])
    return [{"y": int(round(np.mean([p["y"] for p in z]))),
             "type": "resistencia" if z[0]["kind"] == "high" else "soporte", "count": len(z)}
            for z in zones if len(z) >= 2]


def detect_patterns(candles):
    result = []
    for i, c in enumerate(candles):
        body = max(1, abs(c.close - c.open))
        span = max(1, c.low - c.high)
        upper = max(0, min(c.open, c.close) - c.high)
        lower = max(0, c.low - max(c.open, c.close))
        name = "Doji" if body / span <= .1 else "Hammer" if lower >= 2 * body and upper <= .7 * body else "Shooting Star" if upper >= 2 * body and lower <= .7 * body else None
        if name:
            result.append({"i": i, "x": c.x, "y": c.high, "name": name})
        if i:
            p = candles[i - 1]
            plo, phi = sorted((p.open, p.close)); clo, chi = sorted((c.open, c.close))
            if p.bullish != c.bullish and clo <= plo and chi >= phi:
                result.append({"i": i, "x": c.x, "y": c.high, "name": "Engulfing alcista" if c.bullish else "Engulfing bajista"})
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2:i + 1]
        ab, bb = abs(a.close - a.open), abs(b.close - b.open)
        if ab and bb < ab * .55 and not a.bullish and c.bullish:
            result.append({"i": i, "x": c.x, "y": c.high, "name": "Morning Star"})
        elif ab and bb < ab * .55 and a.bullish and not c.bullish:
            result.append({"i": i, "x": c.x, "y": c.high, "name": "Evening Star"})
    return result


def annotate(rgb, pivots, zones, patterns):
    image = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    for kind, color in (("high", COLOR_HIGH), ("low", COLOR_LOW)):
        points = [p for p in pivots if p["kind"] == kind]
        for a, b in zip(points, points[1:]):
            cv2.line(image, (a["x"], a["y"]), (b["x"], b["y"]), COLOR_OUTLINE, 4, cv2.LINE_AA)
            cv2.line(image, (a["x"], a["y"]), (b["x"], b["y"]), COLOR_TREND, 2, cv2.LINE_AA)
    for p in pivots:
        color = COLOR_HIGH if p["kind"] == "high" else COLOR_LOW
        cv2.circle(image, (p["x"], p["y"]), 7, COLOR_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(image, (p["x"], p["y"]), 5, color, -1)
        pos = (p["x"] + 3, max(14, p["y"] - 7))
        cv2.putText(image, p["label"], pos, cv2.FONT_HERSHEY_SIMPLEX, .48, COLOR_OUTLINE, 3, cv2.LINE_AA)
        cv2.putText(image, p["label"], pos, cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    for z in zones:
        color = COLOR_RESISTANCE if z["type"] == "resistencia" else COLOR_SUPPORT
        cv2.line(image, (0, z["y"]), (w - 1, z["y"]), COLOR_OUTLINE, 5, cv2.LINE_AA)
        cv2.line(image, (0, z["y"]), (w - 1, z["y"]), color, 2, cv2.LINE_AA)
        pos = (8, max(15, z["y"] - 5)); label = f'{z["type"].title()} ~{z["y"]} px'
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, .48, COLOR_OUTLINE, 3, cv2.LINE_AA)
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    for p in patterns:
        x, y = p["x"], p["y"]
        bullish = "alcista" in p["name"].lower() or p["name"] == "Morning Star"
        color = COLOR_BULL_PATTERN if bullish else COLOR_BEAR_PATTERN if "bajista" in p["name"].lower() or p["name"] == "Evening Star" else COLOR_DEFAULT_PATTERN
        cv2.rectangle(image, (max(0, x - 8), max(0, y - 5)), (min(w - 1, x + 8), min(h - 1, y + 22)), COLOR_OUTLINE, 4)
        cv2.rectangle(image, (max(0, x - 8), max(0, y - 5)), (min(w - 1, x + 8), min(h - 1, y + 22)), color, 2)
        pos = (min(w - 2, x + 10), max(16, y - 8))
        cv2.putText(image, p["name"], pos, cv2.FONT_HERSHEY_SIMPLEX, .4, COLOR_OUTLINE, 3, cv2.LINE_AA)
        cv2.putText(image, p["name"], pos, cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_bias(candles, pivots, zones, patterns, trend, level_fn):
    """Calcula puntaje auditable con y en pantalla (menor y = mayor precio)."""
    score, signals, evidence = 0, [], []
    def add(label, weight, detail):
        nonlocal score
        score += weight
        signals.append(1 if weight > 0 else -1)
        evidence.append({"label": label, "weight": weight, "detail": detail})
    if trend in ("alcista", "bajista"):
        weight = BIAS_TREND_WEIGHT if trend == "alcista" else -BIAS_TREND_WEIGHT
        add("Tendencia", weight, f"Tendencia {trend}, calculada con los dos últimos máximos y mínimos.")
    labels = [p["label"] for p in pivots if p["label"] in ("HH", "HL", "LH", "LL")]
    bullish_structure = [x for x in labels[-4:] if x in ("HH", "HL")]
    bearish_structure = [x for x in labels[-4:] if x in ("LH", "LL")]
    if len(bullish_structure) >= 2 and not bearish_structure:
        add("Estructura", BIAS_STRUCTURE_WEIGHT, "Secuencia reciente " + "/".join(bullish_structure) + ".")
    elif len(bearish_structure) >= 2 and not bullish_structure:
        add("Estructura", -BIAS_STRUCTURE_WEIGHT, "Secuencia reciente " + "/".join(bearish_structure) + ".")
    last = candles[-1]
    span = float(np.median([max(1, c.low-c.high) for c in candles[-10:]]))
    for ztype, sign in (("soporte", 1), ("resistencia", -1)):
        candidates = [z for z in zones if z["type"] == ztype]
        if candidates:
            z = min(candidates, key=lambda q: abs(q["y"] - last.close))
            distance = abs(z["y"] - last.close)
            if distance <= span * BIAS_NEAR_LEVEL_SPANS:
                add("Nivel cercano", sign * BIAS_LEVEL_WEIGHT,
                    f"Cierre de la última vela a {distance:.0f} px de {ztype} en {level_fn(z)}.")
    recent = [p for p in patterns if p["i"] >= max(0, len(candles)-BIAS_RECENT_PATTERNS)]
    for p in recent:
        name = p["name"]
        if "alcista" in name.lower() or name == "Morning Star":
            add("Patrón", BIAS_PATTERN_WEIGHT, f"{name} en vela {p['i']+1} (x={p['x']} px).")
        elif "bajista" in name.lower() or name == "Evening Star":
            add("Patrón", -BIAS_PATTERN_WEIGHT, f"{name} en vela {p['i']+1} (x={p['x']} px).")
    label = ("Sesgo favorable a compra" if score >= BIAS_BUY_THRESHOLD else
             "Sesgo desfavorable a compra" if score <= BIAS_SELL_THRESHOLD else
             "Sesgo neutral / esperar confirmación")
    positives, negatives = signals.count(1), signals.count(-1)
    conflict = positives > 0 and negatives > 0
    n = len(signals)
    confidence = "baja" if conflict or n < 2 or len(recent) < 2 else "alta" if n >= 4 and min(positives, negatives) == 0 else "media"
    if label == "Sesgo favorable a compra":
        candidates = [z for z in zones if z["type"] == "soporte"]
        invalidation = min(candidates, key=lambda z: abs(z["y"]-last.close)) if candidates else None
        invalidation_text = f"Ruptura del soporte más cercano ({level_fn(invalidation)})." if invalidation else "No se detectó soporte para fijar invalidación."
        alternative = "Si el soporte cercano cede, el escenario favorable pierde validez y conviene reevaluar la estructura."
    else:
        candidates = [z for z in zones if z["type"] == "resistencia"]
        invalidation = min(candidates, key=lambda z: abs(z["y"]-last.close)) if candidates else None
        invalidation_text = f"Ruptura de la resistencia más cercana ({level_fn(invalidation)})." if invalidation else "No se detectó resistencia para fijar invalidación."
        alternative = "Si la resistencia cercana se rompe, el escenario desfavorable pierde validez y conviene reevaluar la estructura."
    return {"score": score, "label": label, "confidence": confidence, "evidence": evidence,
            "invalidation": invalidation_text, "alternative": alternative, "signal_count": n,
            "recent_pattern_count": len(recent)}


def vision_request(image_bytes: bytes, mime: str, prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    data = base64.b64encode(image_bytes).decode("ascii")
    response = client.responses.create(model=model, input=[{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": "high"}]}])
    return response.output_text


def text_request(prompt: str, model: str) -> str:
    """Text-only request: calculated detections are the sole input to interpretation."""
    from openai import OpenAI
    response = OpenAI().responses.create(model=model, input=prompt)
    return response.output_text


def read_axis(image_bytes, mime, model):
    prompt = ('Lee solo 2 o 3 etiquetas numéricas claramente legibles del eje vertical. '
              'Devuelve exclusivamente JSON {"labels":[{"value":numero,"y":coordenada_y_en_pixeles_desde_arriba}]}. '
              'Si no hay al menos dos inequívocas, devuelve {"labels":[]}. No infieras ni inventes precios.')
    raw = vision_request(image_bytes, mime, prompt, model)
    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return [p for p in data.get("labels", [])[:3]
                if isinstance(p.get("value"), (int, float)) and isinstance(p.get("y"), (int, float))]
    except (ValueError, TypeError, AttributeError):
        return []


def price_at(y, labels):
    labels = sorted(labels, key=lambda p: p["y"])
    for a, b in zip(labels, labels[1:]):
        if a["y"] <= y <= b["y"] and b["y"] != a["y"]:
            return a["value"] + (y - a["y"]) * (b["value"] - a["value"]) / (b["y"] - a["y"])
    return None


st.set_page_config(page_title="PriceVisionGo", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at 5% 0%,#321061 0,#101024 42%,#080b15 100%);color:#f3f4ff}
[data-testid="stHeader"]{background:rgba(9,8,22,.72)} [data-testid="stSidebar"]{background:linear-gradient(165deg,#21103b,#111426 65%,#0d1722);border-right:1px solid #6732a3}
[data-testid="stSidebar"] *{color:#f0eaff!important} [data-testid="stSidebar"] label{color:#fff!important;font-weight:600}
.block-container{max-width:1420px;padding-top:2.2rem}
.hero{padding:2rem 2.4rem;margin:.4rem 0 1rem;border:1px solid #a247ff;border-radius:22px;background:linear-gradient(110deg,#301064,#142d59,#092f3d,#301064);background-size:220% 220%;animation:heroShift 14s ease-in-out infinite;box-shadow:0 0 30px #a32cff35,0 20px 60px #02081166}
@keyframes heroShift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.hero h1{font-size:2.65rem;letter-spacing:-.04em;margin:0 0 .45rem;color:#fff}.hero p{font-size:1.08rem;color:#e4ddff;margin:0}
.feature{height:100%;padding:1rem 1.2rem;border-radius:16px;background:linear-gradient(145deg,#1a1436,#101e30);border:1px solid #583a83;transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease}.feature:hover{transform:translateY(-7px);border-color:#e64cff;box-shadow:0 0 26px #c736ff70,0 14px 32px #03040b99}.feature strong{display:block;color:#36e9ff;letter-spacing:.12em;font-size:.78rem;margin:.45rem 0}.feature span{font-size:.93rem;color:#eee9ff}
.hero-compact{padding:.8rem 1.2rem;margin:.3rem 0 .8rem;border-left:4px solid #35efff;border-radius:12px;background:linear-gradient(100deg,#321064,#122e4b,#251044);box-shadow:0 0 18px #ad35ff32}.hero-compact h1{font-size:1.5rem;margin:0;color:#fff}.hero-compact p{margin:.15rem 0 0;color:#e4ddff}
.diag-card{min-height:100px;padding:1rem;border-radius:15px;border:1px solid #51436b;background:#14172a;margin-bottom:.6rem;box-shadow:0 5px 20px #02030a77}.diag-card small{display:block;font-weight:700;letter-spacing:.09em;color:#eee8ff;margin-bottom:.5rem}.diag-card b{font-size:1.05rem;color:#fff}
.bias-card{padding:1.15rem 1.35rem;margin:.5rem 0 1rem;border-radius:16px;border:1px solid #65458b;background:linear-gradient(120deg,#17152b,#211536);box-shadow:0 0 20px #ad35ff2b}
.stButton>button,.stDownloadButton>button{background:linear-gradient(105deg,#bd28ef,#296dff,#00cde8);background-size:200% 100%;color:#fff;border:1px solid #c85bff;border-radius:10px;font-weight:750;transition:all .2s ease;box-shadow:0 0 12px #9d38ff45}.stButton>button:hover,.stDownloadButton>button:hover{background-position:100% 0;transform:translateY(-2px);box-shadow:0 0 22px #36eaff99;border-color:#5df5ff;color:#fff}
input,textarea,[data-baseweb="select"]>div{background-color:#19172b!important;color:#f5f2ff!important;border-color:#70548d!important}[data-testid="stSlider"] [role="slider"]{background:#b7ff38;box-shadow:0 0 12px #b7ff38aa} [data-testid="stSlider"] [data-testid="stMarkdownContainer"]{color:#f7f2ff} div[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#2deaff,#bc42ff,#b7ff38)}
[data-testid="stFileUploader"]{padding:1rem;border:1px dashed #e241ff;border-radius:14px;background:linear-gradient(145deg,#17162b,#101b2b);transition:box-shadow .2s ease,border-color .2s ease}[data-testid="stFileUploader"]:hover{border-color:#42f3ff;box-shadow:0 0 24px #d23bff66}
[data-testid="stFileUploaderDropzone"]{background:#16162a!important}[data-testid="stFileUploaderDropzone"] button{background:linear-gradient(100deg,#9c27ef,#126eea)!important;color:#fff!important;border:1px solid #db68ff!important}
[data-testid="stFileUploaderDropzone"] button p{font-size:0!important}[data-testid="stFileUploaderDropzone"] button p:after{content:"Elegir archivo";font-size:.9rem;color:#fff}
[data-testid="stFileUploaderDropzone"] small{font-size:0!important}[data-testid="stFileUploaderDropzone"] small:after{content:"L?mite: 200 MB por archivo";font-size:.8rem;color:#ded5f5}
[data-testid="stDeployButton"]{display:none!important}
</style>""", unsafe_allow_html=True)
compact_header = bool(st.session_state.get("capture_uploader"))
if compact_header:
    st.markdown('<div class="hero-compact"><h1>PriceVisionGo</h1><p>Del gráfico al análisis técnico automatizado</p></div>', unsafe_allow_html=True)
else:
    st.markdown('<section class="hero"><h1>PriceVisionGo</h1><p>Del gráfico al análisis técnico automatizado</p></section>', unsafe_allow_html=True)
    feature_cols = st.columns(3)
    for col, (icon, title, description) in zip(feature_cols, [("🔎", "DETECTAR", "Identifica velas, pivotes y niveles clave."), ("✏️", "DIBUJAR", "Anota estructuras y patrones sobre el gráfico."), ("✨", "INTERPRETAR", "Resume señales y sesgo con reglas e IA.")]):
        with col:
            st.markdown(f'<div class="feature"><span style="font-size:1.5rem">{icon}</span><strong>{title}</strong><span>{description}</span></div>', unsafe_allow_html=True)
    st.markdown("**Cómo empezar**　 ① Sube una captura　 →　 ② Ajusta parámetros　 →　 ③ Revisa el resultado")
with st.sidebar:
    st.title("Panel de control")
    st.caption("Ajusta la detección y la interpretación")
    st.subheader("1 · Detección de velas")
    scheme = st.selectbox("Esquema de colores", ["Auto", "Verde / rojo", "Blanco / negro"])
    st.subheader("2 · Parámetros del gráfico")
    sensitivity = st.slider("Sensibilidad de pivotes", 1, 8, 2)
    min_height = st.slider("Altura mínima de vela (px)", 5, 40, 10)
    tolerance = st.slider("Agrupación de niveles (px)", 2, 40, 12)
    gap = st.slider("Unión de segmentos (px)", 0, 5, 1)
    debug = st.checkbox("Modo debug: mostrar velas segmentadas", False)
    st.subheader("3 · Interpretación")
    model = st.text_input("Modelo de visión", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

uploaded = st.file_uploader("Sube una captura PNG o JPG", type=["png", "jpg", "jpeg"], key="capture_uploader")
if uploaded:
    raw, mime = uploaded.getvalue(), uploaded.type or "image/png"
    image_key = hashlib.sha256(raw).hexdigest()
    if st.session_state.get("calibration_image") != image_key:
        st.session_state["calibration_image"] = image_key
        st.session_state["calibration"] = []
    rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    candles, mask = detect_candles(rgb, scheme, min_height, gap)
    if not candles:
        st.error("No se detectaron velas. Cambia el esquema o reduce la altura mínima.")
        if debug:
            st.image(mask, caption="Máscara de segmentación", clamp=True)
        st.stop()
    pivots = detect_pivots(candles, sensitivity)
    zones = detect_zones(pivots, tolerance)
    patterns = detect_patterns(candles)
    annotated = annotate(rgb, pivots, zones, patterns)
    st.image(annotated, caption="Análisis sobre la imagen original", use_container_width=True)
    st.markdown("**Leyenda:** 🟢 Soporte · 🔴 Resistencia · 🔵 Líneas de tendencia · 🟢 HH/LH · 🟠 HL/LL · 🟡 Patrones alcistas · 🟣 Patrones bajistas o neutros")
    buffer = io.BytesIO(); Image.fromarray(annotated).save(buffer, format="PNG")
    st.download_button("Descargar análisis PNG", buffer.getvalue(), "analisis_tecnico.png", "image/png")
    if debug:
        debug_image = rgb.copy()
        for c in candles:
            cv2.line(debug_image, (c.x, c.high), (c.x, c.low), (0, 255, 255), 1)
            cv2.rectangle(debug_image, (c.x - max(1, c.width // 2), c.open), (c.x + max(1, c.width // 2), c.close), (255, 0, 255), 1)
        st.image(debug_image, caption=f"Debug: {len(candles)} velas. Cian=extremos; magenta=cuerpo aproximado", use_container_width=True)
        st.image(mask, caption="Máscara morfológica", clamp=True)

    trend = trend_of(pivots)
    supports = [z for z in zones if z["type"] == "soporte"]
    resistances = [z for z in zones if z["type"] == "resistencia"]
    st.subheader("Calibración del eje de precios")
    calibration = st.session_state.get("calibration", [])
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.info("Sin OPENAI_API_KEY: se muestran niveles en píxeles y no se inventan precios.")
    elif st.button("Leer etiquetas del eje con visión"):
        try:
            with st.spinner("Leyendo etiquetas del eje…"):
                calibration = read_axis(raw, mime, model)
                st.session_state["calibration"] = calibration
            if calibration:
                st.success("Etiquetas leídas; solo se interpolarán niveles entre etiquetas válidas.")
            else:
                st.warning("No se pudieron leer al menos dos etiquetas inequívocas. Se mantienen niveles en píxeles.")
        except Exception as exc:
            st.warning(f"No se pudo consultar el modelo: {exc}")
    def level(zone):
        price = price_at(zone["y"], calibration)
        return f"{price:.5g}" if price is not None else f'{zone["y"]} px'
    structure = ", ".join(p["label"] for p in pivots if p["label"] in ("HH", "HL", "LH", "LL")) or "sin secuencia suficiente"
    support_text = ", ".join(level(z) for z in supports) or "no detectado"
    resistance_text = ", ".join(level(z) for z in resistances) or "no detectada"
    pattern_text = ", ".join(dict.fromkeys(p["name"] for p in patterns)) or "no detectado"
    calculated = {"TENDENCIA": trend, "ESTRUCTURA": structure, "SOPORTE": support_text,
                  "RESISTENCIA": resistance_text, "PATRÓN": pattern_text,
                  "velas_detectadas": len(candles), "pivotes": len(pivots)}
    st.subheader("Diagnóstico")
    diag_color = {"alcista": "#147d58", "bajista": "#a43d4b", "lateral": "#9a6a13"}.get(trend, "#35516a")
    diag_cols = st.columns(5)
    for col, key in zip(diag_cols, ("TENDENCIA", "ESTRUCTURA", "SOPORTE", "RESISTENCIA", "PATRÓN")):
        with col:
            card_color = diag_color
            st.markdown(f'<div class="diag-card" style="border-top:3px solid {diag_color};background:{card_color}"><small>{key}</small><b>{calculated[key]}</b></div>', unsafe_allow_html=True)
    st.subheader("Sesgo técnico")
    bias = calculate_bias(candles, pivots, zones, patterns, trend, level)
    bias_data = {k: bias[k] for k in ("score", "label", "confidence", "evidence", "invalidation", "alternative", "signal_count", "recent_pattern_count")}
    bias_color = "#29c982" if bias["score"] >= BIAS_BUY_THRESHOLD else "#ff6674" if bias["score"] <= BIAS_SELL_THRESHOLD else "#ffc04d"
    st.markdown(f'<div class="bias-card" style="border-left:5px solid {bias_color}"><b style="font-size:1.2rem;color:{bias_color}">{bias["label"]}</b><br><span>Confianza: <b>{bias["confidence"]}</b>　·　Puntaje: <b>{bias["score"]:+d}</b></span></div>', unsafe_allow_html=True)
    confidence_progress = {"baja": .33, "media": .66, "alta": 1.0}[bias["confidence"]]
    st.progress(confidence_progress, text=f"Confianza {bias['confidence']}")
    st.markdown("**Motivos**")
    if bias["evidence"]:
        for item in bias["evidence"]:
            direction = "suma" if item["weight"] > 0 else "resta"
            st.markdown(f"- {item['detail']} ({direction} {abs(item['weight'])} punto{'s' if abs(item['weight']) != 1 else ''})")
    else:
        st.write("No hay señales direccionales suficientes para sumar o restar.")
    st.markdown(f"**Invalidación:** {bias['invalidation']}  \n**Escenario alternativo:** {bias['alternative']}")
    if api_key and st.button("Redactar explicación del sesgo con IA"):
        prompt = ("Redacta un único párrafo breve en español usando exclusivamente este JSON calculado. "
                  "No agregues datos, señales ni niveles, no infieras precios ausentes, no prometas resultados y reconoce la incertidumbre. Datos: "
                  + json.dumps(bias_data, ensure_ascii=False))
        try:
            with st.spinner("Redactando explicación…"):
                st.write(text_request(prompt, model))
        except Exception as exc:
            st.warning(f"No se pudo generar la explicación: {exc}")
    st.caption("Lectura técnica educativa basada solo en la imagen. No es una recomendación de inversión.")
    if st.button("Generar interpretación con IA", disabled=not bool(api_key)):
        prompt = ("Redacta un párrafo breve en español basándote únicamente en estos resultados calculados. "
                  "No inventes datos del gráfico, no presentes predicciones como certezas y reconoce la incertidumbre "
                  "del análisis automático. No conviertas píxeles en precios. Resultados: " + json.dumps(calculated, ensure_ascii=False))
        try:
            with st.spinner("Redactando interpretación…"):
                st.write(text_request(prompt, model))
        except Exception as exc:
            st.warning(f"No se pudo generar la interpretación: {exc}")
