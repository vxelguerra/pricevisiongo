# PriceVisionGo

Aplicación educativa en Streamlit que procesa capturas PNG/JPG con OpenCV, detecta velas/pivotes/niveles/patrones, anota la imagen y permite descargar el resultado.

## Ejecutar en Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Para habilitar lectura visual del eje y redacción de la interpretación, define una clave de API en el entorno antes de iniciar Streamlit:

```powershell
$env:OPENAI_API_KEY = "tu-clave"
# Opcional: el valor predeterminado es gpt-4o-mini
$env:OPENAI_MODEL = "gpt-4o-mini"
streamlit run app.py
```

La lectura del eje se solicita con un botón y solo acepta etiquetas numéricas que el modelo reporte como claramente visibles. Si no obtiene dos etiquetas, o un nivel queda fuera del intervalo calibrado, se muestra en píxeles. La interpretación recibe únicamente resultados calculados, no la imagen.

## Controles

- Selecciona **Auto**, **Verde / rojo** o **Blanco / negro** si la segmentación automática no se ajusta a la captura.
- Ajusta sensibilidad de pivotes, altura mínima, tolerancia de agrupación y unión de segmentos desde la barra lateral.
- Activa **Modo debug** para ver los extremos/cuerpos aproximados y la máscara usada por OpenCV.
- Descarga el gráfico anotado como PNG.

## Alcance

La detección se realiza sobre píxeles de la captura y puede verse afectada por cuadrículas, indicadores, compresión, fondos o temas de color. La abertura y cierre se estiman a partir del cuerpo segmentado y del color/orientación convencional; revisa el modo debug antes de interpretar el resultado. Los patrones y niveles son heurísticos para uso académico, no asesoría financiera.

## Colores del gráfico

Las anotaciones usan colores BGR definidos como constantes al inicio de `app.py`, con un contorno oscuro fino para conservar contraste sobre temas claros y oscuros:

- Soportes: verde brillante; resistencias: rojo brillante.
- Líneas que unen pivotes (tendencia): cian eléctrico.
- Pivotes de máximos (HH/LH): verde; pivotes de mínimos (HL/LL): naranja.
- Patrones alcistas: amarillo; bajistas: magenta; patrones sin dirección detectada: cian. Cada patrón se marca con caja y etiqueta.

La leyenda visible bajo el gráfico resume los colores.

## Sesgo técnico por reglas

La sección **Sesgo técnico**, situada después del diagnóstico, no usa IA para calcular el resultado. El puntaje parte de cero y suma/resta estos pesos constantes, configurables al inicio de `app.py`:

- Tendencia alcista/bajista: +2/-2.
- Estructura con al menos dos marcas recientes HH/HL o LH/LL: +2/-2, si la secuencia reciente es coherente.
- Cierre a no más de dos rangos medianos de vela de un soporte/resistencia detectado: +1/-1 por cada tipo de nivel cercano.
- Cada patrón de vela direccional entre las últimas cinco velas: +1 alcista o -1 bajista.

Los rangos, niveles y distancias se miden en píxeles; el diagnóstico solo expresa precio si existe calibración del eje. Puntaje de +2 o más significa **Sesgo favorable a compra**; -2 o menos, **Sesgo desfavorable a compra**; entre ambos, **Sesgo neutral / esperar confirmación**. La confianza baja cuando las señales se contradicen, hay menos de dos señales o se detectaron menos de dos patrones recientes; en los demás casos es media y puede ser alta con cuatro o más señales alineadas. La vista lista cada aporte y su evidencia, además del nivel de invalidación y un escenario alternativo.

Con `OPENAI_API_KEY`, un botón opcional pide al modelo redactar un párrafo a partir exclusivamente del puntaje, señales y niveles ya calculados. Sin clave se conserva toda la lectura por reglas y se omite solo ese párrafo. La sección siempre termina con el aviso educativo.

## Apariencia

La interfaz usa un tema oscuro azul marino con acentos cian, verde y naranja, definido en `.streamlit/config.toml` y complementado con estilos CSS en `app.py`. La portada presenta las etapas Detectar, Dibujar e Interpretar; al cargar una captura pasa a un encabezado compacto. El panel lateral agrupa los controles y el diagnóstico y el sesgo destacan sus estados con tarjetas de color.


## Marca y tema visual

La aplicaci?n se presenta como **PriceVisionGo** en la portada y en la pesta?a del navegador, con el subt?tulo ?Del gr?fico al an?lisis t?cnico automatizado?. El tema combina violeta, cian el?ctrico, magenta y verde lima sobre fondo oscuro, con degradados, efectos de brillo y estados interactivos. Los textos propios de la interfaz est?n en espa?ol; algunas etiquetas internas de Streamlit pueden depender de la versi?n y del navegador.
