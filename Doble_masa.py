import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Doble Masa - By: AMAYA17")

# --- FUNCIONES DE CÁLCULO (EL MOTOR) ---

def calcular_doble_masa(df, cols_estaciones):
    """Prepara el DataFrame para los gráficos de doble masa."""
    df_acum = df.copy()
    for col in cols_estaciones:
        df_acum[f'P_acum_{col}'] = df_acum[col].cumsum()
    df_acum['P_media_anual'] = df_acum[cols_estaciones].mean(axis=1)
    df_acum['P_media_acum'] = df_acum['P_media_anual'].cumsum()
    return df_acum

def encontrar_patron_auto(df_acum, cols_estaciones, col_media_acum):
    """Encuentra la estación patrón calculando el R^2 de cada una."""
    r_squares = {}
    for col in cols_estaciones:
        col_acum = f'P_acum_{col}'
        temp_df = df_acum[[col_acum, col_media_acum]].dropna()
        
        if not temp_df.empty and len(temp_df) > 1:
            temp_df[col_media_acum] = pd.to_numeric(temp_df[col_media_acum], errors='coerce')
            temp_df[col_acum] = pd.to_numeric(temp_df[col_acum], errors='coerce')
            temp_df = temp_df.dropna()

            if len(temp_df) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    temp_df[col_media_acum], 
                    temp_df[col_acum]
                )
                r_squares[col] = r_value**2
            else:
                 r_squares[col] = 0
        else:
            r_squares[col] = 0
            
    if not r_squares:
        return None, {}
        
    valid_r_squares = {k: v for k, v in r_squares.items() if pd.notnull(v) and np.isfinite(v)}
    
    if not valid_r_squares:
        return cols_estaciones[0] if cols_estaciones else None, r_squares

    patron_recomendada = max(valid_r_squares, key=valid_r_squares.get)
    return patron_recomendada, r_squares

def calcular_estadisticas_quiebre(df_original, col_evaluar, ano_quiebre, col_ano):
    """Calcula T-Student (Pooled) y F-Fisher."""
    datos_antes = df_original[df_original[col_ano] < ano_quiebre][col_evaluar].dropna()
    datos_despues_full = df_original[df_original[col_ano] >= ano_quiebre][col_evaluar].dropna()
    
    n1 = len(datos_antes)
    n2_full = len(datos_despues_full)
    
    if n1 < 2:
        return None, None, None, f"Datos insuficientes para el Grupo 1 (Antes de {ano_quiebre}). Se necesitan al menos 2."
    
    if n2_full >= n1:
        datos_despues = datos_despues_full.iloc[:n1]
        n2 = n1
    else:
        datos_despues = datos_despues_full
        n2 = n2_full

    if n2 < 2:
         return None, None, None, f"Datos insuficientes para el Grupo 2 (Después de {ano_quiebre}). Se necesitan al menos 2."

    media1 = datos_antes.mean()
    media2 = datos_despues.mean()
    var1 = datos_antes.var(ddof=1)
    var2 = datos_despues.var(ddof=1)
    
    if var1 is None or var2 is None or pd.isna(var1) or pd.isna(var2) or var1 == 0 or var2 == 0:
        return None, None, None, "Cálculo fallido: Varianza de uno de los grupos es 0 o nula."

    gl_t = n1 + n2 - 2
    if gl_t <= 0:
        return None, None, None, "Cálculo fallido: Grados de libertad T-Student <= 0."

    sp_cuadrada = ((n1 - 1) * var1 + (n2 - 1) * var2) / gl_t
    sp = np.sqrt(sp_cuadrada)
    sd = sp * np.sqrt(1/n1 + 1/n2)
    
    if sd == 0:
        tc = np.inf
    else:
        tc = np.abs(media1 - media2) / sd
    
    alfa_t = 0.025
    tt_tabla = stats.t.ppf(1 - alfa_t, gl_t)

    gl_num_f1 = n1 - 1
    gl_den_f1 = n2 - 1
    gl_num_f2 = n2 - 1
    gl_den_f2 = n1 - 1

    if gl_num_f1 <= 0 or gl_den_f1 <= 0 or gl_num_f2 <= 0 or gl_den_f2 <= 0:
        return None, None, None, "Cálculo fallido: Grados de libertad F-Fisher <= 0."

    if var1 > var2:
        fc = var1 / var2
        gl_num = gl_num_f1
        gl_den = gl_den_f1
    else:
        fc = var2 / var1
        gl_num = gl_num_f2
        gl_den = gl_den_f2
        
    alfa_f = 0.05
    ft_tabla = stats.f.ppf(1 - alfa_f, gl_num, gl_den)

    stats_results = {
        "n1": n1,
        "n2": n2,
        "media_antes": media1,
        "media_despues": media2,
        "var_antes": var1,
        "var_despues": var2,
        "sp": sp,
        "sd": sd,
        "t_stat": tc,
        "tt_tabla": tt_tabla,
        "f_stat": fc,
        "ft_tabla": ft_tabla
    }
    return stats_results, datos_antes, datos_despues, None


def corregir_datos(datos_dudosos, datos_confiables):
    """Aplica la fórmula de corrección."""
    if datos_dudosos.empty or len(datos_dudosos) < 2:
        st.error("Error: El grupo 'dudoso' está vacío o tiene menos de 2 datos.")
        return None
    if datos_confiables.empty or len(datos_confiables) < 2:
        st.error("Error: El grupo 'confiable' está vacío o tiene menos de 2 datos.")
        return None

    media_dudosa = datos_dudosos.mean()
    std_dudosa = datos_dudosos.std(ddof=1)
    media_confiable = datos_confiables.mean()
    std_confiable = datos_confiables.std(ddof=1)
    
    if std_dudosa == 0 or std_dudosa is np.nan or std_confiable is np.nan:
        st.error("Error: La desviación estándar de uno de los grupos es 0 o nula.")
        return None
        
    datos_corregidos = ((datos_dudosos - media_dudosa) / std_dudosa) * std_confiable + media_confiable
    return datos_corregidos

def find_index(col_list, possible_names, default=0):
    """Helper para encontrar el índice por defecto de una columna."""
    col_list_lower = [str(col).lower() for col in col_list]
    for name in possible_names:
        if name.lower() in col_list_lower:
            return col_list_lower.index(name.lower())
    if default < len(col_list):
        return default
    return 0

def cargar_datos_flex(tipo_datos, xls):
    """Función reutilizable para cargar datos de forma flexible."""
    st.subheader(f"Carga de Datos: {tipo_datos}")
    
    sheet_names = ["---"] + xls.sheet_names
    selected_sheet = st.selectbox(f"1. Elige la Hoja (Tab) con los datos de {tipo_datos}:", sheet_names, index=0)
    
    if selected_sheet == "---":
        return None

    st.write("Indica dónde empiezan tus datos en la hoja de Excel:")
    st.info("La primera fila en Excel es la 1, pero para el código, la primera fila es la 0.")
    
    col_fila_header, col_fila_datos = st.columns(2)
    with col_fila_header:
        fila_header = st.number_input(
            f"2. Fila del Encabezado ({tipo_datos}):", 
            min_value=0, value=0, 
            help="Es la fila (0-indexada) que tiene los nombres."
        )
    with col_fila_datos:
        fila_datos = st.number_input(
            f"3. Fila de Inicio de Datos ({tipo_datos}):", 
            min_value=0, value=fila_header + 1, 
            help="Es la fila (0-indexada) con los PRIMEROS NÚMEROS."
        )

    if fila_datos <= fila_header:
        st.error("La 'Fila de Inicio de Datos' debe ser mayor que la 'Fila del Encabezado'.")
        return None
        
    try:
        df_temp = pd.read_excel(xls, sheet_name=selected_sheet, header=fila_header)
        num_filas_basura = (fila_datos) - (fila_header + 1)
        
        if num_filas_basura < 0:
             st.error("Error lógico en las filas.")
             return None
        
        df_final = df_temp.iloc[num_filas_basura:].reset_index(drop=True).dropna(how='all')
        
        st.success(f"Hoja '{selected_sheet}' cargada exitosamente.")
        st.dataframe(df_final.head())
        return df_final
    
    except Exception as e:
        st.error(f"Error al leer la hoja '{selected_sheet}': {e}")
        return None


def main():
    st.title("🔧 Práctica de Diagrama de Doble Masa")
    st.markdown("### **By: AMAYA17**")
    st.markdown("---")

    st.sidebar.header("📁 Paso 1: Cargar Archivo")
    st.sidebar.markdown("**Desarrollado por: AMAYA17**")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel", type=["xlsx"])

    if not uploaded_file:
        st.info("Sube un archivo Excel (`.xlsx`) en la barra lateral izquierda para comenzar.")
        return

    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
        return
    
    st.header("Práctica: Diagrama de Doble Masa")
    
    df_original = cargar_datos_flex("Doble Masa", xls)

    if df_original is not None:
        st.subheader("4. Mapeo de Columnas (Doble Masa)")
        all_columns = df_original.columns.tolist()
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            idx_ano = find_index(all_columns, ['Año', 'Ano', 'Year'], default=0)
            col_ano = st.selectbox("Selecciona la columna de Años:", all_columns, index=idx_ano)
        with col_c2:
            options_estaciones = [col for col in all_columns if col != col_ano]
            cols_estaciones = st.multiselect("Selecciona TODAS las estaciones a analizar:", 
                                             options_estaciones, default=options_estaciones)
        
        if not col_ano or not cols_estaciones:
            st.warning("Por favor, mapea la columna de Años y al menos una estación.")
            return
        else:
            try:
                df_original[col_ano] = pd.to_numeric(df_original[col_ano], errors='coerce').astype(int)
                for col in cols_estaciones:
                    df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
                df_original = df_original.dropna(subset=[col_ano] + cols_estaciones, how='any')
            except Exception as e:
                st.error(f"Error al convertir tipos de datos: {e}")
                return

            st.success("¡Datos mapeados! Iniciando análisis...")
            st.markdown("---")
            
            st.subheader("Paso 1: Identificación de Tendencias (Gráfico 1)")
            st.write("Análisis de la precipitación anual (original) vs. Años.")
            
            df_melted = df_original.melt(id_vars=[col_ano], value_vars=cols_estaciones, 
                                         var_name='Estacion', value_name='Precipitacion_Anual')
            
            for estacion in cols_estaciones:
                st.markdown(f"**Estación: {estacion}**")
                df_estacion = df_melted[df_melted['Estacion'] == estacion]
                
                fig1 = px.line(df_estacion, x=col_ano, y='Precipitacion_Anual', 
                               title=f"Tendencia Anual: {estacion}", markers=True)
                
                fig1.update_layout(height=450, hovermode="x unified")
                st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Paso 1.5: Selección de Estación Patrón (Gráfico 2)")
            
            df_acum = calcular_doble_masa(df_original, cols_estaciones)
            
            patron_auto, r_cuadrados = encontrar_patron_auto(df_acum, cols_estaciones, 'P_media_acum')
            st.write("Análisis de R² (Cercanía a 1 = más recta):", r_cuadrados)
            
            patron_auto_index = 0
            if patron_auto and patron_auto in cols_estaciones:
                patron_auto_index = cols_estaciones.index(patron_auto)
                
            col_patron = st.selectbox("Selecciona la Estación Patrón:", 
                                      cols_estaciones, 
                                      index=patron_auto_index,
                                      help="Recomendación: estación con R² más alto.")
            
            cols_acum_plot = [f'P_acum_{col}' for col in cols_estaciones]
            df_acum_melted = df_acum.melt(id_vars=['P_media_acum'], value_vars=cols_acum_plot, 
                                          var_name='Estacion_Acumulada', value_name='Precipitacion_Acumulada')
            
            fig2 = px.line(df_acum_melted, x='P_media_acum', y='Precipitacion_Acumulada', 
                           color='Estacion_Acumulada',
                           title="Doble Masa: Estaciones Acumuladas vs. Media Acumulada", markers=True)
            
            fig2.update_layout(height=500, xaxis_title="Media Acumulada", 
                             yaxis_title="Precipitación Acumulada")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Tabla de Datos Acumulados (Referencia)")
            st.dataframe(df_acum)

            st.subheader(f"Paso 2: Evaluación de Quiebre (vs. Patrón '{col_patron}')")
            
            opciones_evaluar = [col for col in cols_estaciones if col != col_patron]
            if not opciones_evaluar:
                st.warning("Necesitas al menos 2 estaciones para comparar.")
            else:
                col_evaluar = st.selectbox("Selecciona la Estación a Evaluar:", opciones_evaluar)
                
                col_patron_acum = f'P_acum_{col_patron}'
                col_evaluar_acum = f'P_acum_{col_evaluar}'
                
                fig3 = px.line(df_acum, x=col_patron_acum, y=col_evaluar_acum, 
                                title=f"Doble Masa: {col_evaluar} (Y) vs. {col_patron} (X)",
                                hover_data=[col_ano], 
                                markers=True)
                
                fig3.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                                 line=dict(width=2))
                fig3.update_layout(height=500, xaxis_title=f"Acumulada Patrón: {col_patron}", 
                                 yaxis_title=f"Acumulada a Evaluar: {col_evaluar}")
                st.plotly_chart(fig3, use_container_width=True)
                
                st.info(f"¿En qué año ves el quiebre para '{col_evaluar}'?")
                min_ano = int(df_original[col_ano].min()) + 1
                max_ano = int(df_original[col_ano].max()) - 1
                
                if min_ano >= max_ano:
                    st.error("No hay suficientes años para encontrar un quiebre.")
                else:
                    default_quiebre = min_ano + int((max_ano - min_ano) / 2)
                    
                    if 'ano_quiebre' not in st.session_state or \
                       st.session_state.ano_quiebre < min_ano or \
                       st.session_state.ano_quiebre > max_ano:
                        st.session_state.ano_quiebre = default_quiebre

                    st.number_input(
                        "Año donde inicia el segundo tramo (quiebre):", 
                        min_value=min_ano, 
                        max_value=max_ano,
                        value=st.session_state.ano_quiebre, 
                        key='ano_quiebre'  
                    )
                    
                    ano_quiebre_actual = st.session_state.ano_quiebre
                    
                    stats_results, datos_antes, datos_despues, error_msg = calcular_estadisticas_quiebre(
                        df_original, col_evaluar, ano_quiebre_actual, col_ano
                    )
                    
                    if error_msg:
                        st.error(error_msg)
                    elif stats_results:
                        st.write(f"Análisis estadístico para **{col_evaluar}** quebrando en {ano_quiebre_actual}")
                        
                        st.markdown("---")
                        st.subheader("Resultados de Pruebas Estadísticas")

                        data_stats = {
                            "Parámetro": [
                                "Grupo 1 (Antes)", 
                                "Grupo 2 (Después)", 
                                "Desv. Est. Ponderada (Sp)", 
                                "Desv. Est. de Promedios (Sd)"
                            ],
                            "Valor": [
                                f"{stats_results['n1']} datos (años < {ano_quiebre_actual})",
                                f"{stats_results['n2']} datos",
                                "-",
                                "-"
                            ],
                            "Media": [
                                f"{stats_results['media_antes']:.2f}",
                                f"{stats_results['media_despues']:.2f}",
                                "-",
                                "-"
                            ],
                            "Varianza (S^2)": [
                                f"{stats_results['var_antes']:.2f}",
                                f"{stats_results['var_despues']:.2f}",
                                "-",
                                "-"
                            ],
                            "Cálculos T-Student": [
                                "-",
                                "-",
                                f"{stats_results['sp']:.2f}",
                                f"{stats_results['sd']:.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(data_stats))

                        st.markdown("---")
                        st.subheader("Resultados de Pruebas")

                        data_pruebas = {
                            "Prueba": ["T-Student (Medias)", "F-Fisher (Varianzas)"],
                            "Calculado": [
                                f"{stats_results['t_stat']:.4f}",
                                f"{stats_results['f_stat']:.4f}"
                            ],
                            "Tabla": [
                                f"{stats_results['tt_tabla']:.4f}",
                                f"{stats_results['ft_tabla']:.4f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(data_pruebas))

                        is_t_inconsistent = stats_results['t_stat'] > stats_results['tt_tabla']
                        is_f_inconsistent = stats_results['f_stat'] > stats_results['ft_tabla']

                        if is_t_inconsistent:
                            st.error(f"INCONSISTENTE (T-Student): Tc > Tt")
                        else:
                            st.success(f"CONSISTENTE (T-Student): Tc <= Tt")
                        
                        if is_f_inconsistent:
                            st.error(f"INCONSISTENTE (F-Fisher): Fc > Ft")
                        else:
                            st.success(f"CONSISTENTE (F-Fisher): Fc <= Ft")

                        if is_t_inconsistent or is_f_inconsistent:
                            st.markdown("---")
                            st.subheader("Paso 3: Corrección de Datos")
                            st.warning(f"Inconsistencia detectada. Corrigiendo '{col_evaluar}'.")
                            
                            grupo_dudoso_opcion = st.radio(
                                "¿Qué grupo corregir?",
                                [f"Grupo 1 (Antes de {ano_quiebre_actual})", 
                                 f"Grupo 2 (Después de {ano_quiebre_actual})"]
                            )
                            
                            df_corregido = df_original.copy()
                            col_corregida = f"{col_evaluar}_Corregida"
                            
                            datos_antes_full = df_original[df_original[col_ano] < ano_quiebre_actual][col_evaluar].dropna()
                            datos_despues_full = df_original[df_original[col_ano] >= ano_quiebre_actual][col_evaluar].dropna()

                            if grupo_dudoso_opcion == f"Grupo 1 (Antes de {ano_quiebre_actual})":
                                datos_dudosos = datos_antes_full
                                datos_confiables = datos_despues_full
                            else:
                                datos_dudosos = datos_despues_full
                                datos_confiables = datos_antes_full
                            
                            datos_corregidos_serie = corregir_datos(datos_dudosos, datos_confiables)
                            
                            if datos_corregidos_serie is not None:
                                df_corregido[col_corregida] = df_original[col_evaluar]
                                df_corregido.loc[datos_dudosos.index, col_corregida] = datos_corregidos_serie
                                
                                st.write(f"Tabla: '{col_evaluar}' Original vs. Corregida")
                                st.dataframe(df_corregido[[col_ano, col_evaluar, col_corregida]].dropna())
                                
                                st.write("Gráfico: Original vs. Corregida")
                                
                                fig_final = go.Figure()
                                fig_final.add_trace(go.Scatter(
                                    x=df_corregido[col_ano], y=df_corregido[col_evaluar],
                                    mode='lines+markers', name=f'{col_evaluar} (Original)',
                                    line=dict(dash='dash', color='red', width=2),
                                    marker=dict(size=6, color='red')
                                ))
                                fig_final.add_trace(go.Scatter(
                                    x=df_corregido[col_ano], y=df_corregido[col_corregida],
                                    mode='lines+markers', name=f'{col_evaluar} (Corregida)',
                                    line=dict(color='blue', width=3),
                                    marker=dict(size=8, color='blue')
                                ))
                                fig_final.update_layout(height=500, 
                                                      title=f"Datos Corregidos: {col_evaluar}",
                                                      xaxis_title="Año", 
                                                      yaxis_title="Precipitación Anual",
                                                      hovermode="x unified")
                                st.plotly_chart(fig_final, use_container_width=True)
                            
                        else:
                            st.success("No se detectó inconsistencia. No es necesario corregir (Paso 3).")

if __name__ == "__main__":
    main()
