import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Fuzzy Logic Decision Support System",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff;
        text-align: center;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: #a78bfa;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 20px 0;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #10b98115 0%, #059c6915 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 20px 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1b4b;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #a78bfa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk normalisasi SAW
def normalize_saw(df, criteria_types):
    normalized = df.copy()
    for col in df.columns:
        if col != 'Alternatif':
            if criteria_types.get(col) == 'Benefit':
                normalized[col] = df[col] / df[col].max()
            else:  # Cost
                normalized[col] = df[col].min() / df[col]
    return normalized

# Fungsi untuk menghitung SAW
def calculate_saw(df, weights, criteria_types):
    normalized = normalize_saw(df[df.columns[1:]], criteria_types)
    
    scores = pd.DataFrame()
    scores['Alternatif'] = df['Alternatif']
    
    for col in normalized.columns:
        if col != 'Alternatif':
            scores[col] = normalized[col] * weights.get(col, 0)
    
    scores['Total Score'] = scores[scores.columns[1:]].sum(axis=1)
    scores['Rank'] = scores['Total Score'].rank(ascending=False, method='min').astype(int)
    
    return scores.sort_values('Total Score', ascending=False)

# Fungsi untuk normalisasi bobot WP
def normalize_weights_wp(weights):
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

# Fungsi untuk menghitung WP
def calculate_wp(df, weights, criteria_types):
    normalized_weights = normalize_weights_wp(weights)
    
    scores = pd.DataFrame()
    scores['Alternatif'] = df['Alternatif']
    
    s_values = np.ones(len(df))
    
    for col in df.columns[1:]:
        weight = normalized_weights.get(col, 0)
        if criteria_types.get(col) == 'Cost':
            weight = -weight
        s_values *= np.power(df[col].values, weight)
    
    scores['S Value'] = s_values
    scores['V Value'] = scores['S Value'] / scores['S Value'].sum()
    scores['Rank'] = scores['V Value'].rank(ascending=False, method='min').astype(int)
    
    return scores.sort_values('V Value', ascending=False)

# Fungsi untuk membuat chart perbandingan
def create_comparison_chart(saw_results, wp_results):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='SAW Method',
        x=saw_results['Alternatif'],
        y=saw_results['Total Score'],
        marker_color='rgb(102, 126, 234)',
        text=saw_results['Total Score'].round(4),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='WP Method',
        x=wp_results['Alternatif'],
        y=wp_results['V Value'],
        marker_color='rgb(118, 75, 162)',
        text=wp_results['V Value'].round(4),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Perbandingan Hasil SAW vs WP',
        xaxis_title='Alternatif',
        yaxis_title='Score',
        barmode='group',
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Header
st.markdown("""
    <h1>Sistem Pendukung Keputusan Cloud Storage</h1>
    <p style='text-align: center; color: #a78bfa; font-size: 18px; margin-bottom: 30px;'>
        by Livia Citra Atrianda
    </p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigasi")
    page = st.radio(
        "",
        ["Home", "Perhitungan", "Perbandingan"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Informasi")
    st.markdown("""
        **Dibuat oleh:**  
        Livia 
        
        **Studi Kasus:**  
        Pemilihan Layanan  
        Cloud Storage
    """)

# Home Page
if page == "Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h3 style='text-align: center;'>SAW Method</h3>
                <p style='text-align: center; color: #9ca3af;'>
                    Simple Additive Weighting untuk penjumlahan terbobot
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h3 style='text-align: center;'>WP Method</h3>
                <p style='text-align: center; color: #9ca3af;'>
                    Weighted Product untuk perkalian terbobot
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='info-box'>
                <h3 style='text-align: center;'>Fuzzy Logic</h3>
                <p style='text-align: center; color: #9ca3af;'>
                    Logika kabur untuk ketidakpastian
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload file section
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader(
        "Upload file Excel dengan data alternatif dan kriteria",
        type=['xlsx', 'xls'],
        help="File harus berisi sheet 'Metode SAW' dan 'Metode WP'"
    )
    
    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file
        st.success("File berhasil diupload!")
    
    # Input manual section
    st.markdown("### Input Data Manual")
    
    with st.expander("Input Alternatif & Kriteria"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_alternatives = st.number_input("Jumlah Alternatif", min_value=2, max_value=10, value=5)
        
        with col2:
            num_criteria = st.number_input("Jumlah Kriteria", min_value=2, max_value=10, value=5)
        
        if st.button("Generate Input Form", use_container_width=True):
            st.session_state['num_alternatives'] = num_alternatives
            st.session_state['num_criteria'] = num_criteria
            st.rerun()
    
    
# Perhitungan Page
elif page == "Perhitungan":
    st.markdown("### Perhitungan SPK")
    
    # Tabs untuk SAW dan WP
    tab1, tab2 = st.tabs(["Metode SAW", "Metode WP"])
    
    # Default data jika tidak ada upload
    default_data = {
        'Alternatif': ['Google One', 'Dropbox Plus', 'OneDrive', 'Mega Pro', 'Box Business'],
        'C1 Harga': [26900, 145000, 35000, 80000, 225000],
        'C2 Kapasitas': [100, 2000, 1000, 2000, 1000],
        'C3 Upload': [0, 2, 0, 0, 5],
        'C4 Device': [5, 1, 1, 1, 99],
        'C5 Keamanan': [4, 5, 4, 5, 5]
    }
    
    df_input = pd.DataFrame(default_data)
    
    # Check if file uploaded
    if 'uploaded_file' in st.session_state:
        try:
            uploaded_file = st.session_state['uploaded_file']
            df_saw_upload = pd.read_excel(uploaded_file, sheet_name='Metode SAW', skiprows=17, nrows=10)
            df_saw_upload.columns = df_saw_upload.iloc[0]
            df_saw_upload = df_saw_upload[1:6].reset_index(drop=True)
            df_input = df_saw_upload
        except:
            st.warning("Menggunakan data default karena format file tidak sesuai")
    
    with tab1:
        st.markdown("#### Data Input")
        st.dataframe(df_input, use_container_width=True)
        
        # Kriteria settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tipe Kriteria")
            criteria_types = {}
            for col in df_input.columns[1:]:
                criteria_types[col] = st.selectbox(
                    f"{col}",
                    ['Benefit', 'Cost'],
                    key=f'saw_type_{col}'
                )
        
        with col2:
            st.markdown("#### Bobot Kriteria")
            weights = {}
            for col in df_input.columns[1:]:
                weights[col] = st.slider(
                    f"{col}",
                    0.0, 1.0, 0.2,
                    key=f'saw_weight_{col}'
                )
        
        if st.button("Hitung SAW", use_container_width=True):
            results_saw = calculate_saw(df_input, weights, criteria_types)
            
            st.markdown("---")
            st.markdown("#### Hasil Perhitungan SAW")
            
            # Display winner
            winner = results_saw.iloc[0]
            st.markdown(f"""
                <div class='success-box'>
                    <h2 style='text-align: center; color: #10b981; margin: 0;'>
                        {winner['Alternatif']}
                    </h2>
                    <p style='text-align: center; color: #9ca3af; margin: 5px 0 0 0;'>
                        Score: {winner['Total Score']:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display all results
            st.dataframe(
                results_saw,
                use_container_width=True
            )
            
            # Chart
            fig = px.bar(
                results_saw,
                x='Alternatif',
                y='Total Score',
                color='Total Score',
                color_continuous_scale='Purples',
                text='Total Score'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Data Input")
        st.dataframe(df_input, use_container_width=True)
        
        # Kriteria settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tipe Kriteria")
            criteria_types_wp = {}
            for col in df_input.columns[1:]:
                criteria_types_wp[col] = st.selectbox(
                    f"{col}",
                    ['Benefit', 'Cost'],
                    key=f'wp_type_{col}'
                )
        
        with col2:
            st.markdown("#### Bobot Kriteria")
            weights_wp = {}
            for col in df_input.columns[1:]:
                weights_wp[col] = st.slider(
                    f"{col}",
                    1, 10, 5,
                    key=f'wp_weight_{col}'
                )
        
        if st.button("Hitung WP", use_container_width=True):
            results_wp = calculate_wp(df_input, weights_wp, criteria_types_wp)
            
            st.markdown("---")
            st.markdown("#### Hasil Perhitungan WP")
            
            # Display winner
            winner = results_wp.iloc[0]
            st.markdown(f"""
                <div class='success-box'>
                    <h2 style='text-align: center; color: #10b981; margin: 0;'>
                        {winner['Alternatif']}
                    </h2>
                    <p style='text-align: center; color: #9ca3af; margin: 5px 0 0 0;'>
                        V Value: {winner['V Value']:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display all results
            st.dataframe(
                results_wp,
                use_container_width=True
            )
            
            # Chart
            fig = px.bar(
                results_wp,
                x='Alternatif',
                y='V Value',
                color='V Value',
                color_continuous_scale='Purples',
                text='V Value'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# Perbandingan Page
elif page == "Perbandingan":
    st.markdown("### Perbandingan Metode SAW & WP")
    
    # Default data
    default_data = {
        'Alternatif': ['Hostinger', 'Niagahoster', 'Dewaweb', 'IDCloudHost', 'Qwords'],
        'C1 Harga': [25000, 45000, 40000, 35000, 30000],
        'C2 Storage': [10, 20, 15, 12, 18],
        'C3 Bandwidth': [100, 200, 150, 120, 180],
        'C4 Uptime': [99.5, 99.9, 99.7, 99.6, 99.8],
        'C5 Support': [4, 5, 4, 3, 5]
    }
    
    df_input = pd.DataFrame(default_data)
    
    # Settings for comparison
    criteria_types = {col: 'Benefit' if col != 'C1 Harga' else 'Cost' for col in df_input.columns[1:]}
    weights = {col: 0.2 for col in df_input.columns[1:]}
    weights_wp = {col: 5 for col in df_input.columns[1:]}
    
    if st.button("Bandingkan Kedua Metode", use_container_width=True):
        # Calculate both methods
        results_saw = calculate_saw(df_input, weights, criteria_types)
        results_wp = calculate_wp(df_input, weights_wp, criteria_types)
        
        # Display comparison chart
        fig = create_comparison_chart(results_saw, results_wp)
        st.plotly_chart(fig, use_container_width=True)
        
        # Side by side results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hasil SAW")
            st.dataframe(
                results_saw[['Alternatif', 'Total Score', 'Rank']],
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Hasil WP")
            st.dataframe(
                results_wp[['Alternatif', 'V Value', 'Rank']],
                use_container_width=True
            )
        
        # Comparison metrics
        st.markdown("---")
        st.markdown("#### Metrik Perbandingan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            winner_saw = results_saw.iloc[0]['Alternatif']
            st.metric("Pemenang SAW", winner_saw)
        
        with col2:
            winner_wp = results_wp.iloc[0]['Alternatif']
            st.metric("Pemenang WP", winner_wp)
        
        with col3:
            agreement = "Sama" if winner_saw == winner_wp else "Berbeda"
            st.metric("Konsistensi", agreement)

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #6b7280; font-size: 14px;'>
        by Livia Citra Atrianda | 
        <a href='https://github.com' style='color: #a78bfa;'>GitHub</a> | 
        2025
    </p>
""", unsafe_allow_html=True)
