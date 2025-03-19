# data_sweeper.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import plotly.express as px
from PyPDF2 import PdfReader
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# App Configuration
st.set_page_config(
    page_title="Data Sweeper",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧹 Data Sweeper")
    st.markdown("""
    ## Intelligent Data Transformation Suite
    **Clean • Convert • Visualize**
    """)
    
    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["csv", "xml", "xlsx", "pdf", "json"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        process_file(uploaded_file)

def process_file(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'pdf':
            process_pdf(uploaded_file)
        elif file_ext == 'csv':
            process_csv(uploaded_file)
        elif file_ext == 'xlsx':
            process_excel(uploaded_file)
        elif file_ext == 'xml':
            process_xml(uploaded_file)
        elif file_ext == 'json':
            process_json(uploaded_file)
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")

# PDF Processing Functions
def extract_pdf_text(uploaded_file):
    try:
        pdf = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf.pages])
        with st.expander("Extracted Text Preview"):
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))
        return text
    except Exception as e:
        st.error(f"Text Extraction Failed: {str(e)}")
        return None

def extract_pdf_tables(uploaded_file):
    try:
        dfs = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables({
                    'vertical_strategy': 'lines', 
                    'horizontal_strategy': 'lines',
                    'snap_tolerance': 3
                })
                for table in tables:
                    if len(table) > 1:
                        header = [str(cell or f"Col_{idx}") for idx, cell in enumerate(table[0])]
                        data = [[str(cell or "") for cell in row] for row in table[1:]]
                        dfs.append(pd.DataFrame(data, columns=header))
        return dfs if dfs else None
    except Exception as e:
        st.error(f"Table Extraction Error: {str(e)}")
        return None

def process_pdf(uploaded_file):
    conversion_type = st.radio(
        "Conversion Type:",
        ["Text Extraction", "Table Extraction"],
        horizontal=True
    )
    
    if conversion_type == "Text Extraction":
        text = extract_pdf_text(uploaded_file)
        if text:
            st.download_button(
                "Download Text",
                text,
                "extracted_text.txt",
                "text/plain"
            )
    else:
        dfs = extract_pdf_tables(uploaded_file)
        if dfs:
            st.success(f"Found {len(dfs)} tables")
            for i, df in enumerate(dfs):
                st.subheader(f"Table {i+1}")
                df = data_operations(df, f"pdf_{i}")
                df = data_cleaning(df, f"pdf_{i}")
                show_data_preview(df)
                conversion_interface(df, f"pdf_{i}")
        else:
            st.warning("No tables detected. Try PDFs with text-based tables.")

# Data Operations
def data_operations(df, context=""):
    with st.sidebar.expander("🔧 Column & Visualization", expanded=True):
        # Column Selection
        st.subheader("Column Management")
        selected_cols = st.multiselect(
            "Select columns to keep:",
            df.columns,
            default=df.columns.tolist(),
            key=f"cols_{context}"
        )
        df = df[selected_cols]
        
        # Data Visualization
        st.subheader("Data Explorer")
        plot_type = st.selectbox(
            "Visualization Type:",
            ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie"],
            key=f"plot_{context}"
        )
        
        if plot_type in ["Scatter", "Line", "Bar"]:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-Axis", df.columns, key=f"x_{context}")
            with col2:
                y_axis = st.selectbox("Y-Axis", df.columns, key=f"y_{context}")
            if st.button("Generate", key=f"btn_{context}"):
                fig = getattr(px, plot_type.lower())(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Histogram":
            col = st.selectbox("Column", df.columns, key=f"hist_{context}")
            if st.button("Generate"):
                fig = px.histogram(df, x=col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box":
            col = st.selectbox("Column", df.columns, key=f"box_{context}")
            if st.button("Generate"):
                fig = px.box(df, y=col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Pie":
            col = st.selectbox("Column", df.columns, key=f"pie_{context}")
            if st.button("Generate"):
                fig = px.pie(df, names=col)
                st.plotly_chart(fig, use_container_width=True)
    
    return df

def data_cleaning(df, context=""):
    with st.expander("🧼 Data Cleaning", expanded=True):
        orig_shape = df.shape
        
        # Duplicates
        dup_cols = st.multiselect(
            "Duplicate Columns:",
            df.columns,
            key=f"dup_cols_{context}"
        )
        if dup_cols:
            action = st.selectbox(
                "Duplicate Action:",
                ["Remove", "Mark", "Replace"],
                key=f"dup_action_{context}"
            )
            if action == "Remove":
                df = df.drop_duplicates(dup_cols)
            elif action == "Mark":
                df["is_duplicate"] = df.duplicated(dup_cols, keep=False)
            elif action == "Replace":
                replace_val = st.text_input("Replacement Value:", key=f"replace_{context}")
                df.loc[df.duplicated(dup_cols, keep='first'), dup_cols] = replace_val

        # Missing Values
        missing_action = st.selectbox(
            "Missing Values:",
            ["Delete Rows", "Delete Columns", "Fill Values"],
            key=f"missing_{context}"
        )
        if missing_action == "Delete Rows":
            df = df.dropna()
        elif missing_action == "Delete Columns":
            df = df.dropna(axis=1)
        elif missing_action == "Fill Values":
            fill_val = st.text_input("Fill With:", key=f"fill_{context}")
            df = df.fillna(fill_val)

        st.write(f"Original: {orig_shape} → Cleaned: {df.shape}")
        return df

def conversion_interface(df, context=""):
    with st.sidebar.expander("⚡ Conversion Setup", expanded=True):
        target_format = st.selectbox(
            "Output Format:",
            ["CSV", "Excel", "JSON", "XML", "PDF"],
            key=f"format_{context}"
        )
        
        conv_params = {}
        if st.checkbox("Advanced Options", key=f"adv_{context}"):
            conv_params["index"] = st.checkbox("Include Index", False)
            conv_params["encoding"] = st.selectbox(
                "Encoding:", ["utf-8", "latin-1", "utf-16"]
            )
            
            if target_format == "CSV":
                conv_params["sep"] = st.selectbox(
                    "Separator:", [",", ";", "|", "\t"]
                )
            
            if target_format == "Excel":
                conv_params["sheet_name"] = st.text_input("Sheet Name:", "Sheet1")
            
            if target_format == "JSON":
                conv_params["orient"] = st.selectbox(
                    "JSON Style:", ["records", "split", "index"]
                )

    col1, col2 = st.columns(2)
    with col1:
        show_data_preview(df)
    with col2:
        if target_format == "CSV":
            convert_to_csv(df, conv_params)
        elif target_format == "Excel":
            convert_to_excel(df, conv_params)
        elif target_format == "JSON":
            convert_to_json(df, conv_params)
        elif target_format == "XML":
            convert_to_xml(df, conv_params)
        elif target_format == "PDF":
            convert_to_pdf(df, conv_params)

# Conversion Functions
def convert_to_csv(df, params):
    output = io.BytesIO()
    df.to_csv(output, index=params.get("index", False), 
             sep=params.get("sep", ","), 
             encoding=params.get("encoding", "utf-8"))
    st.download_button(
        "Download CSV",
        output.getvalue(),
        "converted.csv",
        "text/csv"
    )

def convert_to_excel(df, params):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, 
                   index=params.get("index", False),
                   sheet_name=params.get("sheet_name", "Sheet1"))
    st.download_button(
        "Download Excel",
        output.getvalue(),
        "converted.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def convert_to_json(df, params):
    st.download_button(
        "Download JSON",
        df.to_json(orient=params.get("orient", "records")),
        "converted.json",
        "application/json"
    )

def convert_to_xml(df, params):
    xml_data = df.to_xml(index=params.get("index", False))
    st.download_button(
        "Download XML",
        xml_data,
        "converted.xml",
        "application/xml"
    )

def convert_to_pdf(df, params):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    table_data = [df.columns.tolist()] + df.values.tolist()
    pdf_table = Table(table_data)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), '#CCCCCC'),
        ('TEXTCOLOR', (0,0), (-1,0), '#000000'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, '#000000')
    ])
    pdf_table.setStyle(style)
    doc.build([pdf_table])
    st.download_button(
        "Download PDF",
        buffer.getvalue(),
        "converted.pdf",
        "application/pdf"
    )

def show_data_preview(df):
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"Shape: {df.shape} | Columns: {len(df.columns)}")

# File Processors
def process_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = data_operations(df, "csv")
    df = data_cleaning(df, "csv")
    conversion_interface(df, "csv")

def process_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df = data_operations(df, "excel")
    df = data_cleaning(df, "excel")
    conversion_interface(df, "excel")

def process_xml(uploaded_file):
    df = pd.read_xml(uploaded_file)
    df = data_operations(df, "xml")
    df = data_cleaning(df, "xml")
    conversion_interface(df, "xml")

def process_json(uploaded_file):
    df = pd.read_json(uploaded_file)
    df = data_operations(df, "json")
    df = data_cleaning(df, "json")
    conversion_interface(df, "json")

if __name__ == "__main__":
    main()