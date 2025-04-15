import streamlit as st
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import model
import plotly.graph_objects as go

# Initialize session state
if "prediction_counts" not in st.session_state:
    st.session_state.prediction_counts = [0, 0]

def show_images(images):
    """Display images in Streamlit with 4 images per row"""
    if images:
        n_images = len(images)
        n_cols = 4
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        
        if n_rows == 1:
            axes = axes.tolist()
        else:
            axes = axes.flatten().tolist()
        
        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img[:, :, ::-1])  # Convert BGR to RGB
            ax.axis("off")
        
        for i in range(n_images, len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        st.pyplot(fig)

def app():
    # Ensure session state is always a list
    if not isinstance(st.session_state.prediction_counts, list):
        st.session_state.prediction_counts = [0, 0]

    # Excel data loading
    sheet_id = "1RA8m_0xJ3kJGPavrf8JQ-V84MAKHUWGYSmSLSE-0lVA"
    sheet_name = "main"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    main_df = df[["folder","img no.","actual width (mm)","actual length (mm)","actual total weight (g)","actual kernel weight (g)","Nuts_Area","volume( w x h x d )","label"]]
    main_df = main_df.rename(columns={
        "actual width (mm)": "actual_width", 
        "actual length (mm)": "actual_length",
        "actual total weight (g)": "actual_total_weight",
        "actual kernel weight (g)": "actual_kernel_weight",
        "volume( w x h x d )": "volume"
    })
    main_df[["label"]] = main_df[["label"]].astype(str)
    labels = ['True', 'False']
    values = [0,0]
    size_error_percentage_acc = 0
    weight_error_percentage_acc = 0
    
    
    st.markdown(
        """
        <style>
        img {
            border-radius: 0.5rem;
        }
        h2 { padding: 0; }
        .block-container {
            max-width: none;
            padding: 2rem 2rem 0 2rem;
        }
        tspan {
            font-weight: normal !important;
        }
        /* Apply border to the second column (output_panel) */
    
        [data-testid="column"] {
            max-height: 90vh;
        }
        [data-testid="baseButton-secondary"] {
            display: none;
        }
        [data-testid="StyledFullScreenButton"] {
            display: none;
        }
        [data-testid="stVerticalBlockBorderWrapper"][height= "400"]  {
            border: none;
            padding: 0 1rem 0 0;
            margin-top: 0.5rem;
            height: 220px;
            overflow-y: scroll;
        }
        [data-testid="stVerticalBlockBorderWrapper"][height= "70"]  {
            margin: 0 1rem 0 0;
            overflow: hidden;
            padding: 2px;
        }
        [data-testid="stVerticalBlockBorderWrapper"][height= "700"]  {
            border: none;
            padding: 0 1rem 0 0;
            margin: 0;
            max-height: 90vh;
            overflow-y: scroll;
        }
        span > div > p {
            font-size: 18px !important;
            font-weight: 600;
            padding-left: 0.5rem;
        }            
        [data-testid="stExpanderDetails"] {
            padding: 2rem 1rem 0.5rem 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    input_panel, output_panel = st.columns([3, 7], gap="medium")
    images = []
    images_title = []
        
    
    with input_panel:
        st.markdown(
            "<h1 style='font-size: 30px; padding: 0; color: #006000;'>AI System for Assessing Cashew Nut Quality</h1>"
            "<p style='padding-top: 16px; padding-bottom: 16px;'>This is a simple app to demonstrate the results of assessing cashew nut quality.</p>",
            unsafe_allow_html=True
        )

        overall_stat = st.empty()

        
        st.markdown(
            "<h2 style='font-size: 20px; margin-top: 1rem; padding-bottom: 10px;'>Upload Images</h2>",
            unsafe_allow_html=True
        )
        with st.container(height=400):
            
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Select multiple image files to upload"
            )
        
            load_from_directory = st.checkbox(
                "Load from directory if no images uploaded",
                value=False,
                help="Check to load images from 'data' directory when no files are uploaded"
            )
        

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)
                        images_title.append(uploaded_file.name)
                st.markdown(
                    f"<div style='padding: 8px 16px; margin-bottom:1rem; border-radius: 5px; background-color: rgba(33, 195, 84, 0.1);'><p style='color: rgb(23, 114, 51); margin: 0;'>Successfully uploaded {len(images)} images</p></div>",
                    unsafe_allow_html=True
                )
        
            elif load_from_directory:
                images_title = glob.glob("data/*.jpg")
                images = [cv2.imread(img) for img in images_title if cv2.imread(img) is not None]
                if images:
                    st.markdown(
                        f"<p style='color: green;'>Loaded {len(images)} images from 'data' directory</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write("No images found in 'data' directory")
            
            show_images(images)

    with output_panel:
        with st.container(height=700):
            for image, image_title in zip(images, images_title):
                folder = image_title[-11]+"_"+image_title[-9]
                img_no = int(image_title[-7:-4])
                filtered_df = main_df[(main_df["folder"] == folder) & (main_df["img no."] == img_no)]
                
                if not filtered_df.empty:
                    actual_width, actual_length, actual_total_weight, actual_kernel_weight, actual_total_area, actual_volume, label = filtered_df[["actual_width", "actual_length","actual_total_weight","actual_kernel_weight","Nuts_Area","volume","label"]].values[0]
                    predict_width, predict_length, predict_total_area, predict_kernel_area, area_percentage, rect_image, area_image, roi1, roi2 = model.model1(image)
                   
                    model2 = model.model2(actual_kernel_weight, predict_total_area, area_percentage)
                    model3_1, model3_2 = model.model3(roi1, roi2)
                    model5 = model.model5(predict_total_area, actual_total_weight)
                    model6 = model.model6(predict_total_area)
                    result = model.weight_voting(model2, model3_1, model3_2, model5)
                    
                    error_width, error_length, error_weight = (abs(actual_width - predict_width), abs(actual_length - predict_length), abs(actual_total_weight - model6))
                    error_width_percentage, error_length_percentage, error_weight_percentage = ((error_width*100)/actual_width, (error_length*100)/actual_length, (error_weight*100)/actual_total_weight)

                    is_correct = (result == int(label))
                    if is_correct:
                        values[0] += 1  # Correct
                    else:
                        values[1] += 1  # Incorrect
                        
                    size_error_percentage_acc += (error_width_percentage+error_length_percentage)/2
                    weight_error_percentage_acc += (error_weight_percentage)
                    
                else:
                    st.warning(f"No data found for image {image_title} in the dataset.")

                
                with st.expander(image_title):
                    col1, col2 = st.columns([2,5], gap="large")
                    with col1:
                        st.image(rect_image[:,:,::-1])
                        st.image(area_image[:,:,::-1], "Label: "+str(label))
                        # st.image(roi1[:,:,::-1])
                        # st.image(roi2[:,:,::-1])
                        
                    with col2:
                        components.html(f"""
                            <style>
                                body {{
                                    margin: 0;
                                }}
                                .detail-row {{
                                    display: flex; flex-direction: row; gap: 2rem;
                                }}
                                .margin-bottom {{ margin: 0 0 8px 0; }}
                                .detail-column {{
                                    display: flex; flex-direction: row; align-items: center; letter-spacing: 0.5px; font-family: "Source Sans Pro", sans-serif;
                                }}
                                .highlight {{
                                    padding: 5px 8px; border-radius: 5px;
                                }}
                                .background-font-green {{background: rgba(33, 195, 84, 0.1); color: rgb(23, 114, 51); font-size: 14px; width: 2.5rem; text-align: center;}}
                                .background-font-gray  {{background: rgb(240, 242, 246);      color: rgba(49, 51, 63, 0.8); font-size: 14px; width: 2.5rem; text-align: center;}}
                                .background-font-red   {{background: rgba(255, 43, 43, 0.09); color: rgb(125, 53, 59); font-size: 14px; width: 2.5rem; text-align: center;}}
                                .font-blackblue-1 {{ color: rgb(49, 51, 63); width: 70px;}}
                                .font-blackblue-2 {{ color: rgb(49, 51, 63); width: 160px;}}
                                .font-blackblue-3 {{ color: rgb(49, 51, 63); width: 100px;}}
                                
                            </style>
                            <div class="detail-row margin-bottom">
                                <div class="detail-column"><div class="font-blackblue-1">width:</div>  <div class="highlight background-font-green">{round(actual_width,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">predict:</div><div class="highlight background-font-gray">{round(predict_width,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">error:</div>  <div class="highlight background-font-red">{round(error_width,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">error%:</div> <div class="highlight background-font-red">{round(error_width_percentage,2)}</div></div>
                            </div>
                            <div class="detail-row margin-bottom">
                                <div class="detail-column"><div class="font-blackblue-1">length:</div>  <div class="highlight background-font-green">{round(actual_length,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">predict:</div><div class="highlight background-font-gray">{round(predict_length,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">error:</div>  <div class="highlight background-font-red">{round(error_length,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">error%:</div> <div class="highlight background-font-red">{round(error_length_percentage,2)}</div></div>
                            </div>
                            <div class="detail-row">
                                <div class="detail-column"><div class="font-blackblue-1">weight:</div>  <div class="highlight background-font-green">{round(actual_total_weight,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">predict:</div><div class="highlight background-font-gray">{round(model6,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">error:</div>  <div class="highlight background-font-red">{round(error_weight,2)}</div></div>
                                <div class="detail-column"><div class="font-blackblue-1">error%:</div> <div class="highlight background-font-red">{round(error_weight_percentage,2)}</div></div>
                            </div>
                            <div style="display: flex; flex-direction: row; margin-top: 3rem;">
                                <div style="display: flex; flex-direction: column; margin-right: 4rem;">
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-2">predict total area:</div><div style="width: 3rem;" class="highlight background-font-gray">{round(predict_total_area,2)}</div></div>
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-2">predict kernel area:</div><div style="width: 3rem;" class="highlight background-font-gray">{round(predict_kernel_area,2)}</div></div>
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-2">area%:</div><div style="width: 3rem;" class="highlight background-font-gray">{round(area_percentage,2)}</div></div>
                                </div>
                                <div style="display: flex; flex-direction: column; margin-right: 2rem;">
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-3">model2:</div><div class="highlight background-font-gray">{model2}</div></div>
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-3">model3-1:</div><div class="highlight background-font-gray">{model3_1}</div></div>
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-3">model3-2:</div><div class="highlight background-font-gray">{model3_2}</div></div>
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-3">model5:</div><div class="highlight background-font-gray">{model5}</div></div>
                                </div>
                                <div style="display: flex; flex-direction: column; margin-right: 2rem;">
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-1">result:</div><div class="highlight background-font-gray">{result}</div></div>
                                    <div class="detail-column margin-bottom"><div class="font-blackblue-1">label:</div><div class="highlight background-font-green">{label}</div></div>
                                </div>
                            </div>
                        """, height=350)

    with overall_stat.container():
        if True:
            donut_chart, number_chart = st.columns([2,3], gap="small")
            with donut_chart:
                total = sum(values)
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.7,
                    marker=dict(
                        colors=['rgb(0, 96, 0)', 'rgba(0, 96, 0, 0.2)'],
                        line=dict(color='#ffffff', width=2)
                    ),
                    textinfo='none',
                    sort=False,
                    direction='clockwise',
                    rotation=90,
                    domain=dict(x=[0, 1], y=[0, 1])
                )])
                fig.update_layout(
                    showlegend=False,
                    annotations=[dict(
                        text=f"{values[0]}/{total}",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20, color='rgb(0, 96, 0)', family='Arial'),
                        showarrow=False
                    )],
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    width=150,
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=24),
                    title={
                        'text': "prediction",
                        'x': 0.5,
                        'y': 0.01,
                        'xanchor': 'center',
                        'yanchor': 'bottom',
                        'font': dict(size=14, color='black', family="Source Sans Pro, sans-serif")
                    }
                )
                st.plotly_chart(fig)
            
            with number_chart:
                with st.container(height=70):
                    components.html(f"""
                        <h5 style="text-align: center; margin: 0; color: rgb(49, 51, 63); font-family:Source Sans Pro, sans-serif;font-weight: normal;margin-bottom: 3px;">size error</h5>
                        <h2 style="text-align: center; margin: 0; color: green; font-family:Source Sans Pro, sans-serif;">{round(size_error_percentage_acc / len(images) if len(images) > 0 else 0, 2)}%</h2>
                    """)
                with st.container(height=70):
                    components.html(f"""
                        <h5 style="text-align: center; margin: 0; color: rgb(49, 51, 63); font-family:Source Sans Pro, sans-serif;font-weight: normal;margin-bottom: 3px;">weight error</h5>
                        <h2 style="text-align: center; margin: 0; color: green; font-family:Source Sans Pro, sans-serif;">{round(weight_error_percentage_acc / len(images) if len(images) > 0 else 0, 2)}%</h2>
                    """)
            #end

if __name__ == "__main__":
    app()