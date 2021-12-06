import streamlit as st
import pandas as pd 
import numpy as np 
import cv2
import os
import tempfile
from PIL import Image
import NN
import Infer
import pickle
from streamlit_drawable_canvas import st_canvas


st.title('Street Smart')
st.write(" ------ ")

SIDEBAR_OPTION_PROJECT_INFO = "About Project"
SIDEBAR_OPTION_DEMO_IMAGE = "Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE]
NOTA = 'NOTA'
BUS_AND_MAR =  'Busines And Marketing Complexes'
RESIDENT = 'Residential'
OFFICES = 'Offices'
PLANTS_FACT_CONST = 'Power Plants and Factories Construction Sites'
HEALTH =  'Healthcare'
EDUCATIONAL =  'Educational'
STORAGE = 'Godowns And Storage'
TRANSPORT = 'Transport'
HOTELS = 'Hotels And Leisure'
BUILDING_OPTIONS = [BUS_AND_MAR, RESIDENT, OFFICES, PLANTS_FACT_CONST, HEALTH, EDUCATIONAL, STORAGE, TRANSPORT, HOTELS, NOTA]
       
def visualize(data1,data2,x,img):
    X = np.array(img).astype(float)/255.0
    X0 = X[:,:,0].T
    X1 = X[:,:,1].T
    X2 = X[:,:,2].T
    X3 = X[:,:,3].T
    WH = []
    WV = []
    X = np.dstack((np.dstack((np.dstack((X0,X1)),X2)),X3))
    data1 = (data1 > (np.mean(data1)+(x*np.std(data1)))).astype(float)
    go = False
    start = -1
    lens = 0
    for i in range(0,data1.shape[1]):
        if np.mean(data1[:,i]) > 0.001:
            if go:
                lens+=1
            else:
                go = True
                lens = 1
                start = i
        else:
            if go and (lens//6) > 0:
                WH.append((start+int((5*lens)/12.00),start+int((7*lens)/12.00)))
                go = False
    if go and (lens//6) > 0:
        WH.append((start,start+lens,lens//6))
        go = False
    go = False
    data2 = (data2 > np.mean(data2)+(x*np.std(data2))).astype(float)
    for i in range(0,data2.shape[0]):
        if np.mean(data1[i,:]) > 0.001:
            if go:
                lens+=1
            else:
                go = True
                lens = 1
                start = i
        else:
            if go and (lens//6) > 0:
                WV.append((start+int((5*lens)/12.00),start+int((7*lens)/12.00)))
                go = False
    if go and (lens//6) > 0:
        WV.append((start,start+lens,lens//6))
        go = False

    T = np.where(data1+data2 > 0.5)
    for i in range(len(T[0])):
        X[T[0][i],T[1][i],0:3] = 0
    # print(len(WH))
    # print(len(WV))
    for w in WH:
        for i in range(0,X.shape[0],35):
            if np.mean(X[i,w[0],0:3]) < 0.1 and (i+20)<X.shape[0] and np.mean(X[i+20,w[0],0:3]) < 0.1:
                X[i:(i+20),w[0]:w[1],0:3] = 1.00
    for w in WV:
        for i in range(0,X.shape[1],35):
            if np.mean(X[w[0],i,0:3]) < 0.1 and (i+20)<X.shape[1] and np.mean(X[w[0],i+20,0:3]) < 0.1:
                X[w[0]:w[1],i:(i+20),0:3] = 1.00
    return np.transpose(X,(1,0,2))

def get_output(df, img):
    dct = {}
    L = []
    dct[NOTA] = [1,0,0,0,0,0,0,0,0,0]
    dct[BUS_AND_MAR] = [0,1,0,0,0,0,0,0,0,0]
    dct[RESIDENT] = [0,0,1,0,0,0,0,0,0,0]
    dct[OFFICES] = [0,0,0,1,0,0,0,0,0,0]
    dct[PLANTS_FACT_CONST] = [0,0,0,0,1,0,0,0,0,0]
    dct[HEALTH] = [0,0,0,0,0,1,0,0,0,0]
    dct[EDUCATIONAL] = [0,0,0,0,0,0,1,0,0,0]
    dct[STORAGE] = [0,0,0,0,0,0,0,1,0,0]
    dct[TRANSPORT] = [0,0,0,0,0,0,0,0,1,0]
    dct[HOTELS] = [0,0,0,0,0,0,0,0,0,1]
    with open('param.soc','rb') as f:
        Para = pickle.load(f)
    for i in range(len(df)):
        l = df.at[i,'left']
        t = df.at[i, 'top']
        w = df.at[i, 'width']
        h = df.at[i, 'height']
        x = l+ (w/2)
        y = t + (h/2)
        traffic, cac = NN.L_model_forward(np.expand_dims(np.array([w,h]+dct[df.at[i,'type']]),axis = 0),Para)
        L.append(Infer.Rect((x,y),(w,h),float(traffic)*1000))
    Res = Infer.Infer(L, img.size)
    TH = Res.TrafficMapH
    TV = Res.TrafficMapV
    y = 1.00+(4/10)
    return visualize(TV,TH,y, img)




def main():
    app_mode = st.sidebar.selectbox("Please select the mode", SIDEBAR_OPTIONS)
    if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.write("The project StreetSmart aims to create a Machine-Learning based model that will be able to\
        suggest possible methods/modes/routes to help suggest a public transport infrastructure for an\
        under-developed or proposed city in the most efficient way possible using libraries, conventional\
        and non-conventional algorithms.")
    elif app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        st.sidebar.write(" ------ ")

        filepath = os.path.join("Screenshot from 2021-07-17 20-45-39.jpg")
        
        st.empty()
        xb = Image.open(filepath)
        xb = xb.resize((600, 300))
        build_type = st.selectbox("Building Type", BUILDING_OPTIONS)
        st.write("Mark the buildings below:")
        canvas_result = st_canvas(
            stroke_width = 2,
            fill_color = "",
            background_image=xb,
            drawing_mode="rect",
            key="canvas",
            width = 600, height = 300, display_toolbar=False)
        bt = st.button("Proceed")
        if 'build_df' not in st.session_state:
            st.session_state['build_df'] = pd.DataFrame(columns = ["left", "top", "width", "height", "type"])
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"]) 
            if len(objects) > len(st.session_state.build_df):
                last = objects[["left", "top", "width", "height"]].iloc[[len(objects) -1]]
                last['type'] = build_type
                st.session_state.build_df = st.session_state.build_df.append(last)
            #st.dataframe(st.session_state.build_df)
        if bt:
            st.write("Output Image:")
            im = get_output(st.session_state.build_df, xb)
            for i in range(len(st.session_state.build_df)):
                x1 = st.session_state.build_df.at[i,'left']
                y1 = st.session_state.build_df.at[i, 'top']
                w = st.session_state.build_df.at[i, 'width']
                h = st.session_state.build_df.at[i, 'height']
                x2 = x1+ w
                y2 = y1 + h
                im = cv2.rectangle(np.float32(im), (x1,y1), (x2,y2), (255, 0,0 ), 2)
            st.image(im, clamp=True, channels = "BGR")


    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
            #upload = st.empty()
            #with upload:
            st.sidebar.info('PRIVACY POLICY: Uploaded images are never saved or stored. They are held entirely within memory for prediction \
                and discarded after the final results are displayed. ')
            f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif', 'JPG'])
            plc = st.empty()
            plc.warning("Please upload the appropriate image")
            if f is not None:
                plc.empty()
                tfile = tempfile.NamedTemporaryFile(delete=True)
                tfile.write(f.read())
                #st.sidebar.write('Please wait for the playlist to be created! This may take up to a few minutes.')
                st.empty()
                xb = Image.open(tfile)
                while(xb.size[0]>600):
                    xb = xb.resize((xb.size[0]//2, xb.size[1]//2))
                build_type = st.selectbox("Building Type", BUILDING_OPTIONS)
                st.write("Mark the buildings below:")
                canvas_result = st_canvas(
                    stroke_width = 2,
                    fill_color = "",
                    background_image=xb,
                    drawing_mode="rect",
                    key="canvas",
                    width = xb.size[0], height = xb.size[1], display_toolbar= False)
                bt = st.button("Proceed")
                if 'build_df1' not in st.session_state:
                    st.session_state['build_df1'] = pd.DataFrame(columns = ["left", "top", "width", "height", "type"])
                if canvas_result.json_data is not None:
                    objects = pd.json_normalize(canvas_result.json_data["objects"]) 
                    if len(objects) > len(st.session_state.build_df1):
                        last = objects[["left", "top", "width", "height"]].iloc[[len(objects) -1]]
                        last['type'] = build_type
                        st.session_state.build_df1 = st.session_state.build_df1.append(last)
                    #st.dataframe(st.session_state.build_df)
                if bt:
                    st.write("Output Image:")
                    im = get_output(st.session_state.build_df1, xb)
                    for i in range(len(st.session_state.build_df1)):
                        x1 = st.session_state.build_df1.at[i,'left']
                        y1 = st.session_state.build_df1.at[i, 'top']
                        w = st.session_state.build_df1.at[i, 'width']
                        h = st.session_state.build_df1.at[i, 'height']
                        x2 = x1+ w
                        y2 = y1 + h
                        im = cv2.rectangle(np.float32(im), (x1,y1), (x2,y2), (255, 0,0 ), 2)
                    st.image(im, clamp=True, channels = "BGR")
        

main()


