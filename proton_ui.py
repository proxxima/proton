from PIL import Image
from torchvision import transforms
import torch
import streamlit as st

st.title('Proton')
st.write("")

file_up = st.file_uploader('upload image here' , type = 'jpg' , 'jpeg' , 'png')


def predict(image):
    
    resnet = #import_path_here
    img = Image.open(image)
    batch_t = torch.unsqueeze(img,0)
    out = resnet(batch_t)

    with open('classes.txt') as f :
        
        class = [line.strip() for line in f.readlines()] 
    
    prob = torch.nn.functional.torch.softmax(out , dim = 1)[0] *100
    _ , indices = torch.sort (out , descending= True)
    return [(class[idx] , prob[idx].item())] for idx in indices [0][:5]]


if file_up in not None:

    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image' , use_column_width = True)
    st.write("")
    st.write("Processing....")
    labels = predict(file_up)

    for i in labels:
        st.write("Prediction (index,name)" , i[0] , ",  Score: ", i[1])  


