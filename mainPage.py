import streamlit as st
from PIL import Image
import model 

st.set_page_config(page_title="Aerial Drone segmentation")
st.title("Aerial Drone segmentation")

# Developer information
developers = [
    {"name": "Ahmed Adel", "linkedin": "https://www.linkedin.com/in/ahmed23adel/"},
    {"name": "Nour Elding Hammido", "linkedin": "https://www.linkedin.com/in/nourhamedo/"},
    {"name": "Jihad", "linkedin": "https://www.linkedin.com/in/jihad-mahmoud-5804ab18b/"}
]

# Display developer information with links in the sidebar
st.sidebar.markdown("Developed by:")
for developer in developers:
    st.sidebar.markdown(f"[{developer['name']}]({developer['linkedin']})")
st.text("Pick an example image")
examples_img_names = ("000", "001", "002", "003", "004", "005", "006")
image_example_name = st.selectbox("Example Image: ",examples_img_names)

def show_example_image(image_example_name = "000"):
    if image_example_name not in examples_img_names:
        raise Exception("Image not found")
    img_orig_url = r'imgs/{}_orig.jpg'.format(image_example_name) 
    img_ground_truth_url = r'imgs/{}_groundTruth.jpg'.format(image_example_name) 
    img_mask_pred_url = r'imgs/{}_maskPred.png'.format(image_example_name) 
    img_orig = Image.open(img_orig_url)
    mask_ground_truth = Image.open(img_ground_truth_url)
    img_mask_pred = Image.open(img_mask_pred_url)
    st.text("Original image")
    st.image(img_orig, width=600)
    st.text("Mask image original")
    st.image(mask_ground_truth, width=600)
    st.text("Mask image predicted")
    st.image(img_mask_pred, width=600)
show_example_image(image_example_name)


st.header("Now it's time to try your image")
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
if uploaded_file is not None:
    if uploaded_file.type.split('/')[1] in ['jpg', 'jpeg', 'png']:
        st.write("File uploaded successfully!")
        st.image(uploaded_file, caption='Uploaded Image', width=500)
        image = Image.open(uploaded_file)
        st.subheader('Model prediction')
        mask = model.predict_img(image)
        st.image(mask, caption='mask Image predicted', width=500)




st.header("Model details")
st.text("""
I used PyTorch lightning for this project

Model details:
We used DeepLabV3Plus and resnet34 for feature extraction,\n and for optimization I used Adam, and cross entropy loss.

DeepLabV3Plus: 
""")
        
model_arch = Image.open(r'model_arch.png')
st.image(model_arch, caption='Model architecture', width=500)

st.text("""
I used albumentations for transforms, it generates sunny, rainy effect and many more

For metric i used Accurate, jaccard index(IOS)
""")  

# st.image(img, width=600)



