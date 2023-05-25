import streamlit as st
import cv2
import tempfile
import os
import detector as d

# Set page title and favicon
st.set_page_config(page_title='Custom Object Detection', page_icon=':camera:')

# Set page layout to wide
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Page title and description
st.title('Custom Object Detection')
with st.expander('Classes'):
    st.markdown('car, threewheel, bus, truck, motorbike, van')
    st.markdown('---')

# File uploader and prediction button
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Read the image using cv2.imread()
    img = cv2.imread(temp_file.name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(img, use_column_width=True)
    st.markdown('---')

    if st.button('Predict'):
        # Perform object detection
        model_path = 'weights/best.pt'
        model = d.load_model(model_path)
        output_img = d.predict(img, model)

        # Display the predicted image
        st.image(output_img, use_column_width=True)
        st.markdown('---')

    # Remove the temporary file
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)
