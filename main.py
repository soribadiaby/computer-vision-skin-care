import streamlit as st
#import stasm
from PIL import Image
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
import matplotlib.cm as cm

smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


custom_objects = {'dice_coef_loss': dice_coef_loss,
                  'dice_coef': dice_coef}


@st.cache
def load_models():
    age_model = keras.models.load_model("age_model.h5")
    eyebrows_seg_model = keras.models.load_model("eyebrows_seg_model.h5",
                                             custom_objects=custom_objects)
    
    return age_model, eyebrows_seg_model

age_model, eyebrows_seg_model = load_models()


def main():
    
    box1 = 'Quel est l\'âge de votre visage ?'
    box2 = 'Qui est le plus vieux ?'
    box3 = 'Quel est votre type de sourcils ?'
    
    selected_box = st.sidebar.selectbox(
    'Choisissez une option',
    ('Accueil', box1, box2))
    
    if selected_box == 'Accueil':
        welcome() 

    if selected_box == box1:
        photo()
        
    if selected_box == box2:
        compare()
        
        
def welcome():
    st.title('Scoring de visages')
    
    st.subheader('Une application permettant de scorer le niveau de vieillesse')
    
    st.image('images/eyes.jpg', use_column_width=True)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img
    

def get_eyebrows(img, alpha=0.5):
  img = cv2.resize(img, (512, 256))
  img = img / 255
  img2 = eyebrows_seg_model.predict(img[np.newaxis, :])[0]
  gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  a = np.zeros_like(img2)
  a[:,:,0] = gray
  a[:,:,1] = gray
  a[:,:,2] = gray
  (thresh, msk) = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)
  vis = msk * alpha + img
  return vis
  

def get_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = stasm.search_single(gray)
    bottom = 49
    upper = [17, 24]
    left = 0
    right = 12
    
    y1 = min(int(landmarks[upper[0]][1]), int(landmarks[upper[1]][1])) - 10 #marge de 10 pixels
    y2 = int(landmarks[bottom][1])
    
    x1 = int(landmarks[left][0])
    x2 = int(landmarks[right][0])
    
    cropped = img[y1:y2, x1:x2].copy()
    
    return cropped
 
    
def intensite(score):
    if  0 < score < 20:
        intensite = "20-30 ans"
        
    elif 20 <= score < 50:
        intensite = "30-40 ans"
        
    elif 50 <= score < 80:
        intensite = "40-60 ans"
        
    elif 80 <= score:
        intensite = "plus de 60 ans"
        
    return intensite
        
    

def photo():
    st.header("Quel est l'âge de votre visage ?")
    st.text("Le principe est simple : une note de 0 à 100 est attribuée à votre visage,\nplus la note est proche de 0, plus il est jeune")
    uploaded_file = st.file_uploader("Choisissez une photo de visage")
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, 1)
        #cropped = get_crop(img)
        img = cv2.resize(img, (512, 256))
        
        
        score = age_model.predict((img / 255)[np.newaxis, :])[0][0]
        score = round(sigmoid(score).numpy() * 100, 2)
        
            

        last_conv_layer_name = 'block5_conv3'
        heatmap = make_gradcam_heatmap(img[np.newaxis, :], age_model, 
                                       last_conv_layer_name)
        superimposed = save_and_display_gradcam(img, heatmap)
        open_cv_image = np.array(superimposed) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        st.image(open_cv_image, use_column_width=True, channels="BGR")
        st.text("Votre visage a été analysé par notre algorithme.\nLes zones en rouge sont les zones de votre visage présentant des signes de vieillissement \nélevé tels que les cernes et les rides")
          
        st.image(img, use_column_width=True, channels="BGR")
        st.write(f'Le score de votre visage est de {score}/100, vous vous situez donc dans la tranche {intensite(score)}')


def compare():
    st.header("Qui est le plus vieux ?")
    col1, col2 = st.beta_columns(2)
    
    uploaded_file1 = col1.file_uploader("Choisissez une photo de la première personne")
    uploaded_file2 = col2.file_uploader("Choisissez une photo de la deuxième personne")
    if uploaded_file1 is not None and uploaded_file2 is not None:
        col3, col4 = st.beta_columns(2)
        bytes_data1 = uploaded_file1.read()
        nparr1 = np.frombuffer(bytes_data1, np.uint8)
        img1 = cv2.imdecode(nparr1, 1)
        img1 = cv2.resize(img1, (512, 256))
        
        bytes_data2 = uploaded_file2.read()
        nparr2 = np.frombuffer(bytes_data2, np.uint8)
        img2 = cv2.imdecode(nparr2, 1)
        img2 = cv2.resize(img2, (512, 256))
        
        score1 = age_model.predict((img1 / 255)[np.newaxis, :])[0][0]
        score1 = round(sigmoid(score1).numpy() * 100, 2)
        score2 = age_model.predict((img2 / 255)[np.newaxis, :])[0][0]
        score2 = round(sigmoid(score2).numpy() * 100, 2)
        
        
        
        if score1 > score2:
            col1.image(img1, use_column_width=True, channels="BGR")
            col2.image(img2, use_column_width=True, channels="BGR")
            col3.text(f"Cette personne à un score de {score1}/100")
            col3.text(f"Cette personne est donc la plus âgée")
            col4.text(f"Cette personne à un score de {score2}/100")
            col4.text(f"Cette personne est donc la plus jeune")
            
            
        if score2 > score1:
            col1.image(img2, use_column_width=True, channels="BGR")
            col2.image(img1, use_column_width=True, channels="BGR")
            col3.text(f"Cette personne à un score de {score2}/100")
            col3.text(f"Cette personne est donc la plus âgée")
            col4.text(f"Cette personne à un score de {score1}/100")
            col4.text(f"Cette personne est donc la plus jeune")
        
        
if __name__ == "__main__":
    main()