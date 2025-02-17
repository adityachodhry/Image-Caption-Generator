from keras.preprocessing.text import Tokenizer # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.utils import pad_sequences # type: ignore
from keras.applications.xception import Xception # type: ignore
from keras.models import load_model # type: ignore
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# Assume you have a function for color detection
def detect_colors(image):
    # Your color detection code here
    # For example, let's say it returns a list of dominant colors
    dominant_colors = ['red', 'green', 'blue']
    return dominant_colors

# Assume you have a function for image captioning
def generate_caption(image):
    # Your image captioning code here
    # For example, let's say it returns a generated caption
    caption = 'A beautiful scene with a red car'
    return caption

import cv2
from PIL import Image

# Load the image
image_path = os.path.abspath('C:\Projects\Image Caption Generator\Flicker8k_Dataset\72 218201_e0e9c7d65b.jpg')
image = cv2.imread(image_path)

# Perform color detection
detected_colors = detect_colors(image)
# Generate a caption
generated_caption = generate_caption(image)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# from keras.preprocessing.text import Tokenizer
# from keras.utils import pad_sequences
# from keras.applications.xception import Xception
# from keras.models import load_model
# from pickle import load
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import pandas as pd
# import argparse
# import os

# # Function for color detection
# def detect_colors(image):
#     # ... (your existing color detection code)
#     dominant_colors = ['red', 'green', 'blue']  # Replace this with your actual color detection logic
#     return dominant_colors

# # Your existing code for image captioning
# def generate_caption(model, tokenizer, photo, max_length):
#     # ... (your existing caption generation code)
#     return generated_caption

# def main():
#     # ... (your existing code for argument parsing and loading models)

#     # Color Detection Code
#     img = cv2.imread(img_path)
#     dominant_colors = detect_colors(img)

#     # Image Captioning Code
#     max_length = 32
#     tokenizer = load(open("tokenizer.p", "rb"))
#     model = load_model('models/model_9.h5')
#     xception_model = Xception(include_top=False, pooling="avg")

#     photo = extract_features(img_path, xception_model)
#     img = Image.open(img_path)

#     generated_caption = generate_caption(model, tokenizer, photo, max_length)

#     # Displaying Image and Results
#     plt.imshow(img)
#     plt.show()

#     # Merging Results
#     merged_result = {
#         'colors': dominant_colors,
#         'caption': generated_caption
#     }

#     print("Merged Result:")
#     print(merged_result)

# if __name__ == "__main__":
#     main()




# Color Detection Code!

import cv2
import numpy as np
import pandas as pd
import argparse

#Creating argument parser to take image path from command line
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

#Reading the image with opencv
img = cv2.imread(img_path)

#declaring global variables (are used later on)
clicked = False
r = g = b = xpos = ypos = 0

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

#function to get x,y coordinates of mouse double click
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
       
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_function)

while(1):

    cv2.imshow("image",img)
    if (clicked):
   
        #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        #Creating text string to display( Color name and RGB values )
        text = getColorName(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
        
        #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

        #For very light colours we will display text in black colour
        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            
        clicked=False

    #Break the loop when user hits 'esc' key    
    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()


# Merge the color information and caption
merged_result = {
    'colors': detected_colors,
    'caption': generated_caption
}



# GUI 

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore


st.title('Image Caption Dashboard')

stocks = ("GOOG", "APPL", "MSFT", "TSLA")
selected_stocks = st.selectbox("Select Stock for prediction", stocks)
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

st.title('Image Upload and Display')

# Upload an image file
uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # You can perform further processing with the image or extract features as needed.
    # For example, you might want to use a machine learning model to analyze the image.

    # If you want to extract features from the image, you can convert it to a NumPy array.
    image_array = np.array(image)

    # Display the extracted features
    st.write("Image Features:")
    st.write(image_array)







    
