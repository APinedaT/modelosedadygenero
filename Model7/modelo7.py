# -*- coding: utf-8 -*-

""" Importación de las librerias """
import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
# importing cv2 
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go
from tensorflow.keras.utils import to_categorical
from PIL import Image
import plotly.express as px

def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()
    
    pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text='Distribucion de %s' % pd_series.name)
    
    fig.show()

""" Se guarda en una variable el DataSet """
dataset_folder_name = '/home/consultant1/Documents/Personal/ProyectoDeGrado/UTKface_inthewild/archive/utkface_aligned_cropped/UTKFace'

TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198

""" Se genera un diccionario para genero """
""" se omitira el diccionario para raza de dataset """
dataset_dict = {
    'gender_id': {
        0: 'Masculino',
        1: 'Femenino'
    }
}
""" Con el items devuelve cada uno de los elemento en un diccionario como una lista, recorre con un 
arreglo el diccionario ya dado tanto para edad como para raza """
dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())

""" Se genera una función para empezar a extraer los datos del data set 
retorna un pandas con la información de edad, genero y raza """

def analizar_dataset( dataset_path , ext = 'jpg' ) :
    def analizar_info_archivo(path):
        """ se realiza analisis de un solo archivo  """
        try:
            """ dividie el normbre de la ruta en el primer archivo """
            archivo = os.path.split(path)[1]
            """ divide el nombre del archivo separando la extensión """
            archivo = os.path.splitext(archivo)[0]
            """ reemplaza en loss archivos el _ y crea un arreglo """
            edad, genero, raza, _ = archivo.split('_')
        
            return int(edad), dataset_dict['gender_id'][int(genero)]
        except Exception as ex:
            return None, None

    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))

    records = []
    for file in files:
        info = analizar_info_archivo(file)
        #print("info",info)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['archivo'] = files
    df.columns = ['edad', 'genero', 'archivo']
    df = df.dropna()
    
    return df

df = analizar_dataset(dataset_folder_name)
df.head()
#print(df)

def plot_data():
    plot_distribution(df['gender'])

    fig = px.histogram(df, x="age", nbins=20)
    fig.update_layout(title_text='Age distribution')
    fig.show()
    bins = [0, 10, 20, 30, 40, 60, 80, np.inf]
    names = ['<10', '10-20', '20-30', '30-40', '40-60', '60-80', '80+']
    age_binned = pd.cut(df['age'], bins, labels=names)
    plot_distribution(age_binned)

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        #print("p: ",p)
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        #print("train_up_to: ",train_up_to)
        train_idx = p[:train_up_to]
        #print("train_idx: ",train_idx)
        test_idx = p[train_up_to:]
        #print("test_idx: ",test_idx)

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        #print("train_up_to: ",train_up_to)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        #print("train_idx: ",train_idx)
        #print("valid_idx: ",valid_idx)
        
        # converts alias to id
        self.df['gender_id'] = self.df['genero'].map(lambda gender: dataset_dict['gender_alias'][gender])

        self.max_age = self.df['edad'].max()
        print("EDAD maxima ----------------------")
        print(self.df['edad'].max())
        #print("Final")
        #print("train_idx: ",train_idx)
        #print("valid_idx: ",valid_idx)
        #print("test_idx: ",test_idx)

        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Se procesa la imagen para que quede de la forma [192,192,3] ya que esta es la entrada de la red
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, ages, genders =  [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = person['edad']
                gender = person['gender_id']
                file = person['archivo']
                
                im = self.preprocess_image(file)
                ages.append(age / self.max_age)
                if age / self.max_age > 1:
                    print("-----age / self.max_age:",age / self.max_age)
                genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                if len(im.shape)>2:
                    images.append(im)
                
                # yielding condition
                
                if len(images) >= batch_size:
                    #print(np.shape(images))
                    yield np.array(images), [np.array(ages), np.array(genders)]
                    images, ages, genders =  [], [], []

            if not is_training:
                break
                
data_generator = UtkFaceDataGenerator(df)
print("Data generator ",data_generator)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
import tensorflow as tf

class UtkMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains two branches, one for age and other for 
    sex . Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        #print("---x",x)
        return x

    def build_gender_branch(self, inputs, num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        x = Activation("sigmoid", name="gender_output")(x)

        return x

    def build_age_branch(self, inputs):   
        """
        Used to build the age branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.

        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="age_output")(x)

        return x

    def assemble_full_model(self, width, height, ):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        #print("--input_shape",input_shape)

        inputs = Input(shape=input_shape)
        #print("--intpus",inputs)

        age_branch = self.build_age_branch(inputs)
        gender_branch = self.build_gender_branch(inputs)

        model = Model(inputs=inputs,outputs = [age_branch, gender_branch],name="face_net")
        #print("----------Model ",model)
        return model

model = UtkMultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT)

from tensorflow.keras.optimizers import Adam

init_lr = 1e-4
epochs = 2000

opt = Adam(learning_rate=init_lr, decay=init_lr / epochs)


model.compile(optimizer=opt, 
              loss={
                  'age_output': 'mse', 
                  'gender_output': 'binary_crossentropy'},
              loss_weights={
                  'age_output': 4., 
                  'gender_output': 0.1},
              metrics={
                  'age_output': 'mae', 
                  'gender_output': 'accuracy'},run_eagerly=True)

from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)
#print("-------train_gen",train_gen )
#print("------valid_gen",valid_gen)

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

history = model.fit(train_gen,
                    epochs=epochs,
                    steps_per_epoch= 10,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)



#Presicion de genero 
plt.clf()
fig = go.Figure()
fig.add_trace(go.Scatter(
                    y=history.history['gender_output_accuracy'],
                    name='Train'))
fig.add_trace(go.Scatter(
                    y=history.history['val_gender_output_accuracy'],
                    name='Valid'))
fig.update_layout(height=500, 
                  width=700,
                  title='Accuracy for gender feature',
                  xaxis_title='Epoch',
                  yaxis_title='Accuracy')
fig.show()

#Erro absoluto de edad
plt.clf()

fig = go.Figure()
fig.add_trace(go.Scattergl(
                    y=history.history['age_output_mae'],
                    name='Train'))
fig.add_trace(go.Scattergl(
                    y=history.history['val_age_output_mae'],
                    name='Valid'))
fig.update_layout(height=500, 
                  width=700,
                  title='Mean Absolute Error for age feature',
                  xaxis_title='Epoch',
                  yaxis_title='Mean Absolute Error')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scattergl(
                    y=history.history['loss'],
                    name='Train'))
fig.add_trace(go.Scattergl(
                    y=history.history['val_loss'],
                    name='Valid'))
fig.update_layout(height=500, 
                  width=700,
                  title='Overall loss',
                  xaxis_title='Epoch',
                  yaxis_title='Loss')
fig.show()

model.summary()
model.save('model6.h5')

from keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plot_model(model, to_file='model6.png')
img = mpimg.imread('model6.png')

plt.figure(figsize=(40, 30))
plt.imshow(img)

test_batch_size = 128
test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size)
age_pred, gender_pred = model.predict(test_generator, 
                                                           steps=len(test_idx)//test_batch_size,
                                                           verbose= 2)


test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size)
samples = 0
images, age_true, gender_true = [], [], []
for test_batch in test_generator:
    image = test_batch[0]
    labels = test_batch[1]
    
    images.extend(image)
    age_true.extend(labels[0])
    gender_true.extend(labels[1])
    
age_true = np.array(age_true)
gender_true = np.array(gender_true)
gender_true = np.argmax(gender_true,axis=1)
gender_pred = np.argmax(gender_pred,axis=1)


age_true = age_true * data_generator.max_age
age_pred = age_pred * data_generator.max_age
#if age_pred > data_generator.max_age:


from sklearn.metrics import classification_report

cr_gender = classification_report(gender_true, gender_pred, target_names=dataset_dict['gender_alias'].keys())
print(cr_gender)


from sklearn.metrics import r2_score

print('R2 score for age: ', r2_score(age_true, age_pred))