from keras import models
import coremltools as ct

#Load the saved model
model_name = '3-conv-64-nodes-1-dense-512-1669611358'
keras_model = models.load_model(model_name)

labels = ['A',
 'B',
 'BA',
 'BE/BI',
 'BO/BU',
 'D',
 'DA',
 'DE/DI',
 'DO/DU',
 'E/I',
 'G',
 'GA',
 'GE/GI',
 'GO/GU',
 'H',
 'HA',
 'HE/HI',
 'HO/HU',
 'K',
 'KA',
 'KE/KI',
 'KO/KU',
 'L',
 'LA',
 'LE/LI',
 'LO/LU',
 'M',
 'MA',
 'ME/MI',
 'MO/MU',
 'N',
 'NA',
 'NE/NI',
 'NG',
 'NGA',
 'NGE/NGI',
 'NGO/NGU',
 'NO/NU',
 'O/U',
 'P',
 'PA',
 'PE/PI',
 'PO/PU',
 'R',
 'RA',
 'RE/RI',
 'RO/RU',
 'S',
 'SA',
 'SE/SI',
 'SO/SU',
 'T',
 'TA',
 'TE/TI',
 'TO/TU',
 'W',
 'WA',
 'WE/WI',
 'WO/WU',
 'Y',
 'YA',
 'YE/YI',
 'YO/YU']

# Define the input type as image, 
# set pre-processing parameters to normalize the image 
# to have its values in the interval [-1,1] 
# as expected by the mobilenet model
image_input = ct.ImageType(name="conv2d_input", shape=(1, 80, 80, 1,), color_layout=ct.colorlayout.GRAYSCALE, bias=[-1], scale=1/255.0)

# set class labels
classifier_config = ct.ClassifierConfig(labels)

# Convert the model using the Unified Conversion API to an ML Program
model = ct.convert(
    keras_model, 
    convert_to="mlprogram",
    inputs=[image_input], 
    classifier_config=classifier_config,)
#entering metadata
# coreml_model.author = 'Ronnel Matthew Robles'
# coreml_model.license = 'MIT'
# coreml_model.short_description = 'Handwritten Baybayin recognition with a 3 layer network'
# coreml_model.input_description['image'] = '80x80 grayscaled pixel values between 0-1'
# coreml_model.save('SimpleBaybayinRecognition.mlmodel')
# print(coreml_model)

# Set feature descriptions (these show up as comments in XCode)
model.input_description["conv2d_input"] = '80x80 grayscaled pixel values between 0-1'
model.short_description = 'Handwritten Baybayin recognition with a 3 layer network'

# Set model author name
model.author = 'Ronnel Matthew Robles'

# Set the license of the model
model.license = "MIT"

# Set a short description for package UI
model.output_description["classLabel"] = "Most likely character category"

# Set a version for the model
model.version = "1.0"

model.save('SimpleBaybayinRecognition.mlpackage')