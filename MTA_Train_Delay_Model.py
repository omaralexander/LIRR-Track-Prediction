from sklearn.preprocessing 	 import MinMaxScaler
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from collections import Iterable
from random 	 import randint
from numpy 		 import array
from numpy 		 import argmax
from numpy 		 import array_equal
import os
import pandas as pd


n_features 	= 4550 + 1 	#we set this value to b greater than any value that would come up
n_steps_in 	= 2 		#sequence of values that we are predicting on
n_steps_out = 1 		#how many values out are we predicting 
data_count 	= 2114 - n_steps_in
scaler 		= MinMaxScaler(feature_range=(0,n_features - 1)) #optional: If we want to scale down are values we would use this

# function to save our trained model  
def save_model(model):
	model_json = model.to_json()
	with open("model.json","w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")
	print("Saved model to disk")

#flatten our data list so we can feed it to become categorized
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, basestring):
            for x in flatten(item): yield x
        else: yield item
 
# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	
	filename 		= 'mta_data.csv'
	raw_data		= pd.read_csv(filename)
	station_filter 	= raw_data['STATION']=='34 ST-HERALD SQ'
	raw_data 		= raw_data[station_filter]['ENTRY_DELTA'].dropna().astype(int)
	X1, X2, y 		= list(), list(), list()

	for i in range(0,n_samples,3):

		source = raw_data[i:i+n_in+1].tolist()
		
		#uncomment this if we are using the scaler instead
		#source = scaler.fit_transform(array(source).reshape(-1,1))
		#source = list(flatten(source.tolist()))

		# define padded target sequence
		target = [source[-1]]

		#remove the last value since that is what we are predicting
		source.pop()

		# create padded input target sequence
		target_in = [0] + [target[0]]
		
		print(source)

		# encode
		src_encoded  = to_categorical(source,    num_classes=cardinality)
		tar_encoded  = to_categorical(target,    num_classes=cardinality)
		tar2_encoded = to_categorical(target_in, num_classes=cardinality)

		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)

	return array(X1), array(X2), array(y)
 
# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs 	= Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states 	= [state_h, state_c]
	# define training decoder
	decoder_inputs 	= Input(shape=(None, n_output))
	decoder_lstm 	= LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense 	= Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states 	= [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model 	= Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model
 
# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	print(one_hot_decode(state))
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for _ in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):

	decoded = [argmax(vector) for vector in encoded_seq]
	#OPTIONAL: if scaler is on uncomment below
	#convert 	= array(decoded).reshape(-1,1)
	#first 		= scaler.inverse_transform(convert)
	#return scaler.inverse_transform(convert)
 	return decoded

# define model
train, infenc, infdec = define_models(n_features, n_features, 100)
train.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc','mse'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, data_count)
print(X1.shape,X2.shape,y.shape)

# train model
train.fit([X1, X2], y, epochs=100) # 100 is good

#sequence we want to predict
numbers = [95,145]
sample 	= list()

#Prepare the numbers to be predicted by converting them 
for i in numbers:
	test_values = array([i]).reshape(-1,1)
	categorize 	= to_categorical(test_values,num_classes=n_features)
	sample.append(categorize)

sample = array(sample)
target = predict_sequence(infenc, infdec,X1[600].reshape(1,2,4551), n_steps_out, n_features)
#target = predict_sequence(infenc, infdec, sample.reshape(1,2,4551), n_steps_out, n_features)
print('X=%s, yhat=%s' % (one_hot_decode(sample), one_hot_decode(target)))

# If we want to get all the test results of our data 
#for i in range(1,len(X1)):

	#X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, i)

#	target = predict_sequence(infenc, infdec, X1[i].reshape(1,2,4551), n_steps_out, n_features)
#	print('X=%s y=%s, yhat=%s, index=%s' % (one_hot_decode(X1[i]), one_hot_decode(y[i]), one_hot_decode(target),i))



