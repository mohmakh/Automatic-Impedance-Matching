# Predict using model
frequencies = 1203e06
normalized_frequencies =(frequencies - train_mean['Frequency']) / train_std['Frequency']
Normalized_frequencies = np.array([normalized_frequencies])
predictions = model.predict(Normalized_frequencies)

# Denormalize predictions
Train_std = train_std[['C1', 'C2', 'L']].to_numpy()
Train_mean = train_mean[['C1', 'C2', 'L']].to_numpy()
Predictions = predictions * Train_std + Train_mean

print(Predictions)

loaded_model = keras.models.load_model('Impedance_neural_network.h5')

Norm_predictions = loaded_model.predict(Normalized_frequencies)
# Denormalize predictions
Train_std = train_std[['C1', 'C2', 'L']].to_numpy()
Train_mean = train_mean[['C1', 'C2', 'L']].to_numpy()
Predictions = Norm_predictions * Train_std + Train_mean

print(" The Optimum Impedance matching Network Component Values for 1203MHz Frequency are:")
for i in Predictions:
  print("C1 = ",'{:.2e}'.format(float(Predictions[:,0])))
  print("C2 = ",'{:.2e}'.format(float(Predictions[:,1])))
  print("L = ",'{:.2e}'.format(float(Predictions[:,2])))

