"""
KOUKOSIAS ATHANASIOS-UTH-2026
DATASET:hThe data for this excersise were gathered by using OpenMeteo free API 

"""

# --------------------------------------------------------------
#  config.py  –  change values here, they apply to every model
# --------------------------------------------------------------

#model hyperparameters
HIDDEN_SIZE = 128       #LSTM / GRU hidden units per layer
NUM_LAYERS = 1        #number of stacked recurrent layers
LEARNING_RATE = 1e-3     #adam learning rate
EPOCHS = 300      #training epochs per model
SEQ_LEN = 6        #look-back window in months
CLIP_GRAD = 5.0      #gradient clipping (max norm)

#data
TRAIN_YEARS = list(range(2018, 2025))   #2018 – 2024 inclusive
PREDICT_YEAR = 2025
LAT = 39.3637   #Karditsa, Greece
LON = 21.9214
CACHE_CSV = "weather_cache.csv"       #set to None to always re-fetch(no need, really...)

#misc
SEED = 42
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]