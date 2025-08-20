
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_DIR = "models"
MODEL_FILENAME = "model.pkl"
CALIBRATED = True  # Platt scaling via CalibratedClassifierCV
DEFAULT_TARGET = "default_payment_next_month"
DEFAULT_DATA_PATH = "data/default_of_credit_card_clients.csv"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
CSV_FALLBACK_URL = "https://raw.githubusercontent.com/selva86/datasets/master/DefaultCreditCardClients.csv"
