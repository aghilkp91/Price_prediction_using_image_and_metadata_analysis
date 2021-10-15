import emoji
import string
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
import pandas as pd
import os
import DeepImageUtils as IU

image_features_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data/total_features.npz'
data_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data'
images_path = "/ceph/csedu-scratch/project/akaradathodi/Images/"
# images_path = os.path.abspath(data_path + '/../../') + '/Images/'

def give_emoji_free_text(text):
    # return emoji.get_emoji_regexp().sub(r'', text)
    pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"
                         u"\U0001F300-\U0001F5FF"
                         u"\U0001F680-\U0001F6FF"
                         u"\U0001F1E0-\U0001F1FF"
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         "]+", flags=re.UNICODE)

    return pattern.sub(r'', text)

def remove_punchuations(sentence):
    # return text.translate(str.maketrans('', '', string.punctuation)).strip()
    regular_punct = list(string.punctuation)

    for punc in regular_punct:
        if punc in sentence:
            sentence = sentence.replace(punc, ' ')

    return sentence.strip()

def remove_contract_words(text):
    try:
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
    except Exception as exp:
        print(exp)
    return text

STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def remove_stopwords(text):
    # return ' '.join([word for word in text if word not in stopwords.words('english')])
    return ' '.join(e for e in text.split() if e not in STOPWORDS)

def process_text(data, cols):
    # print(data.columns.values.tolist())
    for col in cols:

        processed_data = []

        for sentence in data[col].values:
            sent = remove_contract_words(sentence)
            # sent = sentence
            try:
                sent = give_emoji_free_text(sent)
                sent = remove_punchuations(sent)
                sent = remove_stopwords(sent)
                sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
                sent = re.sub("\s+", " ", sent)
                sent = sent.lower().strip()
            except Exception as exp:
                print(exp)
            # if col == 'title':
            #     print(sent)
            processed_data.append(sent)

        data[col] = processed_data

    return data


def process_category(data):
    for i in range(3):

        def get_part(x):

            if type(x) != str:
                return np.nan

            parts = x.split('/')

            if i >= len(parts):
                return np.nan
            else:
                return parts[i]

        field_name = 'category_' + str(i)

        data[field_name] = data['category'].apply(get_part)

    return data


def preprocess(data):
    data = process_text(data, ['name', 'item_description', 'category'])

    data = process_category(data)

    # data = get_features(data)

    data.fillna({'brand_name': ' ', 'category_0': 'other', 'category_1': 'other', 'category_2': 'other'}, inplace=True)

    # concat columns
    data['name'] = data['name'] + ' ' + data['brand_name'] + ' ' + data['category_name']
    data['text'] = data['name'] + ' ' + data['item_description']

    # drop columns which are not required
    data = data.drop(columns=['brand_name', 'item_description', 'category_name'], axis=1)

    return data


# one hot encoding of category names
def get_ohe(X_train, X_test, col_name):
    """
    Get one hot encoded features
    """
    vect = CountVectorizer()
    tr_ohe = vect.fit_transform(X_train[col_name].values)
    te_ohe = vect.transform(X_test[col_name].values)

    return tr_ohe, te_ohe


# one hot encoding of category names
def get_vohe(X_train, X_validation, X_test, col_name):
    """
    Get one hot encoded features
    """
    vect = CountVectorizer()
    tr_ohe = vect.fit_transform(X_train[col_name].values)
    vl_ohe = vect.transform(X_validation[col_name].values)
    te_ohe = vect.transform(X_test[col_name].values)

    return tr_ohe, vl_ohe, te_ohe


# tfidf word embeddings
def get_text_encodings(X_train, X_test, col_name, min_val, max_val):
    """
    Get TFIDF encodings with max_features capped at 1M
    """
    vect = TfidfVectorizer(min_df=10, ngram_range=(min_val, max_val), max_features=1000000)
    tr_text = vect.fit_transform(X_train[col_name].values)
    te_text = vect.transform(X_test[col_name].values)

    return tr_text, te_text


# tfidf word embeddings
def get_vtext_encodings(X_train, X_validation, X_test, col_name, min_val, max_val):
    """
    Get TFIDF encodings with max_features capped at 1M
    """
    vect = TfidfVectorizer(min_df=10, ngram_range=(min_val, max_val), max_features=1000000)
    tr_text = vect.fit_transform(X_train[col_name].values)
    vl_text = vect.transform(X_validation[col_name].values)
    te_text = vect.transform(X_test[col_name].values)

    return tr_text, vl_text, te_text


def generate_encodings(X_train, X_test):
    """
    Get encodings for all the features. Scale and normalize the numerical features. Stack the encoded features horizontally.
    """
    # print('1st phase')
    X_train = X_train.fillna('others')
    X_test = X_test.fillna('others')
    # X_train = process_text(X_train,
    #                        ['title', 'actual_color', 'product_details', 'complete_the_look', 'specifications', 'care_instruction_0', 'care_instruction_1'])

    # X_test = process_text(X_test,
    #                        [ 'title', 'actual_color','product_details', 'complete_the_look', 'specifications', 'care_instruction_0', 'care_instruction_1'])

    # print('2nd phase')

    tr_ohe_brand, te_ohe_brand = get_ohe(X_train=X_train, X_test=X_test, col_name="brand")
    # tr_ohe_dominant_material, te_ohe_dominant_material = get_ohe(X_train, X_test, 'dominant_material')
    tr_ohe_dominant_color, te_ohe_dominant_color = get_ohe(X_train, X_test, 'dominant_color')
    tr_ohe_product_type, te_ohe_product_type = get_ohe(X_train=X_train, X_test=X_test, col_name='product_type')
    tr_ohe_gender, te_ohe_gender = get_ohe(X_train=X_train, X_test=X_test, col_name='gender')
    tr_ohe_category_1, te_ohe_category_1 = get_ohe(X_train=X_train, X_test=X_test, col_name='category_1')
    tr_ohe_category_0, te_ohe_category_0 = get_ohe(X_train=X_train, X_test=X_test, col_name='category_0')


    # print('3rd phase')
    tr_trans = csr_matrix(
        pd.get_dummies(X_train[['cheap_brands', 'expensive_brands', 'luxurious_brands']], sparse=True).values)
    te_trans = csr_matrix(
        pd.get_dummies(X_test[['cheap_brands', 'expensive_brands', 'luxurious_brands']], sparse=True).values)

    # print('4th phase')
    tr_title, te_title = get_text_encodings(X_train, X_test, 'title', 1, 1)
    tr_product_details, te_product_details = get_text_encodings(X_train, X_test, 'product_details', 1, 2)
    tr_complete_the_look, te_complete_the_look = get_text_encodings(X_train, X_test, 'complete_the_look', 1, 2)
    tr_specifications, te_specifications = get_text_encodings(X_train, X_test, 'specifications', 1, 2)
    tr_ohe_actual_color, te_ohe_actual_color = get_text_encodings(X_train, X_test, 'actual_color', 1, 1)
    tr_ohe_care_instruction_0, te_ohe_care_instruction_0 = get_text_encodings(X_train,X_test, 'care_instruction_0', 1, 1)
    tr_ohe_care_instruction_1, te_ohe_care_instruction_1 = get_text_encodings(X_train, X_test, 'care_instruction_1', 1, 1)
    tr_ohe_dominant_material, te_ohe_dominant_material = get_text_encodings(X_train, X_test, 'dominant_material', 1, 1)
    print('Last phase')
    train_data = hstack((tr_ohe_category_0, tr_ohe_category_1, tr_ohe_brand, tr_ohe_dominant_material, tr_ohe_actual_color,
                         tr_ohe_dominant_color, tr_ohe_product_type, tr_trans, tr_ohe_gender,
                         tr_ohe_care_instruction_0, tr_title, tr_product_details, tr_ohe_care_instruction_1,
                         tr_complete_the_look, tr_specifications)).tocsr().astype('float32')

    test_data = hstack((te_ohe_category_0, te_ohe_category_1, te_ohe_brand, te_ohe_dominant_material, te_ohe_actual_color,
                        te_ohe_dominant_color, te_ohe_product_type, te_trans, te_ohe_gender,
                        te_ohe_care_instruction_0, te_title, te_product_details, te_ohe_care_instruction_1,
                        te_complete_the_look, te_specifications)).tocsr().astype('float32')

    return train_data, test_data

def get_test_and_validation_image_features(X_train, X_validation):
    loaded_features = np.load(image_features_path, allow_pickle=True)
    features = list(loaded_features['arr_0'])
    feature_dict = dict()
    for feature in features:
        feature_dict[feature[2]] = [feature[1]]
    train_features_in_order = []
    validation_features_in_order = []
    for index, row in X_train.iterrows():
        train_features_in_order.insert(index, feature_dict[row.uniq_id][0])
    for index, row in X_validation.iterrows():
        validation_features_in_order.insert(index, feature_dict[row.uniq_id][0])
    return train_features_in_order, validation_features_in_order

def add_features_to_testdf(test_df):
    test_features_in_order = []
    for index, row in test_df.iterrows():
        img = images_path + row['uniq_id'] + '_0.jpg'
        # print(img)
        feature_vector = IU.CreateImageFeaturesVector(img)
        test_features_in_order.insert(index, feature_vector)
    return test_features_in_order

def generate_encodings_for_features(X_train, X_validation, X_test):
    """
        Get encodings for all the features. Scale and normalize the numerical features. Stack the encoded features horizontally.
    """
    X_train = X_train.fillna('others')
    X_test = X_test.fillna('others')
    X_validation = X_validation.fillna('others')
    X_train = process_text(X_train,
                           ['title', 'actual_color', 'product_details', 'complete_the_look', 'specifications', 'care_instruction_0', 'care_instruction_1'])

    X_validation = process_text(X_validation,
                                ['title', 'actual_color', 'product_details', 'complete_the_look', 'specifications', 'care_instruction_0', 'care_instruction_1'])

    X_test = process_text(X_test,
                          ['title', 'actual_color', 'product_details', 'complete_the_look', 'specifications', 'care_instruction_0', 'care_instruction_1'])

    tr_ohe_brand, vl_ohe_brand, te_ohe_brand = get_vohe(X_train, X_validation, X_test, 'brand')
    # tr_ohe_dominant_material, te_ohe_dominant_material = get_vohe(X_train, X_test, 'dominant_material')
    tr_ohe_dominant_color, vl_ohe_dominant_color, te_ohe_dominant_color = get_vohe(X_train, X_validation, X_test, 'dominant_color')
    tr_ohe_product_type, vl_ohe_product_type, te_ohe_product_type = get_vohe(X_train, X_validation, X_test, 'product_type')
    tr_ohe_gender, vl_ohe_gender, te_ohe_gender = get_vohe(X_train, X_validation, X_test, 'gender')
    tr_ohe_category_1, vl_ohe_category_1, te_ohe_category_1 = get_vohe(X_train, X_validation, X_test, 'category_1')
    tr_ohe_category_0, vl_ohe_category_0, te_ohe_category_0 = get_vohe(X_train, X_validation, X_test, 'category_0')

    tr_trans = csr_matrix(
        pd.get_dummies(X_train[['cheap_brands', 'expensive_brands', 'luxurious_brands']], sparse=True).values)
    vl_trans = csr_matrix(
        pd.get_dummies(X_validation[['cheap_brands', 'expensive_brands', 'luxurious_brands']], sparse=True).values)
    te_trans = csr_matrix(
        pd.get_dummies(X_test[['cheap_brands', 'expensive_brands', 'luxurious_brands']], sparse=True).values)

    tr_title, vl_title, te_title = get_vtext_encodings(X_train, X_validation, X_test, 'title', 1, 1)
    tr_product_details, vl_product_details, te_product_details = get_vtext_encodings(X_train, X_validation, X_test, 'product_details', 1, 2)
    tr_complete_the_look, vl_complete_the_look, te_complete_the_look = get_vtext_encodings(X_train, X_validation, X_test, 'complete_the_look', 1, 2)
    tr_specifications, vl_specifications, te_specifications = get_vtext_encodings(X_train, X_validation, X_test, 'specifications', 1, 2)
    tr_ohe_dominant_material, vl_ohe_dominant_material, te_ohe_dominant_material = get_vtext_encodings(X_train, X_validation, X_test, 'dominant_material', 1, 1)
    tr_ohe_actual_color, vl_ohe_actual_color, te_ohe_actual_color = get_vtext_encodings(X_train, X_validation, X_test, 'actual_color', 1, 1)
    tr_ohe_care_instruction_1, vl_ohe_care_instruction_1, te_ohe_care_instruction_1 = get_vtext_encodings(X_train, X_validation, X_test, 'care_instruction_1', 1, 1)
    tr_ohe_care_instruction_0, vl_ohe_care_instruction_0, te_ohe_care_instruction_0 = get_vtext_encodings(X_train, X_validation, X_test, 'care_instruction_0', 1, 1)


    # Add image feature vectors also to hstack
    train_features_in_order, validation_features_in_order = get_test_and_validation_image_features(X_train, X_validation)
    test_features_in_order = add_features_to_testdf(X_test)

    train_data = hstack((train_features_in_order, tr_ohe_category_0, tr_ohe_category_1, tr_ohe_brand, tr_ohe_dominant_material,
                         tr_ohe_dominant_color, tr_ohe_actual_color, tr_ohe_product_type, tr_trans, tr_ohe_gender,
                         tr_ohe_care_instruction_0, tr_title, tr_product_details, tr_ohe_care_instruction_1,
                         tr_complete_the_look, tr_specifications)).tocsr().astype('float32')
    # train_data = np.concatenate((train_data, train_features_in_order))

    validation_data = hstack((validation_features_in_order, vl_ohe_category_0, vl_ohe_category_1, vl_ohe_brand, vl_ohe_dominant_material,
                         vl_ohe_dominant_color, vl_ohe_actual_color, vl_ohe_product_type, vl_trans, vl_ohe_gender,
                         vl_ohe_care_instruction_0, vl_title, vl_product_details, vl_ohe_care_instruction_1,
                         vl_complete_the_look, vl_specifications)).tocsr().astype('float32')

    # validation_data = np.concatenate((validation_features_in_order, validation_data))

    test_data = hstack((test_features_in_order, te_ohe_category_0, te_ohe_category_1, te_ohe_brand, te_ohe_dominant_material,
                        te_ohe_dominant_color, te_ohe_actual_color, te_ohe_product_type, te_trans, te_ohe_gender,
                        te_ohe_care_instruction_0, te_title, te_product_details, te_ohe_care_instruction_1,
                        te_complete_the_look, te_specifications)).tocsr().astype('float32')
    # test_data = np.concatenate((test_features_in_order, validation_data))

    return train_data, validation_data, test_data


