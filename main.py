import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import GUI

trix = {'চুলকানি': 'itching', 'চামড়া ফুসকুড়ি': 'skin_rash', 'চামড়ার বিশেষ ক্ষত': 'nodal_skin_eruptions',
        'হাঁচি': 'continuous_sneezing', 'কাঁপুনি': 'shivering', 'ঠান্ডা অনুভব': 'chills',
        'গিরা ব্যাথা': 'joint_pain',
        'পেটে ব্যাথা': 'stomach_pain', 'পেটে গ্যাস': 'acidity', 'জিহব্বায় ঘা': 'ulcers_on_tongue',
        'হাত পা ব্যাথা': 'muscle_wasting', 'বমি': 'vomiting', 'প্রসাবে জ্বালাপোড়া': 'burning_micturition',
        'প্রসাবের রঙ পরিবর্তন': 'spotting_urination', 'ক্লান্ত অনুভব': 'fatigue',
        'ওজন বৃদ্ধি': 'weight_gain',
        'হতাশা': 'anxiety', 'হাত পা ঠান্ডা': 'cold_hands_and_feets', 'মনে অবস্থার পরিবর্তন': 'mood_swings',
        'ওজন কমা': 'weight_loss', 'অস্থিরতা': 'restlessness', 'ঘন ঘন ঘুম': 'lethargy',
        'জিহব্বায় সাদা দাগ': 'patches_in_throat', 'অনিয়মিত চিনির পরিমাণ': 'irregular_sugar_level',
        'কাশি': 'cough',
        'খুব জ্বর': 'high_fever', 'চোখের নীচে দাগ': 'sunken_eyes', 'নিঃশ্বাস নিতে সমস্যা': 'breathlessness',
        'ঘাম': 'sweating', 'পানিশূন্যতা': 'dehydration', 'বদহজম': 'indigestion', 'মাথা ব্যাথা': 'headache',
        'হলুদ ত্বক': 'yellowish_skin', 'প্রসাব গাঢ়': 'dark_urine', 'বমি বমি ভাব': 'nausea',
        'খেতে অনীহা': 'loss_of_appetite', 'চোখের পাশে ব্যাথা': 'pain_behind_the_eyes',
        'কোমর ব্যাথা': 'back_pain',
        'কোষ্ঠকাঠিন্য': 'constipation', 'পেটের মধ্যখানে ব্যাথা': 'abdominal_pain', 'ডায়রিয়া': 'diarrhoea',
        'হালকা জ্বর': 'mild_fever', 'হলুদ প্রসাব': 'yellow_urine', 'হলুদ চোখ': 'yellowing_of_eyes',
        'যকৃৎ ব্যাথা': 'acute_liver_failure', 'তরল রক্ত': 'fluid_overload',
        'পেট ফোলাভাব': 'swelling_of_stomach',
        'গলার একপাশে ফোলা': 'swelled_lymph_nodes', 'ভিতরে খারাপ লাগা': 'malaise',
        'চোখে ঘোলা দেখা': 'blurred_and_distorted_vision', 'থুঃথুঃ': 'phlegm',
        'গলা জ্বালাপোড়া': 'throat_irritation',
        'চোখ লাল': 'redness_of_eyes', 'সাইনাসের সমস্যা': 'sinus_pressure', 'সর্দি': 'runny_nose',
        'নাকে সুড়সুড়ি ভাব': 'congestion', 'বুকে ব্যাথা': 'chest_pain', 'হাত পায়ে দুর্বলতা': 'weakness_in_limbs',
        'বুক ধড়ফড়': 'fast_heart_rate', 'মল ত্যাগে ব্যাথা': 'pain_during_bowel_movements',
        'নিতম্ব অঞ্চলে ব্যাথা': 'pain_in_anal_region', 'রক্তাক্ত মল': 'bloody_stool',
        'মলদ্বারে জ্বালাপোড়া': 'irritation_in_anus', 'ঘাড়ে ব্যাথা': 'neck_pain', 'মাথা ঘোরা': 'dizziness',
        'পেশীতে ব্যাথা': 'cramps', 'আঘাতের ক্ষত': 'bruising', 'মেদ বেড়ে যাওয়া': 'obesity',
        'পা ফোলা': 'swollen_legs', 'রক্তনালীতে সমস্যা': 'swollen_blood_vessels',
        'ফোলা চোখ এবং মুখ': 'puffy_face_and_eyes', 'থাইরয়েড বৃদ্ধি': 'enlarged_thyroid',
        'ভাঙ্গা নখ': 'brittle_nails', 'শরীর ফুলে যাওয়া': 'swollen_extremeties',
        'অতিরিক্ত ক্ষুধা': 'excessive_hunger',
        'অতিরিক্ত বৈবাহিক যোগাযোগ': 'extra_marital_contacts',
        'শুকনো এবং ফাটা ঠোঁট': 'drying_and_tingling_lips',
        'কথা জড়িয়ে যাওয়া': 'slurred_speech', 'হাঁটু ব্যাথা': 'knee_pain',
        'কুঁচকিতে ব্যাথা': 'hip_joint_pain',
        'পেশীর দূর্বলতা': 'muscle_weakness', 'ঘাড় শক্ত': 'stiff_neck', 'হাড়ের জোড়া ফোলা': 'swelling_joints',
        'শক্ত নড়াচড়া': 'movement_stiffness', 'গোলভাবে ঘুরা': 'spinning_movements',
        'ভারসাম্য কম': 'loss_of_balance',
        'শরীর দুর্বল': 'unsteadiness', 'শরীরের একপাশে দুর্বলতা': 'weakness_of_one_body_side',
        'ঘ্রাণশক্তি কম': 'loss_of_smell', 'প্রসাবে অস্বস্তি': 'bladder_discomfort',
        'প্রসাবে দুর্গন্ধ': 'foul_smell_of urine', 'ঘন ঘন প্রসাব অনুভব': 'continuous_feel_of_urine',
        'গ্যাস নির্গত': 'passage_of_gases', 'শরীরের ভিতরে চুলকানি': 'internal_itching',
        'ফ্যাকাসে চেহারা': 'toxic_look_(typhos)', 'বিষণ্ণতা': 'depression', 'খিটখিটেভাব': 'irritability',
        'পেশির আশেপাশে ব্যাথা': 'muscle_pain', 'দুর্বল মস্তিষ্ক': 'altered_sensorium',
        'শরীরের উপর লালচে দাগ': 'red_spots_over_body', 'পেটের উপরিভাগে ব্যাথা': 'belly_pain',
        'অনিয়মিত মাসিক': 'abnormal_menstruation', 'গায়ের রং পরিবর্তন': 'dischromic _patches',
        'চোখ থেকে পানি পড়ছে': 'watering_from_eyes', 'ক্ষুধা বেড়েছে': 'increased_appetite',
        'প্রসাব খুব ঘন ঘন': 'polyuria', 'পারিবারিক ইতিহাস': 'family_history', 'সাদা থুথু': 'mucoid_sputum',
        'নোংরা থুথু': 'rusty_sputum', 'মনোযোগের অভাব': 'lack_of_concentration',
        'চোখের দৃষ্টিশক্তি কম': 'visual_disturbances', 'শরীরে রক্ত গ্রহন': 'receiving_blood_transfusion',
        'শরীরে প্লাজমা গ্রহন': 'receiving_unsterile_injections', 'জ্ঞান নাই': 'coma',
        'বমি অথবা মলে রক্ত': 'stomach_bleeding', 'পেটের দুই পাশে ফোলা': 'distention_of_abdomen',
        'মদ খাওয়ার প্রবনতা': 'history_of_alcohol_consumption',
        'শরীরে পানি': 'fluid_overload', 'মূত্রনালীতে রক্ত': 'blood_in_sputum',
        'পায়ের রগ ফোলা': 'prominent_veins_on_calf', 'বুকের আশেপাশে খারাপ লাগা': 'palpitations',
        'কষ্টকর হাঁটাচলা': 'painful_walking', 'পুঁজ ভর্তি ব্রন': 'pus_filled_pimples',
        'গায়ে কালো ছোপ': 'blackheads', 'ক্ষত': 'scurring',
        'গা থেকে খোসা উঠা': 'skin_peeling', 'গায়ের রং তামাটে': 'silver_like_dusting',
        'নখে সাদা ছোপ': 'small_dents_in_nails', 'নখে ব্যাথা': 'inflammatory_nails',
        'ফোস্কা': 'blister', 'নাকের পাশে লাল ফুসকুড়ি': 'red_sore_around_nose',
        'গায়ে হলুদ ছোপ': 'yellow_crust_ooze'
        }
dis = {
    'Hepatitis E': 'হেপাটাইটিস ই', 'Peptic ulcer': 'আলসার', 'Urinary tract infection': 'প্রসাবের ইনফেকশন',
    'Hyperthyroidism': 'গলায় থাইরয়েড', 'Hepatitis D': 'হেপাটাইটিস ডি', 'Alcoholic hepatitis': 'যকৃৎ দুর্বল',
    'Paralysis (brain hemorrhage)': 'প্যারিলিসিস', 'Hemorrhoids(piles)': 'পাইলস',
    'Gastroenteritis': 'বদ হজম', 'Jaundice': 'জন্ডিস', 'Varicose veins': 'ভ্যারিকোস ভেইন',
    'Diabetes ': 'ডায়াবেটিস্',
    'Migraine': 'মাইগ্রেন', 'Bronchial Asthma': 'হাঁপানি', 'Hypothyroidism': 'শরীরে থাইরয়েড',
    'Drug Reaction': 'অ্যালকোহলিক পার্শ্বপ্রতিক্রিয়া', 'Allergy': 'এলার্জি', 'Arthritis': 'বাত',
    'Impetigo': 'চর্ম রোগ',
    'Hypoglycemia': 'নিম্ন রক্তচাপ', 'Acne': 'ব্রণ', 'Paroxysmal Positional Vertigo': 'কানের সমস্যা',
    'Heart attack': 'হার্ট অ্যাটাক', 'Pneumonia': 'নিউমোনিয়া', 'Cervical spondylosis': 'বয়সের বাত',
    'Common Cold': 'সাধারন ঠান্ডা',
    'Hypertension ': 'উচ্চ রক্তচাপ', 'Dengue': 'ডেঙ্গু', 'Psoriasis': 'চর্ম রোগ', 'Chicken pox': 'জল বসন্ত',
    'GERD': 'গ্যাসে বুক জ্বালা',
    'Fungal infection': 'ছত্রাক সংক্রমণ', 'Hepatitis B': 'হেপাটাইটিস বি', 'hepatitis A': 'হেপাটাইটিস এ',
    'Malaria': 'ম্যালেরিয়া', 'Corona': 'করোনা', 'Osteoarthristis': 'গাঁটে বাত', 'Typhoid': 'টাইফয়েড',
    'Cholestasis': 'যকৃৎ সমস্যা', 'Tuberculosis': 'যক্ষ্মা', 'AIDS': 'এইডস',
    'Hepatitis C': 'হেপাটাইটিস সি'
}

data = pd.read_csv('Database/Training.csv')
df = pd.DataFrame(data)
cols = df.columns[:-1]

x = df[cols]  # x is the feature
y = df['prognosis']  # y is the target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

features = cols
feature_dict = {}

for i, f in enumerate(features):
    feature_dict[f] = i


# print(sample_x)

def prediction():
    # symptoms = ['joint_pain', 'muscle_wasting']
    symptoms = [GUI.p.get(), GUI.en.get(), GUI.bb.get(), GUI.ee.get(), GUI.hh.get()]

    symptoms = [trix[j] for j in symptoms if j != '']

    hack_set = set()

    pos = []

    for i in range(len(symptoms)):
        pos.append(feature_dict[symptoms[i]])

    sample_x = [1.0 if i in pos else 0.0 for i in range(len(features))]
    sample_x = [sample_x]
    # Decision Tree

    """dt = DecisionTreeClassifier()

    dt.fit(x_train, y_train)

    print(dt.predict(sample_x))

    y_pred = dt.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy Decision Tree: {accuracy * 100}%")"""

    # Naive Bayes

    naive = GaussianNB()

    naive.fit(x_train, y_train)

    # print(f"Naive Bayes: {dis.get(*map(str, naive.predict(sample_x)))}")

    hack_set.add(dis.get(*map(str, naive.predict(sample_x))))

    y_pred = naive.predict(x_test)

    accuracy_naive = accuracy_score(y_test, y_pred) * 100

    # print(f"Accuracy for Naive Bayes: {accuracy_naive}%")

    # Random Forest

    random = RandomForestClassifier()

    random.fit(x_train, y_train)

    hack_set.add(dis.get(*map(str, random.predict(sample_x))))

    # print(f"Random Forest: {dis.get(*map(str, random.predict(sample_x)))}")

    y_pred = random.predict(x_test)

    accuracy_random = accuracy_score(y_test, y_pred) * 100

    # print(f"Accuracy for Random Forest: {accuracy_random}%")

    # LogisticRegression
    """Logic = LogisticRegression()

    Logic.fit(x_train, y_train)

    print(f'LogisticRegression: {dis.get(*map(str, Logic.predict(sample_x)))}')

    y_pred = Logic.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy for Logistic Regression: {accuracy * 100}%")"""

    # SVM

    """Svm = svm.SVC()

    Svm.fit(x_train, y_train)

    y_pred = Svm.predict(x_test)

    print(f"KNN: {dis.get(*map(str, Svm.predict(sample_x)))}")

    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"Accuracy for SVM: {accuracy}%")"""

    magic = list(hack_set)

    s = ""
    if len(hack_set) == 1:
        s = s + "".join(magic[0])
    else:
        s = s + "".join(magic[0]) + ' অথবা ' + "".join(magic[1])
    # Exceptions for Wrong Try
    if not symptoms:
        GUI.final_result.delete(0, GUI.END)
        GUI.final_result.insert(0, "ভুল ! লক্ষন বাছাই করুন")

    elif len(set(symptoms)) != len(symptoms):
        GUI.final_result.delete(0, GUI.END)
        GUI.final_result.insert(0, "ভুল ! ভিন্ন ভিন্ন চেষ্টা করুন")
    else:
        GUI.final_result.delete(0, GUI.END)
        GUI.final_result.insert(0, s)
