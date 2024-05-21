import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read the ICD-10 codes from the Excel file
icd10_df = pd.read_excel("ICDMaster.xlsx")
icd10_df = icd10_df[82000:96000]
# Step 2: Extract relevant fields from the patient file
patient_file = """
CHIEF COMPLAINT:
Cough
CURRENT MEDICATIONS:
Patient or patient's family denies using any medications
Patient or patient's family denies using any medications
Medication Reconciliation Completed and Signed by Test, RN @ 12/01/2022 20:36:51
ALLERGIES:
Fish Containing Products causes Unknown,Unknown.
MSG (Free Text) causes Unknown,Unknown.
PCP:
PATTON ()
PATTON ()
MODE OF ARRIVAL:
The patient arrived by walk in.
SOURCE (TR):
The information was obtained from the patient.
CURRENT VISIT:
Arrival time: 12/01/2022 20:22. 1. Triage time: 12/01/2022 20:28. Assessment time: 12/01/2022 20:28.
Room #: ED 2. Time to room: 12/01/2022 20:28. This is not a return visit for the same problem. Pt c/o
cough for 6 days. Took Advil and Zyrtec 30 mins ago. -[HP@12/01/2022 20:36].
INITIAL VITALS AT PRESENTATION:
12/01/2022 20:25 Weight:230 lb(104 kg) Stated; Height:60 in(152 cm) BMI:45.16
12/01/2022
20:25 BP: 142/63 Arm (Automatic) MAP: 89.33 mmHG, Temp: 98.5 ºF Oral, HR: 80 Sitting Awake ,
RR: 15, O2 Sat: 99% RA , Pain: 0/10 ( Number scale )
GENERAL INFORMATION:
Immunizations are up to date. The patient has had tetanus immunization within the last 5 years. Gender
Reassignment: No. The patient is a full code. The patient does not have an advanced directive. The
preferred language of the patient or the patient's family is English. Interpreter was not used.
PREHOSPITAL (TR):
No prehospital care was provided.
TRIAGE ASSESSMENT:
Chief complaint: Cough
Timing: The onset / occurrence was 6 day(s) ago.
Quality: There has been a cough which was described as nonproductive.
Severity: There was no shortness of breath.
Associated Symptoms: There was no associated earache, fever, foreign body or wheezing.
PAST MEDICAL/SURGICAL HX:
Immunizations: Immunizations are up to date. The patient's last tetanus immunization was within the
previous 5 years.
Medical: PULMONARY: No history of asthma or chronic lung disease (COPD).
Surgical: The patient has a history of cholecystectomy.
SOCIAL HISTORY:
Habits: Tobacco use: Never smoker - [266919005]. No reported history of alcohol use. No reported
illegal drug use.
FUNCTIONAL/SAFETY:
Domestic violence screening: The patient denies being in a relationship in which he/she has been
physically hurt or threatened. The patient reports feeling safe in his/her current environment. The patient
denies being bullied.
Functional screening: Do you have trouble taking care of yourself - with feeding, dressing? No. The
patient denies falling unexpectedly or frequently. Functional and/or Therapy referral given? No. Fall risk
assessment was completed.
Morse Fall Risk screening: History of falling within 3 months: no (0). Secondary diagnosis: no (0).
Ambulatory aid: bed rest / nurse assist / wheelchair / none (0). IV / Saline Lock: no (0). Gait /
Transferring: normal / bedrest / immobile (0). Mental Status: oriented to own ability (0). Total Fall Score:
0.
LIFESTYLE SCREENINGS:
Alcohol screening: The patient denies alcohol use.
Nutritional screening: The patient denies having had an unexpected weight gain/loss of over 20
pounds in the last 6 months. The patient denies being on a special diet. A nutritional referral was not
given to the patient.
MENTAL HEALTH SCREENING:
Depression Screening: The patient denies having felt down, depressed, or hopeless in the last 2 weeks.
Over the past 2 weeks, have you felt little interest or pleasure in doing things? No. The patient denies any
suicidal or homicidal ideation. The patient denies any previous suicidal attempts.
TRIAGE ACUITY:
4 (Less Urgent).
TRIAGE DISPOSITION:
Time to room: 12/01/2022 20:28.
ASSESSMENT
Triage performed with Assessment
GENERAL INFORMATION (TRIAGE):
Immunizations are up to date. The patient has had tetanus immunization within the last 5 years. Gender
Reassignment: No. The patient is a full code. The patient does not have an advanced directive. The
preferred language of the patient or the patient's family is English. Interpreter was not used.
PREHOSPITAL:
No prehospital care was provided.
GENERAL ASSESSMENT:
12/01/2022 20:28.
Constitutional: The patient was well-appearing. The patient was in no distress.
Psychosocial: Attentive and appropriate.
Speech: The patient's speech was coherent.
Skin: The skin was warm and dry. Skin color was normal. Skin turgor was normal. Skin Integrity:
Intact.
Mental Status: The patient was oriented to person, place and time.
Deficits: The patient has no hearing deficit(s). The patient has no vision deficit(s). The patient has no
sensory/motor deficits. The patient has no mental deficits.
FOCUSED ASSESSMENT:
Chief complaint: Cough
Timing: The onset / occurrence was 6 day(s) ago.
Quality: There has been a cough which was described as nonproductive.
Severity: There was no shortness of breath.
Associated Symptoms: There was no associated earache, fever, foreign body or wheezing.
EXAMINATION:
Respiratory: Right lung negative for wheezes, rales and rhonchi. Right breath sounds were not
diminished. Left lung negative for wheezes, rales and rhonchi. Left breath sounds were not diminished.
Chest: Negative for accessory muscle use and retractions.
CV: The patient's heart rate was normal and the rhythm was regular.
GI: Bowel sounds were normal.
Neurological: The patient was alert and awake. The patient was not confused or agitated. There was a
response to verbal stimuli and a response to painful stimuli.
ACUITY:
4 (Less Urgent).
VITALS HISTORY:
12/01/2022 20:25 Weight:230 lb(104 kg) Stated; Height:60 in(152 cm) BMI:45.16
12/01/2022
20:25 BP: 142/63 Arm (Automatic) MAP: 89.33 mmHG, Temp: 98.5 ºF Oral, HR: 80 Sitting Awake ,
RR: 15, O2 Sat: 99% RA , Pain: 0/10 ( Number scale )
ORDERS WITHOUT RESULTS:
Placed Orders:
Decadron (Dexamethasone) Inj * - 10 mg IM Placed at 12/01/2022 20:43, MD, MD; Completed
at 12/01/2022 20:47, Test, H. , RN
AZITHROMYCIN 250 MG TABLET * - 500 mg Oral Placed at 12/01/2022 20:43, MD, MD;
Completed at 12/01/2022 20:47, Test, H. , RN
BENZONATATE 100 MG CAPSULE * - 200 mg PO Placed at 12/01/2022 20:43, MD, MD;
Completed at 12/01/2022 20:47, Test, H. , RN
TREATMENT NOTES:
12/1/2022
20:25 - VS: VITALS : Weight: 230 lb S Height: 60 in - Recorded by Test Test, RN.
20:25 - VS: VITALS : BP: 142/63 Arm (Auto) MAP: 89.33 mmHG, Temp: 98.5 ºF Oral, HR: 80
Sitting Awake, RR: 15, O2 Sat: 99% O2 Given: RA , Pain: 0/10 Number scale - Recorded by Test
Test, RN.
20:47 - MED: Decadron (Dexamethasone) Inj [Patient identity verified by Test Test, RN,
matching DOB and patient name]
GIVEN: 10 mg - L Gluteal IM - Recorded by Test Test, RN.
20:47 - MED: BENZONATATE 100 MG CAPSULE [Patient identity verified by Test Test, RN,
matching DOB and patient name]
GIVEN: 200 mg PO - Recorded by Test Test, RN.
20:47 - MED: AZITHROMYCIN 250 MG TABLET [Patient identity verified by Test Test, RN,
matching DOB and patient name]
GIVEN: 500 mg PO - Recorded by Test Test, RN.
PROCEDURE DETAILS:
IMPRESSION:
ACUTE UPPER RESPIRATORY INFECTION, UNSPECIFIED
ACUTE COUGH
DISPOSITION:
Discharged.
Discharge: The patient was discharged to Home. The patient was felt to be in good condition. On a 0 to
10 pain scale, the patient's pain was 0/10. The patient was given education, information and/or training
regarding follow-up care and medication(s). Verbal discharge instructions were given and Written
discharge instructions were given. Understanding of the instructions was expressed. The patient left the
Facility by walking. Upon discharge, IV status was not applicable. The patient departed the facility on
12/01/2022 20:48. Disposition Signed by Test, RN @ 12/01/2022 20:48:31
Instructions given to the patient: Upper Respiratory Infection (URI), Adult, Cough and the patient
was advised to return if worsening or increasing return as needed.
Special instructions: Use medications as prescribed. Follow up with primary care.
Follow up provider: PATTON, ,.
Follow up: NONE.
Prescriptions: albuterol sulfate 90 mcg/actuation 2 puff every four hours as needed prn
cough/wheezing [e-prescribed] -- HEB Pharmacy Floresville 925 10TH STREET FLORESVILLE TX,
78114 (830) 393-8098 Dispense: 8.5 Generic allowed No refills Rx Num 72483
azithromycin 250 mg 1 tablet once a day [e-prescribed] -- HEB Pharmacy Floresville 925 10TH STREET
FLORESVILLE TX, 78114 (830) 393-8098 Dispense: 4 Generic allowed No refills Rx Num 72481
benzonatate 200 mg 1 capsule three times a day as needed prn cough [e-prescribed] -- HEB Pharmacy
Floresville 925 10TH STREET FLORESVILLE TX, 78114 (830) 393-8098 Dispense: 30 Generic
allowed No refills Rx Num 72482
CALL BACK:
Call Back - Open
Disposition Type Discharged
NOTES:
The disposition paperwork given to the patient included discharge instructions, prescription(s) and
excuse(s).
SIGN OFF:
Test, RN
Triage assessment signed by Test Test, RN on 12/01/2022 20:40
Assessment signed by Test Test, RN on 12/01/2022 20:40 Chart
electronically signed by Test, RN @ 12/01/2022 20:50:55
CHIEF COMPLAINT:
Cough
VITALS HISTORY:
12/01/2022 20:25 Weight:230 lb(104 kg) Stated; Height:60 in(152 cm) BMI:45.16
12/01/2022
20:25 BP: 142/63 Arm (Automatic) MAP: 89.33 mmHG, Temp: 98.5 ºF Oral, HR: 80 Sitting Awake ,
RR: 15, O2 Sat: 99% RA , Pain: 0/10 ( Number scale )
GENERAL INFORMATION (TRIAGE):
Immunizations are up to date. The patient has had tetanus immunization within the last 5 years. Gender
Reassignment: No. The patient is a full code. The patient does not have an advanced directive. The
preferred language of the patient or the patient's family is English. Interpreter was not used.
CURRENT MEDICATIONS:
Patient or patient's family denies using any medications
Patient or patient's family denies using any medications
Medication Reconciliation Completed and Signed by Test, RN @ 12/01/2022 20:36:51
ALLERGIES:
Fish Containing Products causes Unknown,Unknown.
MSG (Free Text) causes Unknown,Unknown.
MODE OF ARRIVAL:
The patient arrived by walk in.
SOURCE (PHYS):
The information was obtained from the patient.
PREHOSPITAL (PHYS):
No prehospital care was provided.
HISTORY OF PRESENT ILLNESS:
This patient is a 37 year old female who presents with a chief complaint of Cough. Provider assessment
time was 12/01/2022 20:42. I reviewed the vital signs, the oxygen saturation result and the nursing/
treatment notes. I agree with the chief complaint selected for this patient's chart. The onset was 6 day(s)
prior to arrival. There has been a cough which was described as nonproductive. There was no shortness of
breath. The patient developed the symptoms spontaneously. There is no history of asthma, COPD, a
recent upper respiratory infection or vaping. The patient has not traveled outside of the United States. The
patient has not had contact with anyone who has traveled outside of the United States within the last 30
days. Travel inside the US within the last 30 days? No. There was no associated congestion, earache,
fever, foreign body, myalgias, rash, sore throat or wheezing.
ROS
CONSTITUTIONAL: Negative for fever.
EYES: Denies any eye problems.
ENT: Negative for ear pain. Negative for nasal congestion. Negative for throat pain and sore throat.
Denies any mouth problems.
RESPIRATORY: Positive for cough. Negative for wheezing.
CV: Denies any cardiovascular problems.
GI: Denies any GI problems.
GU: Denies any GU problems.
NEUROLOGICAL: Denies any neurological problems.
MUSCULOSKELETAL: Denies any musculoskeletal problems.
INTEGUMENTARY: Negative for rash.
ALLERGIC/IMMUNOLOGIC: Denies any allergic/immunologic problems.
HEMATOLOGIC: Denies any hematologic problems.
ENDOCRINE: There has been no change in weight. Denies heat or cold intolerance. Denies excessive
thirst, hunger, or urination.
PSYCHIATRIC: Denies any psychiatric problems.
PAST MEDICAL HISTORY (PHYS):
Immunizations: Immunizations are up to date. The patient's last tetanus immunization was within the
previous 5 years.
Medical: PULMONARY: No history of asthma or chronic lung disease (COPD).
Surgical: The patient has a history of cholecystectomy.
FAMILY HISTORY (PHYS):
SOCIAL HISTORY (PHYS):
Habits: Tobacco use: Never smoker - [266919005]. No reported alcohol use. No reported illegal drug
use.
INITIAL VITALS AT PRESENTATION:
12/01/2022 20:25 Weight:230 lb(104 kg) Stated; Height:60 in(152 cm) BMI:45.16
12/01/2022
20:25 BP: 142/63 Arm (Automatic) MAP: 89.33 mmHG, Temp: 98.5 ºF Oral, HR: 80 Sitting Awake ,
RR: 15, O2 Sat: 99% RA , Pain: 0/10 ( Number scale )
PHYSICAL EXAMINATION
Constitutional: The patient was alert. The patient was not ill-appearing. The patient was in no distress.
Neck: The neck does not have meningeal signs. Negative for anterior and posterior cervical adenopathy.
Eyes: Negative for discharge and redness.
ENT: The airway was patent. Right tympanic membrane negative for redness, bulging, dullness,
retraction and perforation. Left tympanic membrane negative for redness, bulging, dullness, retraction and
perforation. The nose was negative for bleeding, discharge, injection and mucosal edema. Sinus exam
was negative for tenderness. The oropharynx was negative for exudate, redness and tonsillar hypertrophy.
Oral membranes were moist.
Respiratory: The pulse oximeter reading was within a normal range. The lung sounds were clear, and
breath sounds were equal bilaterally.
Chest: Negative for accessory muscle use and retractions.
CV: Heart rate was normal and the rhythm was regular. There was no systolic murmur or diastolic
murmur. There was no extremity edema.
GI: Palpation negative for hepatomegaly, splenomegaly and a non-pulsatile mass. There was no
abdominal tenderness.
Skin: No rash was present.
Neurological: The patient was oriented to person, place and time.
Psychiatric: Affect was appropriate.
DIAGNOSTIC CONSIDERATIONS FOR COUGH:
DIAGNOSTIC CONSIDERATIONS: Bronchitis, COVID -19, Pneumonia and URI.
(I have considered the above as the potential cause of the patient's condition. I have based my
consideration on a limited patient encounter, and my considerations may not be all-inclusive. History,
physical examination, and/or diagnostic studies, in combination with medical judgment, have been used
in determining the final diagnosis)
PROVIDER TREATMENT NOTES:
ADDITIONAL NOTES:
Prudent Layperson: Medical Decision Making: decadron 10 mg azithro and tessalon given here. d/c'd
home. shared decision making done w pt. likely viral etiology. -[MD@12/02/2022 00:58].
CONSULTATION(S):
ORDERS AND RESULTS:
Placed Orders:
Decadron (Dexamethasone) Inj * - 10 mg IM Placed at 12/01/2022 20:43, MD, MD; Completed
at 12/01/2022 20:47, Test, H. , RN
AZITHROMYCIN 250 MG TABLET * - 500 mg Oral Placed at 12/01/2022 20:43, MD, MD;
Completed at 12/01/2022 20:47, Test, H. , RN
BENZONATATE 100 MG CAPSULE * - 200 mg PO Placed at 12/01/2022 20:43, MD, MD;
Completed at 12/01/2022 20:47, Test, H. , RN
PROCEDURES:
TREATMENT TIMES:
No critical care was performed on this patient.
IMPRESSION:
ACUTE UPPER RESPIRATORY INFECTION, UNSPECIFIED
ACUTE COUGH
DISPOSITION (PHYS):
The disposition time decision was 12/01/2022 20:44.
Discharge.
Discharge: The patient was discharged to Home. The patient's condition upon discharge was good and
stable. Education was provided to the patient in reference to the final impressions, prognosis and need for
follow up.
Instructions given to the patient: Upper Respiratory Infection (URI), Adult, Cough and the patient
was advised to return if worsening or increasing return as needed.
Special instructions: Use medications as prescribed. Follow up with primary care.
Follow up provider: PATTON, ,.
Follow up: NONE.
Prescriptions: albuterol sulfate 90 mcg/actuation 2 puff every four hours as needed prn
cough/wheezing [e-prescribed] -- HEB Pharmacy Floresville 925 10TH STREET FLORESVILLE TX,
78114 (830) 393-8098 Dispense: 8.5 Generic allowed No refills Rx Num 72483
azithromycin 250 mg 1 tablet once a day [e-prescribed] -- HEB Pharmacy Floresville 925 10TH STREET
FLORESVILLE TX, 78114 (830) 393-8098 Dispense: 4 Generic allowed No refills Rx Num 72481
benzonatate 200 mg 1 capsule three times a day as needed prn cough [e-prescribed] -- HEB Pharmacy
Floresville 925 10TH STREET FLORESVILLE TX, 78114 (830) 393-8098 Dispense: 30 Generic
allowed No refills Rx Num 72482
PDMP:
PDMP data reviewed: No. Reason for not accessing: not Prescribing Controlled Medications.
SIGN OFF:
M. Test, MD
Chart electronically signed by M. Test, MD @ 12/02/2022 00:58:12
CALL BACK:
Call Back - Open
Disposition Type Discharged
DISPOSITION NOTES:
Smoking History (from Social History review) Never smoker - [266919005].
SIGN OFF:

"""

# Extracting chief complaint and impressions from the patient file
# chief_complaint_start = patient_file.find("CHIEF COMPLAINT:") + len("CHIEF COMPLAINT:")
# chief_complaint_end = patient_file.find("IMPRESSION:")
# chief_complaint = patient_file[chief_complaint_start:chief_complaint_end].strip()

# impressions_start = patient_file.find("IMPRESSION:") + len("IMPRESSION:")
# impressions_end = patient_file.find("ORDERS AND RESULTS:")
# impressions = patient_file[impressions_start:impressions_end].strip()

# # Step 3: Preprocess the extracted text
# # Preprocessing may involve lowercasing, removing punctuation, etc. depending on the model requirements.
# chief_complaint = chief_complaint.lower()
# impressions = impressions.lower()
chief_complaint = 'cough'
impressions = 'acute upper respiratory inflamatory infection, unspecified'
# Step 4: Use a large pre-trained language model to find similar descriptions
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode the queries and descriptions
queries = [chief_complaint, impressions]
encoded_queries = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')

encoded_descriptions = tokenizer(icd10_df['Description'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Forward pass through the model to obtain embeddings
with torch.no_grad():
    query_outputs = model(**encoded_queries)
    description_outputs = model(**encoded_descriptions)

query_embeddings = query_outputs.pooler_output
description_embeddings = description_outputs.pooler_output

# Calculate cosine similarity between query embeddings and description embeddings
similarities = cosine_similarity(query_embeddings, description_embeddings)

# Step 5: Output the corresponding ICD-10 codes for the matched descriptions
for i, query in enumerate(queries):
    max_sim_index = similarities[i].argmax()  # Index of the most similar description
    matched_description = icd10_df.iloc[max_sim_index]['Description']
    matched_code = icd10_df.iloc[max_sim_index]['ICD']
    print(f"For query matched description is '{matched_description}' with ICD-10 code '{matched_code}'.")
