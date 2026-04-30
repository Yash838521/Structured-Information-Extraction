# 0 Shot LLM end-to-end prompt ------------------------------------------------------------------------
END_TO_END_PROMPT_0_SHOT = """You are a biomedical research assistant. Your job is to read a clinical trial abstract and extract three pieces of structured information from it.

DEFINITIONS:
- Participants: Who was studied? Look for descriptions of the study population — their condition, demographics, sample size, eligibility criteria. Examples: "patients with type 2 diabetes", "healthy adult volunteers aged 18-65", "children with asthma".
- Interventions: What treatments or procedures were compared? Look for drugs, therapies, dosages, surgical techniques, or behavioural programs. Examples: "metformin 500mg twice daily", "cognitive behavioural therapy", "laparoscopic surgery versus open surgery".
- Outcomes: What was measured or assessed? Look for clinical endpoints, test scores, biomarkers, adverse events. Examples: "overall survival", "pain score on VAS", "rate of adverse events", "HbA1c levels".


Abstract: "{abstract}"

Return ONLY a JSON object with keys "participants", "interventions", "outcomes", each containing a list of short extracted phrases. Extract the most specific phrases you can find — do not copy entire sentences unless necessary. If a field is not mentioned, use an empty list."""





# 1 Shot LLM end-to-end prompt ------------------------------------------------------------------------
END_TO_END_PROMPT_1_SHOT = """You are a biomedical research assistant. Your job is to read a clinical trial abstract and extract three pieces of structured information from it.

DEFINITIONS:
- Participants: Who was studied? Look for descriptions of the study population — their condition, demographics, sample size, eligibility criteria. Examples: "patients with type 2 diabetes", "healthy adult volunteers aged 18-65", "children with asthma".
- Interventions: What treatments or procedures were compared? Look for drugs, therapies, dosages, surgical techniques, or behavioural programs. Examples: "metformin 500mg twice daily", "cognitive behavioural therapy", "laparoscopic surgery versus open surgery".
- Outcomes: What was measured or assessed? Look for clinical endpoints, test scores, biomarkers, adverse events. Examples: "overall survival", "pain score on VAS", "rate of adverse events", "HbA1c levels".

Example:
Abstract: "A comparison of the serologic responses to oral and injectable trivalent poliovirus vaccines. United States children two months of age were randomly assigned to two groups that received either the commercially available oral trivalent poliovirus vaccine ( OPV ) or an injectable (inactivated) trivalent poliovirus vaccine (IPV) with a confirmed minimum D-antigen content of 27, 3.5, and 29 units for poliovirus types 1, 2, and 3, respectively. Vaccine was given at two, four, and 18 months of age. Sera obtained from 439 children at two, four, and six months of age and from 85 children at 18 and 20 months of age were examined for neutralizing antibodies. The percentage of children with detectable antibodies and the reciprocal geometric mean titers were similar for both groups at two months of age for antibodies to all three poliovirus types. At 20 months of age, all children but one had detectable antibodies to all three poliovirus types. Significantly higher geometric mean titers against types 2 and 3 were noted at 20 months of age for the IPV group."

Output:
{
    "participants": ["United States children two months of age", "439 children at two , four , and six months of age and from 85 children at 18 and 20 months of age"],
    "interventions": ["trivalent poliovirus vaccines .", "oral trivalent poliovirus vaccine ( OPV )", "injectable ( inactivated ) trivalent poliovirus vaccine ( IPV )"],
    "outcomes": ["neutralizing antibodies .", "percentage of children with detectable antibodies and the reciprocal geometric mean titers", "antibodies to all three poliovirus types .", "geometric mean titers against types 2 and 3"]
}

Abstract: "{abstract}"

Return ONLY a JSON object with keys "participants", "interventions", "outcomes", each containing a list of short extracted phrases. Extract the most specific phrases you can find — do not copy entire sentences unless necessary. If a field is not mentioned, use an empty list."""





# Multi-shot (5) LLM end-to-end prompt ------------------------------------------------------------------------
END_TO_END_PROMPT_MULTISHOT = """You are a biomedical research assistant. Your job is to read a clinical trial abstract and extract three pieces of structured information from it.

DEFINITIONS:
- Participants: Who was studied? Look for descriptions of the study population — their condition, demographics, sample size, eligibility criteria. Examples: "patients with type 2 diabetes", "healthy adult volunteers aged 18-65", "children with asthma".
- Interventions: What treatments or procedures were compared? Look for drugs, therapies, dosages, surgical techniques, or behavioural programs. Examples: "metformin 500mg twice daily", "cognitive behavioural therapy", "laparoscopic surgery versus open surgery".
- Outcomes: What was measured or assessed? Look for clinical endpoints, test scores, biomarkers, adverse events. Examples: "overall survival", "pain score on VAS", "rate of adverse events", "HbA1c levels".

Example 1:
Abstract: "A comparison of the serologic responses to oral and injectable trivalent poliovirus vaccines. United States children two months of age were randomly assigned to two groups that received either the commercially available oral trivalent poliovirus vaccine ( OPV ) or an injectable (inactivated) trivalent poliovirus vaccine (IPV) with a confirmed minimum D-antigen content of 27, 3.5, and 29 units for poliovirus types 1, 2, and 3, respectively. Vaccine was given at two, four, and 18 months of age. Sera obtained from 439 children at two, four, and six months of age and from 85 children at 18 and 20 months of age were examined for neutralizing antibodies. The percentage of children with detectable antibodies and the reciprocal geometric mean titers were similar for both groups at two months of age for antibodies to all three poliovirus types. At 20 months of age, all children but one had detectable antibodies to all three poliovirus types. Significantly higher geometric mean titers against types 2 and 3 were noted at 20 months of age for the IPV group."

Output:
{
    "participants": ["United States children two months of age", "439 children at two , four , and six months of age and from 85 children at 18 and 20 months of age"],
    "interventions": ["trivalent poliovirus vaccines .", "oral trivalent poliovirus vaccine ( OPV )", "injectable ( inactivated ) trivalent poliovirus vaccine ( IPV )"],
    "outcomes": ["neutralizing antibodies .", "percentage of children with detectable antibodies and the reciprocal geometric mean titers", "antibodies to all three poliovirus types .", "geometric mean titers against types 2 and 3"]
}

Example 2:
Abstract: "Tracheal soiling with blood during intranasal surgery--comparison of two endotracheal tubes. Sixty adult patients, ASA Classes I & II, were involved in a study to compare the effectiveness of Mallinckrodt Hi-Lo-Evac tube and Portex blue line tube in preventing soiling of the lower airways during intranasal surgery. The Hi-Lo-Evac tube with and without pack was significantly more effective than the Portex tube with pharyngeal pack (P less than 0.002) and (P less than 0.01 respectively). There was no significant difference when the Hi-Lo-Evac tube was used with or without a pack (P greater than 0.2). The more effective protection of the lower airways by the Hi-Lo-Evac tube is attributed to the facility of subglottic aspiration during surgery. It is suggested that the Hi-Lo-Evac tube could be used with safety during intranasal surgery in order to reduce postoperative morbidity associated with the use of pharyngeal pack."

Output:
{
    "participants": ["Sixty adult patients , ASA Classes I & II"],
    "interventions": ["intranasal surgery -- comparison", "Mallinckrodt Hi-Lo-Evac tube and Portex blue line tube"],
    "outcomes": ["effective protection of the lower airways", "postoperative morbidity"]
}

Example 3: 
Abstract: "The effect of values affirmation on race-discordant patient-provider communication. BACKGROUND Communication between African American patients and white health care providers has been shown to be of poorer quality when compared with race-concordant patient-provider communication. Fear on the part of patients that providers stereotype them negatively might be one cause of this poorer communication. This stereotype threat may be lessened by a values-affirmation intervention. METHODS In a blinded experiment, we randomized 99 African American patients with hypertension to perform a values-affirmation exercise or a control exercise before a visit with their primary care provider. We compared patient-provider communication for the 2 groups using audio recordings of the visit analyzed with the Roter Interaction Analysis System. We also evaluated visit satisfaction, trust, stress, and mood after the visit by means of a questionnaire. RESULTS Patients in the intervention group requested and provided more information about their medical condition (mean SE number of utterances, 66.3 6.8 in the values-affirmation group vs 48.1 5.9 in the control group P = .03). Patient-provider communication in the intervention group was characterized as being more interested, friendly, responsive, interactive, and respectful (P = .02) and less depressed and distressed (P = .03). Patient questionnaires did not detect differences in visit satisfaction, trust, stress, or mood. Mean visit duration did not differ significantly between the groups (19.2 minutes in the control group vs 20.5 minutes in the intervention group P = .29). CONCLUSIONS A values-affirmation exercise improves aspects of patient-provider communication in race-discordant primary care visits. The clinical impact of the intervention must be defined before widespread implementation can be recommended."

Output:
{
    "participants": ["race-discordant patient-provider", "African American patients and white health care providers", "99 African American patients with hypertension"],
    "interventions": ["values-affirmation exercise or a control exercise", "care visits"],
    "outcomes": ["medical condition", "and respectful ( P = .02 ) and less depressed", ". Patient questionnaires", "mood . Mean visit duration did not differ significantly between the groups ("]
}

Example 4:
Abstract: "Effect of guided imagery on length of stay, pain and anxiety in cardiac surgery patients."

Output:
{
    "participants": ["cardiac surgery patients ."],
    "interventions": ["guided imagery"],
    "outcomes": ["length of stay , pain and anxiety"]
}

Example 5:
Abstract: "A comparative trial of fixed ratio beta-adrenoceptor blocker and diuretic combination products in moderate hypertension. Two fixed ratio combination tablets, 10 mg pindolol combined with 5 mg clopamide and 100 mg metoprolol combined with 12.5 mg hydrochlorothiazide, were compared at two dose levels in a double-blind crossover trial in 10 previously untreated hypertensive patients. No significant difference was observed between resting blood pressure on the two combinations at either dose level. Exercise systolic pressure was lower after the pindolol/clopamide combination at low dose (p less than 0.05). The incidence of side-effects in this trial was high but proper comparison of the two products could not be made because of the small number of patients. It is suggested that combination products should be used only after patients have failed to respond adequately to a single agent."

Output:
{
    "participants": ["moderate hypertension .", "10 previously untreated hypertensive patients ."],
    "interventions": ["beta-adrenoceptor blocker and diuretic combination products", "pindolol combined with 5 mg clopamide", "metoprolol combined with 12.5 mg hydrochlorothiazide", "combination products"],
    "outcomes": ["resting blood pressure", "Exercise systolic pressure", "incidence of side-effects"]
}

Abstract: "{abstract}"

Return ONLY a JSON object with keys "participants", "interventions", "outcomes", each containing a list of short extracted phrases. Extract the most specific phrases you can find — do not copy entire sentences unless necessary. If a field is not mentioned, use an empty list."""




# 0 Shot LLM end-to-end prompt ------------------------------------------------------------------------
DECOMPOSED_PROMPTS_0_SHOT = {
    'participants': """You are a biomedical research assistant.

Your task is to extract PARTICIPANT information from a clinical trial abstract.

Participants = who was studied. Look for:
- Medical condition or disease
- Demographics (age, sex, population type)
- Sample size
- Eligibility criteria

Instructions:
- Extract short, specific phrases (not full sentences)
- Prefer the most informative span
- Do not include explanations
- If no participants are mentioned, return an empty list

Return ONLY a JSON list of strings.

Abstract: "{abstract}"
""",

    'interventions': """You are a biomedical research assistant.

Your task is to extract INTERVENTION information from a clinical trial abstract.

Interventions = what treatments or procedures were tested. Look for:
- Drug names and dosages
- Therapies or procedures
- Comparators (placebo, control, alternative treatment)
- Treatment regimens

Instructions:
- Extract short, specific phrases (not full sentences)
- Prefer precise treatment descriptions (include dosage if available)
- Do not include explanations
- If no interventions are mentioned, return an empty list

Return ONLY a JSON list of strings.

Abstract: "{abstract}"
""",

    'outcomes': """You are a biomedical research assistant.

Your task is to extract OUTCOME information from a clinical trial abstract.

Outcomes = what was measured or assessed. Look for:
- Clinical endpoints
- Biomarkers or lab measurements
- Patient-reported outcomes
- Adverse events or safety measures

Instructions:
- Extract short, specific phrases (not full sentences)
- Focus on measurable variables
- Do not include explanations
- If no outcomes are mentioned, return an empty list

Return ONLY a JSON list of strings.

Abstract: "{abstract}"
"""
}





# 1 Shot LLM end-to-end prompt ------------------------------------------------------------------------
DECOMPOSED_PROMPTS_1_SHOT = {
    'participants': """You are extracting PARTICIPANT information from a clinical trial abstract.

Participants = who was studied. Look for:
- The medical condition or disease being studied
- Demographics (age, sex, specific populations)
- Sample size (number of patients/subjects)
- Eligibility criteria

Example:
Abstract: "United States children two months of age were randomly assigned to two groups that received either the commercially available oral trivalent poliovirus vaccine ( OPV ) or an injectable (inactivated) trivalent poliovirus vaccine (IPV) with a confirmed minimum D-antigen content of 27, 3.5, and 29 units for poliovirus types 1, 2, and 3, respectively. Sera obtained from 439 children at two, four, and six months of age and from 85 children at 18 and 20 months of age were examined for neutralizing antibodies."
Output: ["United States children two months of age", "439 children at two , four , and six months of age and from 85 children at 18 and 20 months of age"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
""",

    'interventions': """You are extracting INTERVENTION information from a clinical trial abstract.

Interventions = what treatments, drugs, or procedures were tested. Look for:
- Drug names and dosages
- Surgical or therapeutic procedures
- Comparators (placebo, standard care, alternative treatment)
- Treatment duration or regimen

Example:
Abstract: "A comparison of the serologic responses to oral and injectable trivalent poliovirus vaccines. United States children two months of age were randomly assigned to two groups that received either the commercially available oral trivalent poliovirus vaccine ( OPV ) or an injectable (inactivated) trivalent poliovirus vaccine (IPV) with a confirmed minimum D-antigen content of 27, 3.5, and 29 units for poliovirus types 1, 2, and 3, respectively."
Output: ["trivalent poliovirus vaccines .", "oral trivalent poliovirus vaccine ( OPV )", "injectable ( inactivated ) trivalent poliovirus vaccine ( IPV )"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
""",

    'outcomes': """You are extracting OUTCOME information from a clinical trial abstract.

Outcomes = what was measured or assessed. Look for:
- Primary and secondary endpoints
- Clinical measurements (blood pressure, survival, cure rate)
- Patient-reported measures (pain scores, quality of life)
- Safety outcomes (adverse events, side effects)

Example:
Abstract: "Sera obtained from 439 children at two, four, and six months of age and from 85 children at 18 and 20 months of age were examined for neutralizing antibodies. The percentage of children with detectable antibodies and the reciprocal geometric mean titers were similar for both groups at two months of age for antibodies to all three poliovirus types. At 20 months of age, all children but one had detectable antibodies to all three poliovirus types. Significantly higher geometric mean titers against types 2 and 3 were noted at 20 months of age for the IPV group."
Output: ["neutralizing antibodies .", "percentage of children with detectable antibodies and the reciprocal geometric mean titers", "antibodies to all three poliovirus types .", "geometric mean titers against types 2 and 3"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
"""
}





# Multi-shot (5) LLM end-to-end prompt ------------------------------------------------------------------------
DECOMPOSED_PROMPTS_MULTISHOT = {
    'participants': """You are extracting PARTICIPANT information from a clinical trial abstract.

Participants = who was studied. Look for:
- The medical condition or disease being studied
- Demographics (age, sex, specific populations)
- Sample size (number of patients/subjects)
- Eligibility criteria

Example 1:
Abstract: "United States children two months of age were randomly assigned to two groups that received either the commercially available oral trivalent poliovirus vaccine ( OPV ) or an injectable (inactivated) trivalent poliovirus vaccine (IPV) with a confirmed minimum D-antigen content of 27, 3.5, and 29 units for poliovirus types 1, 2, and 3, respectively. Sera obtained from 439 children at two, four, and six months of age and from 85 children at 18 and 20 months of age were examined for neutralizing antibodies."
Output: ["United States children two months of age", "439 children at two , four , and six months of age and from 85 children at 18 and 20 months of age"]

Example 2:
Abstract: "Sixty adult patients, ASA Classes I & II, were involved in a study to compare the effectiveness of Mallinckrodt Hi-Lo-Evac tube and Portex blue line tube in preventing soiling of the lower airways during intranasal surgery."
Output: ["Sixty adult patients , ASA Classes I & II"]

Example 3:
Abstract: "The effect of values affirmation on race-discordant patient-provider communication. METHODS In a blinded experiment, we randomized 99 African American patients with hypertension to perform a values-affirmation exercise or a control exercise before a visit with their primary care provider."
Output: ["race-discordant patient-provider", "African American patients and white health care providers", "99 African American patients with hypertension"]

Example 4:
Abstract: "Effect of guided imagery on length of stay, pain and anxiety in cardiac surgery patients."
Output: ["cardiac surgery patients ."]

Example 5:
Abstract: "A comparative trial of fixed ratio beta-adrenoceptor blocker and diuretic combination products in moderate hypertension. Two fixed ratio combination tablets, 10 mg pindolol combined with 5 mg clopamide and 100 mg metoprolol combined with 12.5 mg hydrochlorothiazide, were compared at two dose levels in a double-blind crossover trial in 10 previously untreated hypertensive patients."
Output: ["moderate hypertension .", "10 previously untreated hypertensive patients ."]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
""",

    'interventions': """You are extracting INTERVENTION information from a clinical trial abstract.

Interventions = what treatments, drugs, or procedures were tested. Look for:
- Drug names and dosages
- Surgical or therapeutic procedures
- Comparators (placebo, standard care, alternative treatment)
- Treatment duration or regimen

Example 1:
Abstract: "A comparison of the serologic responses to oral and injectable trivalent poliovirus vaccines. United States children two months of age were randomly assigned to two groups that received either the commercially available oral trivalent poliovirus vaccine ( OPV ) or an injectable (inactivated) trivalent poliovirus vaccine (IPV) with a confirmed minimum D-antigen content of 27, 3.5, and 29 units for poliovirus types 1, 2, and 3, respectively."
Output: ["trivalent poliovirus vaccines .", "oral trivalent poliovirus vaccine ( OPV )", "injectable ( inactivated ) trivalent poliovirus vaccine ( IPV )"]

Example 2:
Abstract: "Tracheal soiling with blood during intranasal surgery--comparison of two endotracheal tubes. Sixty adult patients, ASA Classes I & II, were involved in a study to compare the effectiveness of Mallinckrodt Hi-Lo-Evac tube and Portex blue line tube in preventing soiling of the lower airways during intranasal surgery."
Output: ["intranasal surgery -- comparison", "Mallinckrodt Hi-Lo-Evac tube and Portex blue line tube"]

Example 3:
Abstract: "METHODS In a blinded experiment, we randomized 99 African American patients with hypertension to perform a values-affirmation exercise or a control exercise before a visit with their primary care provider. CONCLUSIONS A values-affirmation exercise improves aspects of patient-provider communication in race-discordant primary care visits."
Output: ["values-affirmation exercise or a control exercise", "care visits"]

Example 4:
Abstract: "Effect of guided imagery on length of stay, pain and anxiety in cardiac surgery patients."
Output: ["guided imagery"]

Example 5:
Abstract: "A comparative trial of fixed ratio beta-adrenoceptor blocker and diuretic combination products in moderate hypertension. Two fixed ratio combination tablets, 10 mg pindolol combined with 5 mg clopamide and 100 mg metoprolol combined with 12.5 mg hydrochlorothiazide, were compared at two dose levels in a double-blind crossover trial in 10 previously untreated hypertensive patients. It is suggested that combination products should be used only after patients have failed to respond adequately to a single agent."
Output: ["beta-adrenoceptor blocker and diuretic combination products", "pindolol combined with 5 mg clopamide", "metoprolol combined with 12.5 mg hydrochlorothiazide", "combination products"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
""",

    'outcomes': """You are extracting OUTCOME information from a clinical trial abstract.

Outcomes = what was measured or assessed. Look for:
- Primary and secondary endpoints
- Clinical measurements (blood pressure, survival, cure rate)
- Patient-reported measures (pain scores, quality of life)
- Safety outcomes (adverse events, side effects)

Example 1:
Abstract: "Sera obtained from 439 children at two, four, and six months of age and from 85 children at 18 and 20 months of age were examined for neutralizing antibodies. The percentage of children with detectable antibodies and the reciprocal geometric mean titers were similar for both groups at two months of age for antibodies to all three poliovirus types. At 20 months of age, all children but one had detectable antibodies to all three poliovirus types. Significantly higher geometric mean titers against types 2 and 3 were noted at 20 months of age for the IPV group."
Output: ["neutralizing antibodies .", "percentage of children with detectable antibodies and the reciprocal geometric mean titers", "antibodies to all three poliovirus types .", "geometric mean titers against types 2 and 3"]

Example 2:
Abstract: "The more effective protection of the lower airways by the Hi-Lo-Evac tube is attributed to the facility of subglottic aspiration during surgery. It is suggested that the Hi-Lo-Evac tube could be used with safety during intranasal surgery in order to reduce postoperative morbidity associated with the use of pharyngeal pack."
Output: ["effective protection of the lower airways", "postoperative morbidity"]

Example 3:
Abstract: "RESULTS Patients in the intervention group requested and provided more information about their medical condition (mean SE number of utterances, 66.3 6.8 in the values-affirmation group vs 48.1 5.9 in the control group P = .03). Patient-provider communication in the intervention group was characterized as being more interested, friendly, responsive, interactive, and respectful (P = .02) and less depressed and distressed (P = .03). Patient questionnaires did not detect differences in visit satisfaction, trust, stress, or mood. Mean visit duration did not differ significantly between the groups (19.2 minutes in the control group vs 20.5 minutes in the intervention group P = .29)."
Output: ["medical condition", "and respectful ( P = .02 ) and less depressed", ". Patient questionnaires", "mood . Mean visit duration did not differ significantly between the groups ("]

Example 4:
Abstract: "Effect of guided imagery on length of stay, pain and anxiety in cardiac surgery patients."
Output: ["length of stay , pain and anxiety"]

Example 5:
Abstract: "No significant difference was observed between resting blood pressure on the two combinations at either dose level. Exercise systolic pressure was lower after the pindolol/clopamide combination at low dose (p less than 0.05). The incidence of side-effects in this trial was high but proper comparison of the two products could not be made because of the small number of patients."
Output: ["resting blood pressure", "Exercise systolic pressure", "incidence of side-effects"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
"""
}