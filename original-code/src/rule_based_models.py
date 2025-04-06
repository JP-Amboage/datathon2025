import json
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Tuple, Any

from openai import OpenAI

from base_data import ClientData
import re
import pandas as pd

import Levenshtein

import re
from collections import defaultdict


TODAY = "2025-04-01"
COUNTTRY_CODES = pd.read_csv("./data/country_codes.csv")
COUNTTRY_CODES["name"] = COUNTTRY_CODES["name"].str.split(",").str[0].str.strip()
CURRENCY_CODES = pd.read_csv("./data/countries_currencies.csv")

NATIONALITIES = pd.read_csv("./data/nationalities.csv")
NATIONALITIES["en_short_name"] = (
    NATIONALITIES["en_short_name"].str.split(",").str[0].str.strip()
)
NATIONALITIES["nationality"] = (
    NATIONALITIES["nationality"].str.split(",").str[0].str.strip()
)


def simple_mrz(passport_data: dict) -> Tuple[str, str]:

    given_names = (
        f"{passport_data['first_name']}<{passport_data['middle_name'].upper()}"
    )
    if passport_data["middle_name"] == "":
        given_names = passport_data["first_name"]

    line1 = f"P<{passport_data['country_code']}{passport_data['last_name'].upper()}<<{given_names}".ljust(
        45, "<"
    )

    birth_date = datetime.strptime(passport_data["birth_date"], "%Y-%m-%d").strftime(
        "%y%m%d"
    )
    line2 = f"{passport_data['passport_number'].upper()}{passport_data['country_code']}{birth_date}".ljust(
        45, "<"
    )

    return line1.upper(), line2.upper()


def extract_paths(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from extract_paths(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from extract_paths(item, f"{path}[{i}]")
    else:
        yield (path, obj)


class Model(ABC):
    @abstractmethod
    def predict(self, client: ClientData) -> int:
        pass


def flag_name_integrity(client: ClientData) -> bool:

    client_name = client.account_form["name"].lower().replace(" ", "")

    if client_name != client.account_form["first_name"].lower().replace(
        " ", ""
    ) + client.account_form["middle_name"].lower().replace(
        " ", ""
    ) + client.account_form[
        "last_name"
    ].lower().replace(
        " ", ""
    ):
        return True

    if client_name != client.client_profile["name"].lower().replace(" ", ""):
        return True

    return False


def flag_verify_email(client: ClientData):
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    if client.account_form["email_address"] != client.client_profile["email_address"]:
        return True

    if not re.match(email_pattern, client.account_form["email_address"]):
        return True

    return False


def flag_phone(client: ClientData):
    phone_number = client.account_form["phone_number"].replace(" ", "")

    if phone_number != client.client_profile["phone_number"].replace(" ", ""):
        return True

    if not re.match("^\+?\d+$", phone_number):
        return True

    if len(phone_number) > 15 or len(phone_number) < 8:
        return True

    return False


def flag_address(client: ClientData):
    if client.account_form["address"] != client.client_profile["address"]:
        return True

    return False


def flag_country(client: ClientData):
    if client.account_form["address"] != client.client_profile["address"]:
        return True

    if (
        client.account_form["country_of_domicile"]
        != client.client_profile["country_of_domicile"]
    ):
        return True

    return False


def flat_date_consistencies(client: ClientData):

    for duplicate_field in (
        "birth_date",
        "passport_issue_date",
        "passport_expiry_date",
        "passport_number",
    ):
        if client.passport[duplicate_field] != client.client_profile[duplicate_field]:
            return True

    today = datetime.strptime("2025-04-01", "%Y-%m-%d").date()
    birth_date = datetime.strptime(
        client.client_profile["birth_date"], "%Y-%m-%d"
    ).date()
    passport_issue_date = datetime.strptime(
        client.client_profile["passport_issue_date"], "%Y-%m-%d"
    ).date()
    passport_expiry_date = datetime.strptime(
        client.client_profile["passport_expiry_date"], "%Y-%m-%d"
    ).date()

    secondary_school_grad = client.client_profile["secondary_school"]["graduation_year"]
    higher_education_years = [
        edu["graduation_year"] for edu in client.client_profile["higher_education"]
    ]
    employment_start_ends = [
        (e["start_year"], e["end_year"])
        for e in client.client_profile["employment_history"]
    ]

    try:
        prev = 0
        for start, end in employment_start_ends:
            assert prev <= start  # TODO should be an error?

        assert birth_date < passport_issue_date < passport_expiry_date
        assert passport_issue_date < today
        assert (
            birth_date.year + 16 < employment_start_ends[0][0]
            if employment_start_ends
            else today.year
        )
        assert birth_date.year + 12 < secondary_school_grad
        assert today.year - birth_date.year < 120
        assert birth_date < date(secondary_school_grad, 1, 1) <= today
        assert all(
            date(secondary_school_grad, 1, 1) <= date(higher_edu, 1, 1) <= today
            for higher_edu in higher_education_years
        )
        assert all(
            birth_date
            < date(start, 1, 1)
            <= (date(end, 1, 1) if end else today)
            <= today
            for start, end in employment_start_ends
        )
    except AssertionError:
        return True
    return False


def flag_passport(client: ClientData):
    if not (
        client.client_profile["passport_number"]
        == client.account_form["passport_number"]
        == client.passport["passport_number"]
    ):
        return True

    if len(client.passport["passport_mrz"]) != 2:
        return True

    mrz_line1, mrz_line2 = simple_mrz(client.passport)

    passport_line1, passport_line2 = client.passport["passport_mrz"]

    if mrz_line1 != passport_line1 or mrz_line2 != passport_line2:
        return True

    if not re.match('\w\w\d{7}', client.passport["passport_number"]):
        return True

    return False


def flag_dates(client: ClientData):
    passport_issue_date = client.client_profile["passport_issue_date"]
    if passport_issue_date != client.passport["passport_issue_date"]:
        return True

    date_pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$"
    if not re.match(date_pattern, passport_issue_date):
        return True

    if passport_issue_date < client.client_profile["birth_date"]:
        return True

    if passport_issue_date > TODAY:
        return True

    if client.client_profile["birth_date"] > TODAY:
        return True

    if (len(client.client_profile["higher_education"]) > 0) and client.client_profile[
        "higher_education"
    ][0]["graduation_year"] < client.client_profile["secondary_school"][
        "graduation_year"
    ] + 2:
        return True

    return False


def flag_wealth(client: ClientData):
    real_state_value = client.client_profile["aum"]["real_estate_value"]
    if (
        sum([x["property value"] for x in client.client_profile["real_estate_details"]])
        != real_state_value
    ):
        return True

    return False


def flag_country_exist(client: ClientData):
    if (
        client.client_profile["country_of_domicile"]
        not in COUNTTRY_CODES["name"].values
    ):
        return True

    return False


def flag_currency_exist(client: ClientData):
    if client.client_profile["currency"] not in CURRENCY_CODES["AlphabeticCode"].values:
        return True
    return False


def flag_currencies_match(client: ClientData):
    if client.client_profile["currency"] != client.account_form["currency"]:
        return True
    return False


def flag_currency_matches_country(client: ClientData):
    # relevant countries
    # countries = []
    # preferred_markets = client.client_profile["preferred_markets"]
    # countries.extend(preferred_markets)
    # countries.append(client.client_profile["country_of_domicile"])

    residence_country = client.client_profile["country_of_domicile"].upper()
    currency = client.client_profile["currency"]
    # Check if the currency is in the list of currencies for the residence country
    expected_currency = CURRENCY_CODES[CURRENCY_CODES["Entity"] == residence_country][
        "AlphabeticCode"
    ].tolist()[0]
    if expected_currency != currency:
        return True
    return False


def flag_country_matches_code(client: ClientData):
    country = client.passport["country"]
    country_code = client.passport["country_code"]
    if (
        NATIONALITIES[NATIONALITIES["en_short_name"] == country][
            "alpha_3_code"
        ].tolist()[0]
        != country_code
    ):
        return True
    return False


def flag_nationality(client: ClientData):
    if client.passport["nationality"] != client.client_profile["nationality"]:
        return True
    nationality = client.passport["nationality"]
    country = client.passport["country"]
    if (
        NATIONALITIES[NATIONALITIES["en_short_name"] == country][
            "nationality"
        ].tolist()[0]
        != nationality
    ):
        return True
    return False


def flag_missing_values(client: ClientData):

    NULLABLE_FIELDS = ("end_year", "middle_name")

    for data in (client.client_profile, client.client_description, client.passport):
        for path, value in extract_paths(data):
            if path.split(".")[-1] in NULLABLE_FIELDS:
                continue

            if value is None or value == "":
                return True

    return False



def flag_savings(client: ClientData):

    earned_money = 0

    for employment in client.client_profile["employment_history"]:
        earned_money += (
            (employment["end_year"] or 2025) - employment["start_year"]
        ) * employment["salary"]

    aum = client.client_profile["aum"]

    total_wealth_today = aum["savings"]  # + aum["real_estate_value"]
    income = earned_money + aum["inheritance"]
    if income < total_wealth_today:
        return True
    return False


def flag_yachts(client: ClientData):

    for party in [
        #'yacht party in Berlin',
        #'yacht party in Paris',
        #'yacht party in Vienna',
        #'yacht party in Zürich',
        #'yacht party in Madrid',
    ]:
        if party in client.client_description["Summary Note"]:
            return True

    return False


def flag_salary_increases(client: ClientData):
    if not client.client_profile["employment_history"]:
        return False

    prev_salary = client.client_profile["employment_history"][0]["salary"]
    for employment in client.client_profile["employment_history"]:
        salary = employment["salary"]
        if salary >= prev_salary * 20:
            # print(client.passport['passport_number'])
            return True
        prev_salary = salary

    return False

def flag_gender(client: ClientData):
    if client.passport["gender"] != client.client_profile["gender"]:
        return True
    return False


prompt = """
I have this json:
{client_data}

fill these keys:
{{
     "age": age,
     "marital_status": marital_status,
     "secondary_school": {{
        "name": name,
        "graduation_year": graduation_year
    }}
    "higher_education": [
           {{
                 "university": university,
                 "graduation_year": graduation_year
           }}
     ]
      "employment_history": [
        {{
            "company": company,
            "position": position
        }}
        ],
        "savings": savings,
        "inheritance": inheritance,
        "real_estate_value": real_estate_value
         "real_estate_details": [
            {{
            "property value": property_value,
            "property location": property_location
            }}
        ],
}}
 The lists for higher_education, employment_history and real_estate_details may have 0, 1 or more objects.
 reason for yourself. DONT infer data, just use whats there.
 Missing values should be marked with empty string ""
 Denominations, dimensions for prices and numbers should be ignored. just put the number e.g 13000 and NOT 13000 EUR.
 marital status should be 1 word, e.g. single, married, divorced, widowed
 output should be a valid json with the given template. just fill the values, nothing else character. 
"""
def flag_description(client: ClientData):
    openai_client = OpenAI()

    response = openai_client.responses.create(
        model="gpt-4o",
        input=prompt.format(client_data=client.client_description)
    )
    try:
        response_data = json.loads(response.output_text[7:-3])
    except json.decoder.JSONDecodeError:
        print('OYYOY')
        return False

    try:
        if flag_compare_age(response_data.get('age'), client):
            return True

        for simple_path in [
            'marital_status',
            'secondary_school.name',
            'secondary_school.graduation_year',
            # 'real_estate_value',
            'savings',
            'inheritance'
        ]:
            if len(simple_path.split('.')) == 1:
                gpt_value = response_data.get(simple_path)
                if simple_path in ('real_estate_value', 'savings', 'inheritance'):
                    client_value = client.client_profile['aum'][simple_path]
                else:
                    client_value = client.client_profile[simple_path]
            else:
                parent, subpath = simple_path.split('.')
                gpt_value = response_data[parent][subpath]
                client_value = client.client_profile[parent][subpath]

            gpt_value = str(gpt_value).lower()
            client_value = str(client_value).lower()

            if simple_compare(gpt_value, client_value):
                return True

        if flag_higher_education(response_data, client):
            return True

        if flag_real_estate_details(response_data, client):
            return True

    except:
        return False

    return False

def flag_real_estate_details(response, client: ClientData):
    response_properties = response["real_estate_details"]

    response_properties = [prop for prop in response_properties
                           if not all(v == '' for v in prop.values())]
    if not response_properties:
        return False
    true_properties = client.client_profile["real_estate_details"]

    if len(response_properties) != len(true_properties):
        if len(response_properties) == 1 and response_properties[0]["location"] == '':
            return False

    true_values = [str(tp["property value"]) for tp in true_properties]
    for rp in response_properties:
        value = str(rp["property value"])
        if value != None and value.isdigit():
            if value in true_values:
                true_values.remove(value)

            else:
                return True

    true_locations = [tp["property location"] for tp in true_properties]
    for rp in response_properties:
        loc = rp["property location"]
        if loc != None and len(loc) > 0:
            if loc in true_locations:
                true_locations.remove(loc)

            else:
                return True


    return False


def flag_higher_education(response, client: ClientData):
    higher_education_response = response["higher_education"]
    if len(higher_education_response) != len(client.client_profile["higher_education"]):
        return True

    for x,y in zip(higher_education_response, client.client_profile["higher_education"]):
        if str(x["graduation_year"]) != str(y["graduation_year"]):
            return True
        
        if Levenshtein.distance(x["university"].lower(), y["university"].lower()) > 13:
            return True

    return False


def flag_compare_age(gpt_age, client: ClientData):
    if gpt_age in (None, "", 'none', 'None'):
        return False
    gpt_age = int(gpt_age)
    today = date.today()

    birth_date = datetime.strptime(
        client.client_profile["birth_date"], "%Y-%m-%d"
    ).date()

    birthday_age = today.year - birth_date.year

    return abs(gpt_age-birthday_age) > 2

def simple_compare(gpt_value: Any, client_value: Any):
    if gpt_value in (None, "", 'none', 'None'):
        return False
    gpt_value = str(gpt_value).lower()
    client_value = str(client_value).lower()

    return gpt_value != client_value

def flag_experience_gaps(client: ClientData):
    if client.client_profile["employment_history"] == []:
        return False
    prev = client.client_profile["employment_history"][0]["end_year"]
    for employment in client.client_profile["employment_history"][1:]:
        start = employment["start_year"]
        if start - prev > 2:
            return True
        prev = employment["end_year"]
    return False

def flag_late_graduation(client: ClientData):
    secondary_school_grad = client.client_profile["secondary_school"]["graduation_year"]
    secondary_school_grad = int(secondary_school_grad)
    birth_year = client.client_profile["birth_date"].split("-")[0]
    birth_year = int(birth_year)
    if secondary_school_grad - birth_year > 20:
        return True
    return False

def flag_university_before_graduation(client: ClientData):
    if len(client.client_profile["higher_education"]) == 0:
        return False
    secondary_school_grad = client.client_profile["secondary_school"]["graduation_year"]
    secondary_school_grad = int(secondary_school_grad)
    higher_education_years = [
        edu["graduation_year"] for edu in client.client_profile["higher_education"]
    ]
    for higher_education in higher_education_years:
        if int(higher_education) < secondary_school_grad + 3:
            return True
    return False

def find_redundant_sentences(data_dict):
    sentence_map = defaultdict(set)  # sentence -> set of field names

    # Helper: normalize a sentence
    def clean_sentence(sentence):
        return re.sub(r'\s+', ' ', sentence.strip()).rstrip('.')

    # Step 1: Split and map sentences to fields
    for field, content in data_dict.items():
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for raw_sentence in sentences:
            sentence = clean_sentence(raw_sentence)
            if sentence:
                sentence_map[sentence].add(field)

    # Step 2: Find duplicates (appearing in >1 field)
    redundant_sentences = {
        sentence: fields
        for sentence, fields in sentence_map.items()
        if len(fields) > 1
    }

    return redundant_sentences

def flag_copy_paste(client: ClientData):
    info = client.client_description
    redundant_sentences = find_redundant_sentences(info)
    if not redundant_sentences:
        return False
    for sentence, fields in redundant_sentences.items():
        if len(sentence) > 130:
            return True

def flag_late_university_graduation(client: ClientData):
    if len(client.client_profile["higher_education"]) == 0:
        return False
    secondary_school_grad = client.client_profile["secondary_school"]["graduation_year"]
    secondary_school_grad = int(secondary_school_grad)
    higher_education_year = client.client_profile["higher_education"][0]["graduation_year"]
    higher_education_year = int(higher_education_year)

    if higher_education_year - secondary_school_grad > 7:
        return True
    return False

# def flag_company_evolution(client: ClientData):
#     if len(client.client_profile["employment_history"]) <= 1:
#         return False
#     prev = client.client_profile["employment_history"][0]
#     for employment in client.client_profile["employment_history"][1:]:
#         company = employment["company"]
#         if prev["company"].lower() == company.lower() and "ceo" in prev["position"].lower() and "venture" in employment["position"].lower():
#             return True
#         prev = employment
#   return False

# def flag_euro_inheritance_gap(client: ClientData):
#     if client.client_profile["currency"].lower() == "eur" and client.client_profile["aum"]["inheritance"] and int(client.client_profile["aum"]["inheritance"]) > 1_000_000:
#         employment_history = client.client_profile["employment_history"]
#         if len(employment_history)>1:
#             prev_end_year = int(employment_history[0]["end_year"])
#             for employment in employment_history[1:]:
#                 start_year = int(employment["start_year"])
#                 if start_year - prev_end_year > 2:
#                     return True
#                 prev_end_year = employment["end_year"]
#   return False

def flag_investment_horizont(client: ClientData):
    horizont = client.client_profile["investment_horizon"].lower()
    if "month" in horizont or "week" in horizont or "day" in horizont:
        return True
    return False

def flag_type_of_mandate(client: ClientData):
    if client.client_profile["type_of_mandate"] == "Execution-only" or client.client_profile["type_of_mandate"] == "Hybrid":
        return True
    return False

def flag_investment_risk_profile(client: ClientData):
    if client.client_profile["investment_risk_profile"] == "Aggressive" or client.client_profile["investment_risk_profile"] == "Balanced" or client.client_profile["investment_risk_profile"] == "Conservative":
        return True
    return False

class SimpleModel(Model):

    def predict(self, client: ClientData) -> int:
        if False and flag_yachts(client):
            return True
        if flag_missing_values(client):
            return True
        if flag_name_integrity(client):
            return True
        if flag_verify_email(client):
            return True
        if flag_phone(client):
            return True
        if flag_address(client):
            return True
        if flag_country(client):
            return True
        if flag_passport(client):
            return True
        if flag_wealth(client):
            return True
        if flag_dates(client):
            return True
        if flat_date_consistencies(client):
            return True
        if flag_country_exist(client):
            return True
        if flag_currency_exist(client):
            return True
        if flag_currencies_match(client):
            return True
        if flag_nationality(client):
            return True
        if flag_country_matches_code(client):
            return True
        if flag_currency_matches_country(client):
            return True
        if flag_savings(client):
            return True
        # if flag_salary_increases(client): #adds FP but slightly bumps accuracy
        #     return True
        if flag_gender(client):
            return True
        # if flag_description(client):
        #     return True

        if flag_experience_gaps(client):
            return True

        # if flag_late_graduation(client):
        #     return True
        # if flag_university_before_graduation(client):
        #     return True
        # if flag_copy_paste(client):
        #     return True
        # if flag_late_university_graduation(client):
        #     return True

        # if flag_company_evolution(client):
        #     return True

        # if flag_euro_inheritance_gap(client):
        #     return True

        if flag_investment_horizont(client):
            return True
        
        if flag_type_of_mandate(client):
            return True

        if flag_investment_risk_profile(client):
            return True


        return False

import dotenv
dotenv.load_dotenv()