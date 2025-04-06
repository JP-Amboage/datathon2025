import torch
import pickle

from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from src.base_data import ClientData


def get_education(client_data: ClientData) -> str:
    description = client_data.client_description
    profile = client_data.client_profile

    # a. secondary school
    secondary_school = profile.get('secondary_school', {})
    school_name = secondary_school.get('name', 'N/A')
    grad_year = secondary_school.get('graduation_year', 'N/A')
    secondary_school_flat = f'Secondary education at {school_name} graduated in {grad_year}.'

    # b. higher education
    higher_edu = profile.get('higher_education', [])
    higher_edu_flat = ''
    for i, he in enumerate(higher_edu):
        uni_name = he.get('university', 'N/A')
        grad_year = he.get('graduation_year', 'N/A')
        if i > 0:
            higher_edu_flat += ' '
        higher_edu_flat += f'Higher education at {uni_name} graduated in {grad_year}.'

    # get the premise
    premise = f"{secondary_school_flat} {higher_edu_flat}" 

    # description: hypothesis
    description_edu = description.get('Education Background', '').strip()
    hypothesis = f"Description says: {description_edu}"

    # total education
    return premise, hypothesis


def get_employment(client_data: ClientData) -> str:
    profile = client_data.client_profile
    description = client_data.client_description

    # a. current employment
    employment = profile.get('employment_history', [])
    if not employment:
        employment_flat = "No employment history recorded."
    else:
        jobs = []
        for job in employment:
            start = job.get('start_year', 'N/A')
            end = job.get('end_year', 'present')
            company = job.get('company', 'N/A')
            position = job.get('position', 'N/A')
            salary = job.get('salary', 'N/A')
            jobs.append(f"Worked at {company} as {position} from {start} to {end}, earning {salary}.")
        employment_flat = ' '.join(jobs)

    # premise
    premise = f"Current employment: {employment_flat}"

    # description employment
    description_emp = description.get("Occupation History", '').strip()
    hypothesis = f"Description says: {description_emp}"

    return premise, hypothesis



def get_inheritance_profile(client_data: ClientData) -> str:
    profile = client_data.client_profile
    description = client_data.client_description

    aum = profile.get("aum", {})
    inheritance_amt = aum.get("inheritance", "N/A")

    inheritance_details = profile.get("inheritance_details", {})
    inh_year = inheritance_details.get("inheritance year", "N/A")
    inh_relation = inheritance_details.get("relationship", "N/A")
    inh_profession = inheritance_details.get("profession", "N/A")

    # premise
    premise = f"Client inherited {inheritance_amt} DKK in {inh_year} from their {inh_relation}, who was a {inh_profession}."

    # description inheritance
    description_wealth = description.get("Wealth Summary", "").strip()
    hypothesis = f"Description says: {description_wealth}"

    return premise, hypothesis


def get_wealth_profile(client_data: ClientData) -> str:
    profile = client_data.client_profile
    description = client_data.client_description

    aum = profile.get("aum", {})
    savings = aum.get("savings", "N/A")
    real_estate_total = aum.get("real_estate_value", "N/A")

    real_estate_details = profile.get("real_estate_details", [])
    real_estate_flat = ""
    if real_estate_details:
        real_estate_parts = []
        for prop in real_estate_details:
            value = prop.get("property value", "N/A")
            ptype = prop.get("property type", "N/A")
            loc = prop.get("property location", "N/A")
            real_estate_parts.append(f"{ptype} in {loc} valued at {value}")
        real_estate_flat = " Properties include: " + "; ".join(real_estate_parts) + "."

    premise = (
        f"Client has savings of {savings} DKK and total real estate value of {real_estate_total} DKK."
        f"{real_estate_flat}"
    )

    # hypothesis
    description_wealth = description.get("Wealth Summary", "").strip()
    if not description_wealth:
        hypothesis = "No wealth summary provided."
    else:
        hypothesis = f"Description says: {description_wealth}"

    return premise, hypothesis


def get_investment_profile(client_data: ClientData) -> str:
    profile = client_data.client_profile
    description = client_data.client_description

    # a. investment risk profile
    risk = profile.get('investment_risk_profile', 'N/A')
    horizon = profile.get('investment_horizon', 'N/A')
    experience = profile.get('investment_experience', 'N/A')
    mandate = profile.get('type_of_mandate', 'N/A')
    markets = profile.get('preferred_markets', [])
    # flatten the markets list
    markets_flat = ', '.join(markets) if markets else 'N/A'

    # premise
    premise = (
        f"Risk profile: {risk}. Horizon: {horizon}. Experience: {experience}. "
        f"Mandate: {mandate}. Preferred markets: {markets_flat}."
    )

    # description investment
    description_investment = description.get("Client Summary", "").strip()
    if not description_investment:
        hypothesis = "No investment summary provided."
    else:
        hypothesis = f"Description says: {description_investment}"

    return premise, hypothesis


def get_summary_note_profile(client_data: ClientData) -> str:
    profile = client_data.client_profile
    passport = client_data.passport
    description = client_data.client_description

    summary = description.get("Summary Note", "").strip()
    hypothesis = f"Description says: {summary}"
    if not summary:
        hypothesis = "No summary note provided."

    # Extract key facts to compare against structured fields
    birth_date = profile.get("birth_date", "N/A")
    country = profile.get("country_of_domicile", "N/A")
    nationality = profile.get("nationality", "N/A")
    name = profile.get("name", "N/A")

    passport_country = passport.get("country", "N/A")

    premise = (
        f"Profile says: Name: {name}. Birth date: {birth_date}. "
        f"Domicile: {country}. Nationality: {nationality}. Passport country: {passport_country}."
    )

    return premise, hypothesis



def get_family_background_profile(client_data: ClientData) -> str:
    profile = client_data.client_profile
    description = client_data.client_description

    family_bg = description.get("Family Background", "").strip()
    marital_status = profile.get("marital_status", "N/A")

    premise = f"Family background: {family_bg}."
    hypothesis = f"Description says: {family_bg}"
    if not family_bg:
        hypothesis = "No family background provided."

    return premise, hypothesis



def flatten_fields(client_data: ClientData) -> list[str]:
    premises, hypotheses = [], []

    # 1. Education
    education_premise, education_hypothesis = get_education(client_data)
    premises.append(education_premise)
    hypotheses.append(education_hypothesis)

    # 2. Employment
    employment_premise, employment_hypothesis = get_employment(client_data)
    premises.append(employment_premise)
    hypotheses.append(employment_hypothesis)

    # 3. Inheritance
    inheritance_premise, inheritance_hypothesis = get_inheritance_profile(client_data)
    premises.append(inheritance_premise)
    hypotheses.append(inheritance_hypothesis)

    # 4. Wealth
    wealth_premise, wealth_hypothesis = get_wealth_profile(client_data)
    premises.append(wealth_premise)
    hypotheses.append(wealth_hypothesis)

    # 5. Investment profile
    investment_premise, investment_hypothesis = get_investment_profile(client_data)
    premises.append(investment_premise)
    hypotheses.append(investment_hypothesis)

    # 6. Summary note analysis
    summary_note_premise, summary_note_hypothesis = get_summary_note_profile(client_data)
    premises.append(summary_note_premise)
    hypotheses.append(summary_note_hypothesis)

    # 7. Family background analysis
    family_background_premise, family_background_hypothesis = get_family_background_profile(client_data)
    premises.append(family_background_premise)
    hypotheses.append(family_background_hypothesis)

    return premises, hypotheses


def get_model_and_tokenizer(model_name: str = 'roberta-large-mnli'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    return tokenizer, model


def roberta_embeddings(client_data: list[ClientData], tokenizer, model, savepath: Path):
    all_embeddings = torch.zeros(len(client_data), 7, 1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, cd in enumerate(tqdm(client_data, desc='RoBERTa Embeddings')):
        premises, hypotheses = flatten_fields(cd)
        inputs = tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 1024)
            all_embeddings[i] = cls_embedding.cpu()
    all_embeddings = all_embeddings.numpy()
    with open(savepath, 'wb') as f:
        pickle.dump(all_embeddings, f)
    return all_embeddings

# from math import ceil


# def roberta_embeddings(client_data: list[ClientData], tokenizer, model, batch_size=32):
#     all_texts = []
#     client_text_indices = []
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     for cd in client_data:
#         texts = flatten_fields(cd)  # each returns 7 texts
#         assert len(texts) == 7
#         client_text_indices.append(len(all_texts))  # where this client's texts start
#         all_texts.extend(texts)

#     # Now all_texts has len = num_clients * 7
#     total_texts = len(all_texts)
#     num_batches = ceil(total_texts / batch_size)

#     cls_embeddings = []

#     for i in tqdm(range(num_batches), desc='Batched RoBERTa Embeddings'):
#         batch_texts = all_texts[i * batch_size: (i + 1) * batch_size]
#         inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)

#         with torch.no_grad():
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = model(**inputs)
#             cls_batch = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
#             cls_embeddings.append(cls_batch)

#     # Concatenate all batches into one big tensor
#     cls_embeddings = torch.cat(cls_embeddings, dim=0)  # shape: (total_texts, hidden_dim)

#     # Optional: split back by client if needed
#     for i, start_idx in enumerate(client_text_indices):
#         client_embeds = cls_embeddings[start_idx:start_idx + 7]
#         print(f'Client {i} embeddings shape: {client_embeds.shape}')  # should be (7, hidden_dim)
