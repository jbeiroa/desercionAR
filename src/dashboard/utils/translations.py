# src/dashboard/utils/translations.py

# Home page content
HOME_PRESENTATION = """
In Argentina, education is a right established by the National Constitution and regulated by National Education Law No. 26,206. Compulsory schooling spans 14 consecutive years, from the age of 4 in kindergarten, through primary school (lasting 6 or 7 years depending on the jurisdiction), to secondary school (lasting 6 or 5 years, depending on the length of primary school in the jurisdiction).\n
This project uses a machine learning model to predict school dropout from one quarter to the next using the EPH database. The model is applied to the data from the third quarter of 2023 to generate the predictions shown on this page.
"""

# Analytics page content
ANALYTICS_TITLE = "Histogram and Bivariate Distribution of the Selected Variable"
ANALYTICS_DROPDOWN_PLACEHOLDER = "Select variable to display"

# Variable descriptions, moved from analytics.py
VARIABLE_DESCRIPTIONS = {
    "CODUSU": "Dwelling code, allows matching with Households and Persons. Also allows tracking across quarters.",
    "NRO_HOGAR": "Household code, allows matching with Persons.",
    "COMPONENTE": "Component number: order number assigned to people within each household.",
    "ANO4": "Year of survey (4 digits)",
    "TRIMESTRE": "Observation window",
    "REGION": "Region",
    "MAS_500": "MAS_500",
    "AGLOMERADO": "Agglomerate code",
    "PONDERA": "Weighting factor",
    "CH03": "Relationship of kinship",
    "CH04": "Sex",
    "CH06": "How many years old are you?",
    "CH07": "Are you currently...",
    "CH08": "Do you have any type of medical coverage for which you pay or have a discount?",
    "CH09": "Do you know how to read and write?",
    "CH10": "Do you attend or have you attended an educational establishment? (college, school, university)",
    "CH11": "That establishment is...",
    "CH15": "Where were you born?",
    "CH16": "Where did you live 5 years ago?",
    "ESTADO": "Activity condition",
    "PP02E": "During those 30 days, you did not look for a job because...",
    "PP02H": "In the last 12 months, did you look for a job at any time?",
    "servicio_domestico": "If the person provides domestic service in another household",
    "NIVEL_ED": "Educational level",
    "IV1": "Type of housing",
    "IV2": "Number of rooms",
    "IV3": "Interior floor material",
    "IV4": "Exterior roof covering material",
    "IV5": "Does the roof have a ceiling/interior lining?",
    "IV6": "Do you have water...",
    "IV7": "The water is from...",
    "IV9": "The bathroom or latrine is...",
    "IV11": "The bathroom drain is...",
    "IV12_2": "The dwelling is located in a floodable area",
    "II1": "Rooms for exclusive use",
    "II2": "Rooms used for sleeping",
    "II3": "Rooms used exclusively as workplaces",
    "II4_1": "It has a kitchen room",
    "II4_2": "It has a laundry room",
    "II4_3": "It has a garage",
    "II8": "Fuel used for cooking",
    "II9": "Bathroom (tenure and use)",
    "V1": "Do the people in the household live off what they earn from work?",
    "V2": "Do the people in the household live off a retirement or pension?",
    "V21": "Do the people in the household live off a retirement bonus?",
    "V22": "Do the people in the household live off a retirement retroactive payment?",
    "V3": "Do the people in the household live off severance pay?",
    "V5": "Do the people in the household live off a subsidy or social assistance?",
    "V6": "Do the people in the household live off goods, clothes, food provided by institutions?",
    "V7": "Do the people in the household live off goods, clothes, food provided by people?",
    "V8": "Do the people in the household live off a rent?",
    "V11": "Do the people in the household live off a study grant?",
    "V12": "Do the people in the household live off child support payments or money help from other people?",
    "V13": "Do the people in the household live by spending savings?",
    "V14": "Do the people in the household live off loans from family/friends?",
    "IX_TOT": "Total number of people in the household",
    "IX_MEN10": "Number of children under 10 in the household",
    "DECCFR": "Family income decile",
    "CH06_jefx": "Age of the head of household",
    "ESTADO_jefx": "Occupational status of the head of household",
    "NIVEL_ED_jefx": "Educational level of the head of household",
    "APORTES_JUBILATORIOS_jefx": "Retirement contributions of the head of household",
    "PP04B1_jefx": "If the head of household performs domestic service in other households",
    "ESTADO_conyuge": "Labor status of the spouse",
    "JEFA_MUJER": "Female head of household",
    "HOGAR_MONOP": "Single-parent household",
    "ratio_ocupados": "Ratio of employed/household members",
    "NBI_COBERTURA_PREVISIONAL": "NBI_COBERTURA_PREVISIONAL",
    "NBI_DIFLABORAL": "NBI_DIFLABORAL",
    "NBI_HACINAMIENTO": "NBI_HACINAMIENTO",
    "NBI_SANITARIA": "NBI_SANITARIA",
    "NBI_TENENCIA": "NBI_TENENCIA",
    "NBI_TRABAJO_PRECARIO": "NBI_TRABAJO_PRECARIO",
    "NBI_VIVIENDA": "NBI_VIVIENDA",
    "NBI_ZONA_VULNERABLE": "NBI_ZONA_VULNERABLE",
    "DESERTO": "DESERTO",
}

def get_variable_description(var_name: str) -> str:
    """
    Get the description of a variable.

    Args:
        var_name (str): The name of the variable.

    Returns:
        str: The description of the variable.
    """
    return VARIABLE_DESCRIPTIONS.get(var_name, var_name)
