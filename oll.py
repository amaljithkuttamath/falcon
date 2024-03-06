# import ollama

# messages = []

# code_prompt = """
# Date	Ransom Amount (USD)	Incident Type
# 2023-01-01	46036	Phishing
# 2023-01-08	34785	Phishing
# 2023-01-15	10596	Insider Threat
# 2023-01-22	23932	Ransomware
# 2023-01-29	17785	DDoS

#             Generate the code <code> for plotting the previous data in plotly,
#             in the format requested. The solution should be given using plotly
#             and only plotly. Do not use matplotlib.
#             Return the code <code> in the following
#             format ```python <code>```
#         """
# messages.append({
#             "role": "assistant",
#             "content": code_prompt
#         })

# model = "gemma:2b"
# response = ollama.chat(
#         model=model,
#         messages=messages
#         # temperature=temperature,
#         # max_tokens=max_tokens,
#         # top_p=top_p,
#         # frequency_penalty=0,
#         # presence_penalty=0,
#         # stop=None
#     )
# from pprint import pprint
# pprint(response)


import json

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="gemma:2b")

dataset_analysis = """Given a dataset with 1470 rows and 35 columns, covering the following areas: Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager.

    A statistical summary reveals:
                    Age Attrition BusinessTravel    DailyRate              Department  DistanceFromHome    Education EducationField  EmployeeCount  EmployeeNumber  EnvironmentSatisfaction Gender   HourlyRate  JobInvolvement     JobLevel          JobRole  JobSatisfaction MaritalStatus  MonthlyIncome   MonthlyRate  NumCompaniesWorked Over18 OverTime  PercentSalaryHike  PerformanceRating  RelationshipSatisfaction  StandardHours  StockOptionLevel  TotalWorkingYears  TrainingTimesLastYear  WorkLifeBalance  YearsAtCompany  YearsInCurrentRole  YearsSinceLastPromotion  YearsWithCurrManager
count   1470.000000      1470           1470  1470.000000                    1470       1470.000000  1470.000000           1470         1470.0     1470.000000              1470.000000   1470  1470.000000     1470.000000  1470.000000             1470      1470.000000          1470    1470.000000   1470.000000         1470.000000   1470     1470        1470.000000        1470.000000               1470.000000         1470.0       1470.000000        1470.000000            1470.000000      1470.000000     1470.000000         1470.000000              1470.000000           1470.000000
unique          NaN         2              3          NaN                       3               NaN          NaN              6            NaN             NaN                      NaN      2          NaN             NaN          NaN                9              NaN             3            NaN           NaN                 NaN      1        2                NaN                NaN                       NaN            NaN               NaN                NaN                    NaN              NaN             NaN                 NaN                      NaN                   NaN
top             NaN        No  Travel_Rarely          NaN  Research & Development               NaN          NaN  Life Sciences            NaN             NaN                      NaN   Male          NaN             NaN          NaN  Sales Executive              NaN       Married            NaN           NaN                 NaN      Y       No                NaN                NaN                       NaN            NaN               NaN                NaN                    NaN              NaN             NaN                 NaN                      NaN                   NaN
freq            NaN      1233           1043          NaN                     961               NaN          NaN            606            NaN             NaN                      NaN    882          NaN             NaN          NaN              326              NaN           673            NaN           NaN                 NaN   1470     1054                NaN                NaN                       NaN            NaN               NaN                NaN                    NaN              NaN             NaN                 NaN                      NaN                   NaN
mean      36.923810       NaN            NaN   802.485714                     NaN          9.192517     2.912925            NaN            1.0     1024.865306                 2.721769    NaN    65.891156        2.729932     2.063946              NaN         2.728571           NaN    6502.931293  14313.103401            2.693197    NaN      NaN          15.209524           3.153741                  2.712245           80.0          0.793878          11.279592               2.799320         2.761224        7.008163            4.229252                 2.187755              4.123129
std        9.135373       NaN            NaN   403.509100                     NaN          8.106864     1.024165            NaN            0.0      602.024335                 1.093082    NaN    20.329428        0.711561     1.106940              NaN         1.102846           NaN    4707.956783   7117.786044            2.498009    NaN      NaN           3.659938           0.360824                  1.081209            0.0          0.852077           7.780782               1.289271         0.706476        6.126525            3.623137                 3.222430              3.568136
min       18.000000       NaN            NaN   102.000000                     NaN          1.000000     1.000000            NaN            1.0        1.000000                 1.000000    NaN    30.000000        1.000000     1.000000              NaN         1.000000           NaN    1009.000000   2094.000000            0.000000    NaN      NaN          11.000000           3.000000                  1.000000           80.0          0.000000           0.000000               0.000000         1.000000        0.000000            0.000000                 0.000000              0.000000
25%       30.000000       NaN            NaN   465.000000                     NaN          2.000000     2.000000            NaN            1.0      491.250000                 2.000000    NaN    48.000000        2.000000     1.000000              NaN         2.000000           NaN    2911.000000   8047.000000            1.000000    NaN      NaN          12.000000           3.000000                  2.000000           80.0          0.000000           6.000000               2.000000         2.000000        3.000000            2.000000                 0.000000              2.000000
50%       36.000000       NaN            NaN   802.000000                     NaN          7.000000     3.000000            NaN            1.0     1020.500000                 3.000000    NaN    66.000000        3.000000     2.000000              NaN         3.000000           NaN    4919.000000  14235.500000            2.000000    NaN      NaN          14.000000           3.000000                  3.000000           80.0          1.000000          10.000000               3.000000         3.000000        5.000000            3.000000                 1.000000              3.000000
75%       43.000000       NaN            NaN  1157.000000                     NaN         14.000000     4.000000            NaN            1.0     1555.750000                 4.000000    NaN    83.750000        3.000000     3.000000              NaN         4.000000           NaN    8379.000000  20461.500000            4.000000    NaN      NaN          18.000000           3.000000                  4.000000           80.0          1.000000          15.000000               3.000000         3.000000        9.000000            7.000000                 3.000000              7.000000
max       60.000000       NaN            NaN  1499.000000                     NaN         29.000000     5.000000            NaN            1.0     2068.000000                 4.000000    NaN   100.000000        4.000000     5.000000              NaN         4.000000           NaN   19999.000000  26999.000000            9.000000    NaN      NaN          25.000000           4.000000                  4.000000           80.0          3.000000          40.000000               6.000000         4.000000       40.000000           18.000000                15.000000             17.000000

    Data quality checks indicate:
    
                            Data Quality Report:

                            Missing Values:
                            No missing values detected.

                            Duplicates:
                            Number of duplicate rows: 0

                            Potential Outliers (Numerical Columns):
                            TotalWorkingYears          16
YearsAtCompany             25
YearsInCurrentRole         13
YearsSinceLastPromotion    42
YearsWithCurrManager       14

                            Data Types:
                            Age                          int64
Attrition                   object
BusinessTravel              object
DailyRate                    int64
Department                  object
DistanceFromHome             int64
Education                    int64
EducationField              object
EmployeeCount                int64
EmployeeNumber               int64
EnvironmentSatisfaction      int64
Gender                      object
HourlyRate                   int64
JobInvolvement               int64
JobLevel                     int64
JobRole                     object
JobSatisfaction              int64
MaritalStatus               object
MonthlyIncome                int64
MonthlyRate                  int64
NumCompaniesWorked           int64
Over18                      object
OverTime                    object
PercentSalaryHike            int64
PerformanceRating            int64
RelationshipSatisfaction     int64
StandardHours                int64
StockOptionLevel             int64
TotalWorkingYears            int64
TrainingTimesLastYear        int64
WorkLifeBalance              int64
YearsAtCompany               int64
YearsInCurrentRole           int64
YearsSinceLastPromotion      int64
YearsWithCurrManager         int64
                            

    Based on this information, please identify:
    1. Any apparent patterns or correlations between variables.
    2. Insights into anomalies or outliers.
    3. General predictive insights that the data might suggest.
    4. Recommendations for further data analysis or additional data collection."""

messages = [
    HumanMessage(content=dataset_analysis),
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | llm | StrOutputParser()

print(chain.invoke({}))
